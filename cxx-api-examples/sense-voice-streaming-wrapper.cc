// cxx-api-examples/sense-voice-streaming-wrapper.cc


#include <chrono>
#include <iostream>
#include <mutex>
#include <vector>
#include <queue>
#include <memory>
#include <unordered_map>
#include <string>
#include <cstring>  // for strcpy
#include <thread>   // 添加线程支持
#include <atomic>   // 添加原子操作支持

#include "sherpa-onnx/c-api/cxx-api.h"

// 添加符号导出属性
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

class SenseVoiceStreamingWrapper {
private:
    // 共享的模型实例（线程安全）
    std::unique_ptr<sherpa_onnx::cxx::OfflineRecognizer> recognizer_;
    
    // VAD配置模板
    sherpa_onnx::cxx::VadModelConfig vad_config_template_;
    
    // 每个会话的状态
    struct SessionState {
        std::unique_ptr<sherpa_onnx::cxx::VoiceActivityDetector> vad;
        std::vector<float> buffer;
        int32_t offset;
        bool started;
        std::chrono::steady_clock::time_point started_time;
        std::string last_intermediate_result;
        std::mutex session_mutex;
        std::chrono::steady_clock::time_point last_activity;
        
        SessionState() : offset(0), started(false), last_activity(std::chrono::steady_clock::now()) {}
    };
    
    std::unordered_map<std::string, std::shared_ptr<SessionState>> sessions_;
    std::mutex sessions_mutex_;
    
    int32_t expected_sample_rate_;
    int32_t window_size_;
    bool initialized_;
    bool debug_mode_;
    float intermediate_decode_interval_;
    
    // 添加默认会话ID用于向后兼容
    std::string default_session_id_;

    // 添加自动清理相关成员
    std::atomic<bool> auto_cleanup_enabled_;
    std::atomic<int> session_timeout_seconds_;
    std::unique_ptr<std::thread> cleanup_thread_;
    std::atomic<bool> stop_cleanup_thread_;
    
    // 自动清理线程函数
    void AutoCleanupThread() {
        while (!stop_cleanup_thread_.load()) {
            // 每30秒检查一次过期会话
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
            if (auto_cleanup_enabled_.load()) {
                CleanupExpiredSessions(session_timeout_seconds_.load());
            }
        }
    }

public:
    SenseVoiceStreamingWrapper() 
        : expected_sample_rate_(16000), window_size_(512), initialized_(false),
          debug_mode_(false), intermediate_decode_interval_(0.2),
          auto_cleanup_enabled_(false), session_timeout_seconds_(300),
          stop_cleanup_thread_(false) {}
    
    ~SenseVoiceStreamingWrapper() {
        // 停止自动清理线程
        StopAutoCleanup();
    }

    bool InitModel(const char* vad_model_path, const char* sense_voice_model_path, 
                   const char* tokens_path,
                   float vad_threshold = 0.5,
                   float min_silence_duration = 0.1,
                   float min_speech_duration = 0.25,
                   float max_speech_duration = 8.0,
                   int sample_rate = 16000,
                   bool use_itn = true,
                   const char* language = "auto",
                   int num_threads = 1,
                   const char* hr_dict_dir = "",
                   const char* hr_lexicon = "",
                   const char* hr_rule_fsts = "") {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        
        try {
            // 更新采样率
            expected_sample_rate_ = sample_rate;
            
            // 保存VAD配置模板
            using namespace sherpa_onnx::cxx;
            vad_config_template_.silero_vad.model = vad_model_path;
            vad_config_template_.silero_vad.threshold = vad_threshold;
            vad_config_template_.silero_vad.min_silence_duration = min_silence_duration;
            vad_config_template_.silero_vad.min_speech_duration = min_speech_duration;
            vad_config_template_.silero_vad.max_speech_duration = max_speech_duration;
            vad_config_template_.sample_rate = expected_sample_rate_;
            vad_config_template_.debug = false;

            // 获取VAD的window_size
            window_size_ = 512;

            // Create Offline Recognizer (共享)
            OfflineRecognizerConfig recognizer_config;
            recognizer_config.model_config.sense_voice.model = sense_voice_model_path;
            recognizer_config.model_config.sense_voice.use_itn = use_itn;
            recognizer_config.model_config.sense_voice.language = language;
            recognizer_config.model_config.tokens = tokens_path;
            recognizer_config.model_config.num_threads = num_threads;
            recognizer_config.model_config.debug = false;

            // HR 参数化
            recognizer_config.hr.dict_dir = hr_dict_dir;
            recognizer_config.hr.lexicon = hr_lexicon;
            recognizer_config.hr.rule_fsts = hr_rule_fsts;

            auto recognizer = OfflineRecognizer::Create(recognizer_config);
            if (!recognizer.Get()) {
                std::cerr << "Failed to create recognizer\n";
                return false;
            }
            recognizer_ = std::make_unique<OfflineRecognizer>(std::move(recognizer));

            initialized_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing model: " << e.what() << std::endl;
            return false;
        }
    }

    // 创建新会话
    std::string CreateSession(const std::string& session_id) {
        if (!initialized_) {
            return "Model not initialized";
        }
        
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        
        if (sessions_.find(session_id) != sessions_.end()) {
            return "Session already exists";
        }
        
        auto session = std::make_shared<SessionState>();
        
        // 为每个会话创建独立的VAD实例
        using namespace sherpa_onnx::cxx;
        auto vad = VoiceActivityDetector::Create(vad_config_template_, 20);
        if (!vad.Get()) {
            return "Failed to create VAD for session";
        }
        
        session->vad = std::make_unique<VoiceActivityDetector>(std::move(vad));
        
        sessions_[session_id] = session;
        
        if (debug_mode_) {
            std::cout << "[DEBUG] Created session: " << session_id << std::endl;
        }
        
        return "OK";
    }
    
    // 处理特定会话的音频
    std::string ProcessChunkForSession(const std::string& session_id,
                                     const char* audio_data, 
                                     int32_t num_samples, 
                                     bool is_last = false) {
        if (!initialized_) {
            return "Model not initialized";
        }
        
        // 安全地获取会话的共享指针
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> sessions_lock(sessions_mutex_);
            auto it = sessions_.find(session_id);
            if (it == sessions_.end()) {
                return "Session not found";
            }
            session = it->second;  // 获取共享指针
        }
        
        // 现在可以安全地使用session，即使它从map中被删除
        std::lock_guard<std::mutex> session_lock(session->session_mutex);
        
        // 更新最后活动时间
        session->last_activity = std::chrono::steady_clock::now();
        
        // Convert bytes to float samples (assuming 16-bit PCM)
        const int16_t* samples_int16 = reinterpret_cast<const int16_t*>(audio_data);
        std::vector<float> chunk(num_samples);
        for (int32_t i = 0; i < num_samples; ++i) {
            chunk[i] = static_cast<float>(samples_int16[i]) / 32768.0f;
        }
        
        // Add chunk to buffer (simulating streaming input)
        session->buffer.insert(session->buffer.end(), chunk.begin(), chunk.end());
        
        std::string result;
        
        // Process VAD on the buffer
        while (session->offset + window_size_ < session->buffer.size()) {
            session->vad->AcceptWaveform(session->buffer.data() + session->offset, window_size_);
            if (!session->started && session->vad->IsDetected()) {
                session->started = true;
                session->started_time = std::chrono::steady_clock::now();
                session->last_intermediate_result.clear();
                if (debug_mode_) {
                    std::cout << "[DEBUG] Session " << session_id << ": Speech detected" << std::endl;
                }
            }
            session->offset += window_size_;
        }

        // Trim buffer if no speech detected for a while
        if (!session->started) {
            if (session->buffer.size() > 10 * window_size_) {
                int keep_samples = 10 * window_size_;
                session->offset -= session->buffer.size() - keep_samples;
                session->buffer = {session->buffer.end() - keep_samples, session->buffer.end()};
            }
        }

        // Intermediate decoding every 0.2s during speech
        if (session->started) {
            auto current_time = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - session->started_time).count() / 1000.0f;
            
            if (elapsed > intermediate_decode_interval_) {
                using namespace sherpa_onnx::cxx;
                OfflineStream stream = recognizer_->CreateStream();
                stream.AcceptWaveform(expected_sample_rate_, session->buffer.data(), session->buffer.size());
                recognizer_->Decode(&stream);
                
                OfflineRecognizerResult recognition_result = recognizer_->GetResult(&stream);
                if (!recognition_result.text.empty()) {
                    if (recognition_result.text != session->last_intermediate_result) {
                        result = "[intermediate] " + recognition_result.text;
                        session->last_intermediate_result = recognition_result.text;
                        if (debug_mode_) {
                            std::cout << "[DEBUG] Session " << session_id << " intermediate: '" 
                                      << recognition_result.text << "'" << std::endl;
                        }
                    }
                }
                session->started_time = current_time;
            }
        }

        // Process completed VAD segments
        while (!session->vad->IsEmpty()) {
            auto segment = session->vad->Front();
            session->vad->Pop();
            
            using namespace sherpa_onnx::cxx;
            OfflineStream stream = recognizer_->CreateStream();
            stream.AcceptWaveform(expected_sample_rate_, segment.samples.data(),
                                segment.samples.size());
            recognizer_->Decode(&stream);
            
            OfflineRecognizerResult recognition_result = recognizer_->GetResult(&stream);
            if (!recognition_result.text.empty()) {
                std::string final_result = "[final] " + recognition_result.text;
                if (!result.empty()) {
                    result += " | ";
                }
                result += final_result;
                if (debug_mode_) {
                    std::cout << "[DEBUG] Session " << session_id << " final: '" 
                              << recognition_result.text << "'" << std::endl;
                }
            }
            
            // Reset state for next segment
            session->buffer.clear();
            session->offset = 0;
            session->started = false;
            session->last_intermediate_result.clear();
        }

        // If is_last is true, force process remaining buffer as final result
        if (is_last && session->started && !session->buffer.empty()) {
            using namespace sherpa_onnx::cxx;
            OfflineStream stream = recognizer_->CreateStream();
            stream.AcceptWaveform(expected_sample_rate_, session->buffer.data(), session->buffer.size());
            recognizer_->Decode(&stream);
            
            OfflineRecognizerResult recognition_result = recognizer_->GetResult(&stream);
            if (!recognition_result.text.empty()) {
                std::string final_result = "[final] " + recognition_result.text;
                if (!result.empty()) {
                    result += " | ";
                }
                result += final_result;
                if (debug_mode_) {
                    std::cout << "[DEBUG] Session " << session_id << " forced final: '" 
                              << recognition_result.text << "'" << std::endl;
                }
            }
            
            // Reset state
            session->buffer.clear();
            session->offset = 0;
            session->started = false;
            session->last_intermediate_result.clear();
        }

        return result;
    }
    
    // 删除会话
    void DestroySession(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        auto it = sessions_.find(session_id);
        if (it != sessions_.end()) {
            if (debug_mode_) {
                std::cout << "[DEBUG] Destroyed session: " << session_id << std::endl;
            }
            sessions_.erase(it);
        }
    }
    
    // 获取活跃会话数量
    int GetActiveSessionCount() {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        return sessions_.size();
    }

    void SetDebugMode(bool debug) {
        debug_mode_ = debug;
    }

    // 添加清理过期会话的方法
    void CleanupExpiredSessions(int timeout_seconds = 300) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        auto now = std::chrono::steady_clock::now();
        
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - it->second->last_activity).count();
            
            if (elapsed > timeout_seconds) {
                if (debug_mode_) {
                    std::cout << "[DEBUG] Cleaning up expired session: " << it->first << std::endl;
                }
                it = sessions_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 向后兼容的单会话API
    std::string ProcessChunk(const char* audio_data, int32_t num_samples, bool is_last = false) {
        // 如果没有默认会话，创建一个
        if (default_session_id_.empty()) {
            default_session_id_ = "default_session";
            std::string create_result = CreateSession(default_session_id_);
            if (create_result != "OK") {
                return "Failed to create default session: " + create_result;
            }
        }
        
        return ProcessChunkForSession(default_session_id_, audio_data, num_samples, is_last);
    }
    
    std::string FlushRemaining() {
        if (default_session_id_.empty()) {
            return "";
        }
        
        // 发送空音频块并标记为最后一块
        std::vector<int16_t> silence(1600, 0);  // 0.1秒的静音
        return ProcessChunkForSession(default_session_id_, 
                                    reinterpret_cast<const char*>(silence.data()), 
                                    silence.size(), 
                                    true);
    }
    
    void Reset() {
        if (!default_session_id_.empty()) {
            DestroySession(default_session_id_);
            default_session_id_.clear();
        }
    }

    // 启动自动清理
    void StartAutoCleanup(int timeout_seconds = 300, int check_interval_seconds = 30) {
        if (cleanup_thread_ && cleanup_thread_->joinable()) {
            return; // 已经启动
        }
        
        session_timeout_seconds_.store(timeout_seconds);
        auto_cleanup_enabled_.store(true);
        stop_cleanup_thread_.store(false);
        
        cleanup_thread_ = std::make_unique<std::thread>([this, check_interval_seconds]() {
            while (!stop_cleanup_thread_.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(check_interval_seconds));
                
                if (auto_cleanup_enabled_.load()) {
                    CleanupExpiredSessions(session_timeout_seconds_.load());
                }
            }
        });
        
        if (debug_mode_) {
            std::cout << "[DEBUG] Auto cleanup started with timeout: " << timeout_seconds 
                      << "s, check interval: " << check_interval_seconds << "s" << std::endl;
        }
    }
    
    // 停止自动清理
    void StopAutoCleanup() {
        auto_cleanup_enabled_.store(false);
        stop_cleanup_thread_.store(true);
        
        if (cleanup_thread_ && cleanup_thread_->joinable()) {
            cleanup_thread_->join();
            cleanup_thread_.reset();
        }
        
        if (debug_mode_) {
            std::cout << "[DEBUG] Auto cleanup stopped" << std::endl;
        }
    }
    
    // 设置会话超时时间
    void SetSessionTimeout(int timeout_seconds) {
        session_timeout_seconds_.store(timeout_seconds);
        if (debug_mode_) {
            std::cout << "[DEBUG] Session timeout set to: " << timeout_seconds << "s" << std::endl;
        }
    }
    
    // 获取当前超时设置
    int GetSessionTimeout() const {
        return session_timeout_seconds_.load();
    }
    
    // 检查自动清理是否启用
    bool IsAutoCleanupEnabled() const {
        return auto_cleanup_enabled_.load();
    }
};

// C接口
extern "C" {
    EXPORT SenseVoiceStreamingWrapper* create_wrapper() {
        return new SenseVoiceStreamingWrapper();
    }
    
    EXPORT void destroy_wrapper(SenseVoiceStreamingWrapper* wrapper) {
        delete wrapper;
    }
    
    EXPORT int init_model_with_params(SenseVoiceStreamingWrapper* wrapper, 
                                    const char* vad_model_path,
                                    const char* sense_voice_model_path,
                                    const char* tokens_path,
                                    float vad_threshold,
                                    float min_silence_duration,
                                    float min_speech_duration,
                                    float max_speech_duration,
                                    int sample_rate,
                                    int use_itn,
                                    const char* language,
                                    int num_threads,
                                    const char* hr_dict_dir,
                                    const char* hr_lexicon,
                                    const char* hr_rule_fsts) {
        return wrapper->InitModel(vad_model_path, sense_voice_model_path, tokens_path,
                                vad_threshold, min_silence_duration, min_speech_duration,
                                max_speech_duration, sample_rate, use_itn != 0, language,
                                num_threads, hr_dict_dir, hr_lexicon, hr_rule_fsts) ? 1 : 0;
    }
    
    // 保留原有的简化版本用于向后兼容
    EXPORT int init_model(SenseVoiceStreamingWrapper* wrapper, 
                         const char* vad_model_path,
                         const char* sense_voice_model_path,
                         const char* tokens_path) {
        return wrapper->InitModel(vad_model_path, sense_voice_model_path, tokens_path) ? 1 : 0;
    }
    
    EXPORT const char* create_session(SenseVoiceStreamingWrapper* wrapper, 
                                    const char* session_id) {
        static thread_local std::string result;
        result = wrapper->CreateSession(session_id);
        return result.c_str();
    }
    
    EXPORT int process_chunk_for_session_safe(SenseVoiceStreamingWrapper* wrapper,
                                             const char* session_id,
                                             const char* audio_data,
                                             int num_samples,
                                             int is_last,
                                             char* result_buffer,
                                             int buffer_size) {
        std::string result = wrapper->ProcessChunkForSession(session_id, audio_data, num_samples, is_last != 0);
        
        if (result.length() >= buffer_size) {
            return -1;  // 缓冲区太小
        }
        
        strcpy(result_buffer, result.c_str());
        return result.length();
    }
    
    EXPORT void destroy_session(SenseVoiceStreamingWrapper* wrapper, 
                              const char* session_id) {
        wrapper->DestroySession(session_id);
    }
    
    EXPORT int get_active_session_count(SenseVoiceStreamingWrapper* wrapper) {
        return wrapper->GetActiveSessionCount();
    }
    
    EXPORT void set_debug_mode(SenseVoiceStreamingWrapper* wrapper, int debug) {
        wrapper->SetDebugMode(debug != 0);
    }
    
    // 向后兼容的单会话API
    EXPORT const char* process_chunk(SenseVoiceStreamingWrapper* wrapper,
                                   const char* audio_data,
                                   int num_samples,
                                   int is_last) {
        static thread_local std::string result;
        result = wrapper->ProcessChunk(audio_data, num_samples, is_last != 0);
        return result.c_str();
    }
    
    EXPORT const char* flush_remaining(SenseVoiceStreamingWrapper* wrapper) {
        static thread_local std::string result;
        result = wrapper->FlushRemaining();
        return result.c_str();
    }
    
    EXPORT void reset_wrapper(SenseVoiceStreamingWrapper* wrapper) {
        wrapper->Reset();
    }
    
    EXPORT void cleanup_expired_sessions(SenseVoiceStreamingWrapper* wrapper, int timeout_seconds) {
        wrapper->CleanupExpiredSessions(timeout_seconds);
    }
    
    EXPORT void start_auto_cleanup(SenseVoiceStreamingWrapper* wrapper, 
                                  int timeout_seconds, 
                                  int check_interval_seconds) {
        wrapper->StartAutoCleanup(timeout_seconds, check_interval_seconds);
    }
    
    EXPORT void stop_auto_cleanup(SenseVoiceStreamingWrapper* wrapper) {
        wrapper->StopAutoCleanup();
    }
    
    EXPORT void set_session_timeout(SenseVoiceStreamingWrapper* wrapper, 
                                   int timeout_seconds) {
        wrapper->SetSessionTimeout(timeout_seconds);
    }
    
    EXPORT int get_session_timeout(SenseVoiceStreamingWrapper* wrapper) {
        return wrapper->GetSessionTimeout();
    }
    
    EXPORT int is_auto_cleanup_enabled(SenseVoiceStreamingWrapper* wrapper) {
        return wrapper->IsAutoCleanupEnabled() ? 1 : 0;
    }
} 