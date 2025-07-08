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
#include <sstream>  // 添加字符串流支持

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
    int32_t intermediate_decode_samples_;  // 替代 intermediate_decode_interval_
    
    // 添加自动清理相关成员
    std::atomic<bool> auto_cleanup_enabled_;
    std::atomic<int> session_timeout_seconds_;
    std::unique_ptr<std::thread> cleanup_thread_;
    std::atomic<bool> stop_cleanup_thread_;
    
    int32_t min_vad_process_samples_;  // 最小VAD处理样本数阈值
    
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

    // 辅助函数：将时间戳向量转换为字符串
    std::string TimestampsToString(const std::vector<float>& timestamps) {
        if (timestamps.empty()) {
            return "[]";
        }
        
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < timestamps.size(); ++i) {
            if (i > 0) oss << ",";
            oss << timestamps[i];
        }
        oss << "]";
        return oss.str();
    }
    
    // 辅助函数：创建包含文本和时间戳的JSON格式结果
    std::string CreateResultJson(const std::string& type, const std::string& text, 
                                const std::vector<float>& timestamps) {
        std::ostringstream oss;
        oss << "{\"type\":\"" << type << "\",\"text\":\"" << text 
            << "\",\"timestamps\":" << TimestampsToString(timestamps) << "}";
        return oss.str();
    }

public:
    SenseVoiceStreamingWrapper() 
        : expected_sample_rate_(16000), window_size_(512), initialized_(false),
          intermediate_decode_samples_(3200),  // 0.2秒 * 16000 = 3200 samples
          auto_cleanup_enabled_(false), session_timeout_seconds_(300),
          stop_cleanup_thread_(false),
          min_vad_process_samples_(1024) {}  // 默认2个窗口
    
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
        
        return "OK";
    }
    
    // 处理特定会话的音频
    std::string ProcessChunkForSession(const std::string& session_id,
                                     const char* audio_data, 
                                     int32_t num_samples, 
                                     bool is_last = false) {
        if (!initialized_) {
            return "{\"type\":\"error\",\"text\":\"Model not initialized\",\"timestamps\":[]}";
        }
        
        // 安全地获取会话的共享指针
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> sessions_lock(sessions_mutex_);
            auto it = sessions_.find(session_id);
            if (it == sessions_.end()) {
                return "{\"type\":\"error\",\"text\":\"Session not found\",\"timestamps\":[]}";
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
        bool has_final_result = false;  // 标记是否有最终结果
        
        // 优化：只有当buffer中有足够数据时才进行VAD处理
        // 避免频繁的小片段VAD计算
        const int32_t min_vad_process_samples = window_size_ * 2;  // 至少2个窗口的数据
        
        // Process VAD on the buffer - 优化版本
        if (session->buffer.size() - session->offset >= min_vad_process_samples || is_last) {
            while (session->offset + window_size_ <= session->buffer.size()) {
                session->vad->AcceptWaveform(session->buffer.data() + session->offset, window_size_);
                if (!session->started && session->vad->IsDetected()) {
                    session->started = true;
                    session->started_time = std::chrono::steady_clock::now();
                    session->last_intermediate_result.clear();
                }
                session->offset += window_size_;
            }
        }

        // Trim buffer if no speech detected for a while
        if (!session->started) {
            if (session->buffer.size() > 10 * window_size_) {
                int keep_samples = 10 * window_size_;
                session->offset -= session->buffer.size() - keep_samples;
                session->buffer = {session->buffer.end() - keep_samples, session->buffer.end()};
            }
        }

        // Process completed VAD segments first (highest priority)
        while (!session->vad->IsEmpty()) {
            auto segment = session->vad->Front();
            session->vad->Pop();

            // 当判断到端点的时候，清除中间结果，因为此时以端点的检测结果为准
            session->last_intermediate_result.clear();
            
            using namespace sherpa_onnx::cxx;
            OfflineStream stream = recognizer_->CreateStream();
            stream.AcceptWaveform(expected_sample_rate_, segment.samples.data(),
                                segment.samples.size());
            recognizer_->Decode(&stream);
            
            OfflineRecognizerResult recognition_result = recognizer_->GetResult(&stream);
            if (!recognition_result.text.empty()) {
                std::string final_result = CreateResultJson("final", recognition_result.text, 
                                                           recognition_result.timestamps);
                if (!result.empty()) {
                    result += " | ";
                }
                result += final_result;
                has_final_result = true;
            }
            
            // Reset state for next segment
            session->buffer.clear();
            session->offset = 0;
            session->started = false;
            session->last_intermediate_result.clear();
        }

        // 改为基于音频数据量的中间解码逻辑
        if (!has_final_result && session->started) {
            // 计算从开始检测到语音后累积的样本数
            int32_t accumulated_samples = session->buffer.size();
            
            // 只要累积的样本数达到设定阈值就进行中间解码，不再限制当前传入的音频片段长度
            if (accumulated_samples >= intermediate_decode_samples_) {
                using namespace sherpa_onnx::cxx;
                OfflineStream stream = recognizer_->CreateStream();
                // 使用完整的buffer进行解码，保持前文信息
                stream.AcceptWaveform(expected_sample_rate_, session->buffer.data(), session->buffer.size());
                recognizer_->Decode(&stream);
                
                OfflineRecognizerResult recognition_result = recognizer_->GetResult(&stream);
                if (!recognition_result.text.empty()) {
                    if (recognition_result.text != session->last_intermediate_result) {
                        result = CreateResultJson("intermediate", recognition_result.text, 
                                                recognition_result.timestamps);
                        session->last_intermediate_result = recognition_result.text;
                    }
                }
                
                // 不要截断buffer，保持完整的上下文
                // 只有当buffer过大时才进行适当的清理
                const int32_t max_buffer_size = expected_sample_rate_ * 20; // 最大保持30秒
                if (session->buffer.size() > max_buffer_size) {
                    // 保留最后10秒的数据
                    int32_t keep_samples = expected_sample_rate_ * 10;
                    std::vector<float> new_buffer(session->buffer.end() - keep_samples, session->buffer.end());
                    session->buffer = std::move(new_buffer);
                    session->offset = std::max(0, session->offset - (accumulated_samples - keep_samples));
                }
            }
        }

        // If is_last is true, force process remaining buffer as final result
        if (is_last && session->started && !session->buffer.empty() && !has_final_result) {
            using namespace sherpa_onnx::cxx;
            OfflineStream stream = recognizer_->CreateStream();
            stream.AcceptWaveform(expected_sample_rate_, session->buffer.data(), session->buffer.size());
            recognizer_->Decode(&stream);
            
            OfflineRecognizerResult recognition_result = recognizer_->GetResult(&stream);
            if (!recognition_result.text.empty()) {
                std::string final_result = CreateResultJson("final", recognition_result.text, 
                                                           recognition_result.timestamps);
                if (!result.empty()) {
                    result += " | ";
                }
                result += final_result;
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
            sessions_.erase(it);
        }
    }
    
    // 获取活跃会话数量
    int GetActiveSessionCount() {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        return sessions_.size();
    }

    // 添加清理过期会话的方法
    void CleanupExpiredSessions(int timeout_seconds = 300) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        auto now = std::chrono::steady_clock::now();
        
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - it->second->last_activity).count();
            
            if (elapsed > timeout_seconds) {
                it = sessions_.erase(it);
            } else {
                ++it;
            }
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
    }
    
    // 停止自动清理
    void StopAutoCleanup() {
        auto_cleanup_enabled_.store(false);
        stop_cleanup_thread_.store(true);
        
        if (cleanup_thread_ && cleanup_thread_->joinable()) {
            cleanup_thread_->join();
            cleanup_thread_.reset();
        }
    }
    
    // 设置会话超时时间
    void SetSessionTimeout(int timeout_seconds) {
        session_timeout_seconds_.store(timeout_seconds);
    }
    
    // 获取当前超时设置
    int GetSessionTimeout() const {
        return session_timeout_seconds_.load();
    }
    
    // 检查自动清理是否启用
    bool IsAutoCleanupEnabled() const {
        return auto_cleanup_enabled_.load();
    }

    // 添加设置中间解码样本数的方法
    void SetIntermediateDecodeSamples(int32_t samples) {
        intermediate_decode_samples_ = samples;
    }
    
    // // 设置中间解码的音频长度（秒）
    // void SetIntermediateDecodeInterval(float seconds) {
    //     intermediate_decode_samples_ = static_cast<int32_t>(seconds * expected_sample_rate_);
    // }
    
    // 获取当前设置的中间解码样本数
    int32_t GetIntermediateDecodeSamples() const {
        return intermediate_decode_samples_;
    }

    // 添加配置VAD处理阈值的方法
    void SetVadProcessThreshold(int32_t min_samples) {
        min_vad_process_samples_ = min_samples;
    }
    
    int32_t GetVadProcessThreshold() const {
        return min_vad_process_samples_;
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
    
    EXPORT void set_intermediate_decode_samples(SenseVoiceStreamingWrapper* wrapper, 
                                               int32_t samples) {
        wrapper->SetIntermediateDecodeSamples(samples);
    }
    
    EXPORT int32_t get_intermediate_decode_samples(SenseVoiceStreamingWrapper* wrapper) {
        return wrapper->GetIntermediateDecodeSamples();
    }
    
    EXPORT void set_vad_process_threshold(SenseVoiceStreamingWrapper* wrapper, 
                                         int32_t min_samples) {
        wrapper->SetVadProcessThreshold(min_samples);
    }
    
    EXPORT int32_t get_vad_process_threshold(SenseVoiceStreamingWrapper* wrapper) {
        return wrapper->GetVadProcessThreshold();
    }
} 