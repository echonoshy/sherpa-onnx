import ctypes
import numpy as np
import os
import wave
import time
import sys
import uuid
import threading
import concurrent.futures
from ctypes import c_char_p, c_int, c_void_p, POINTER

class SenseVoiceMultiSession:
    def __init__(self, so_path=None, debug_mode=False):
        # 设置库路径环境变量
        build_lib_path = "/root/sherpa-onnx/build/lib"
        if os.path.exists(build_lib_path):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if build_lib_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{build_lib_path}:{current_ld_path}"
        
        # 查找库文件
        if so_path is None:
            possible_paths = [
                "/root/sherpa-onnx/build/lib/libsense_voice_streaming_wrapper.so",
                "/root/sherpa-onnx/build/cxx-api-examples/libsense_voice_streaming_wrapper.so",
                "./libsense_voice_streaming_wrapper.so",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    so_path = path
                    break
            
            if so_path is None:
                raise FileNotFoundError("Cannot find libsense_voice_streaming_wrapper.so")
        
        print(f"Loading library from: {so_path}")
        
        try:
            # Load the shared library
            self.lib = ctypes.CDLL(so_path)
            
            # Define function signatures
            self.lib.create_wrapper.restype = c_void_p
            self.lib.destroy_wrapper.argtypes = [c_void_p]
            
            # 原有的简化版本
            self.lib.init_model.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]
            self.lib.init_model.restype = c_int
            
            # 新的参数化版本
            self.lib.init_model_with_params.argtypes = [
                c_void_p, c_char_p, c_char_p, c_char_p,  # wrapper, vad_model, sense_voice_model, tokens
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,  # VAD参数
                c_int, c_int, c_char_p, c_int,  # sample_rate, use_itn, language, num_threads
                c_char_p, c_char_p, c_char_p  # HR参数
            ]
            self.lib.init_model_with_params.restype = c_int
            
            # 多会话API
            self.lib.create_session.argtypes = [c_void_p, c_char_p]
            self.lib.create_session.restype = c_char_p
            
            self.lib.process_chunk_for_session_safe.argtypes = [c_void_p, c_char_p, c_char_p, c_int, c_int, c_char_p, c_int]
            self.lib.process_chunk_for_session_safe.restype = c_int
            
            self.lib.destroy_session.argtypes = [c_void_p, c_char_p]
            self.lib.get_active_session_count.argtypes = [c_void_p]
            self.lib.get_active_session_count.restype = c_int
            self.lib.set_debug_mode.argtypes = [c_void_p, c_int]
            
            self.lib.cleanup_expired_sessions.argtypes = [c_void_p, c_int]
            
            # Create wrapper instance
            self.wrapper = self.lib.create_wrapper()
            
            # Set debug mode
            self.lib.set_debug_mode(self.wrapper, 1 if debug_mode else 0)
            
        except AttributeError as e:
            print(f"Error loading library functions: {e}")
            print("Available symbols in library:")
            os.system(f"nm -D {so_path} | grep -E '(create_wrapper|init_model|create_session|process_chunk)'")
            raise
    
    def __del__(self):
        if hasattr(self, 'wrapper'):
            self.lib.destroy_wrapper(self.wrapper)
    
    def init_model(self, vad_model_path, sense_voice_model_path, tokens_path):
        """Initialize the model with given paths"""
        result = self.lib.init_model(
            self.wrapper,
            vad_model_path.encode('utf-8'),
            sense_voice_model_path.encode('utf-8'),
            tokens_path.encode('utf-8')
        )
        return result == 1
    
    def init_model_with_params(self, vad_model_path, sense_voice_model_path, tokens_path,
                             vad_threshold=0.5,
                             min_silence_duration=0.1,
                             min_speech_duration=0.25,
                             max_speech_duration=8.0,
                             sample_rate=16000,
                             use_itn=True,
                             language="auto",
                             num_threads=1,
                             hr_dict_dir="",
                             hr_lexicon="",
                             hr_rule_fsts=""):
        """Initialize the model with detailed parameters"""
        result = self.lib.init_model_with_params(
            self.wrapper,
            vad_model_path.encode('utf-8'),
            sense_voice_model_path.encode('utf-8'),
            tokens_path.encode('utf-8'),
            ctypes.c_float(vad_threshold),
            ctypes.c_float(min_silence_duration),
            ctypes.c_float(min_speech_duration),
            ctypes.c_float(max_speech_duration),
            sample_rate,
            1 if use_itn else 0,
            language.encode('utf-8'),
            num_threads,
            hr_dict_dir.encode('utf-8'),
            hr_lexicon.encode('utf-8'),
            hr_rule_fsts.encode('utf-8')
        )
        return result == 1
    
    def create_session(self, session_id=None):
        """Create a new session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        result = self.lib.create_session(self.wrapper, session_id.encode('utf-8'))
        result_str = result.decode('utf-8') if result else ""
        
        if result_str == "OK":
            return session_id
        else:
            raise RuntimeError(f"Failed to create session: {result_str}")
    
    def process_chunk_for_session(self, session_id, audio_bytes, is_last=False):
        """Process audio chunk for specific session"""
        num_samples = len(audio_bytes) // 2
        
        # 准备结果缓冲区
        result_buffer = ctypes.create_string_buffer(4096)
        
        result_len = self.lib.process_chunk_for_session_safe(
            self.wrapper,
            session_id.encode('utf-8'),
            audio_bytes,
            num_samples,
            1 if is_last else 0,
            result_buffer,
            4096
        )
        
        if result_len < 0:
            return "Buffer too small"
        
        return result_buffer.value.decode('utf-8') if result_len > 0 else ""
    
    def destroy_session(self, session_id):
        """Destroy a session"""
        self.lib.destroy_session(self.wrapper, session_id.encode('utf-8'))
    
    def get_active_session_count(self):
        """Get number of active sessions"""
        return self.lib.get_active_session_count(self.wrapper)
    
    def cleanup_expired_sessions(self, timeout_seconds=300):
        """清理过期会话"""
        self.lib.cleanup_expired_sessions(self.wrapper, timeout_seconds)
    
    def cleanup(self):
        """主动清理所有资源"""
        if hasattr(self, 'wrapper') and self.wrapper:
            self.lib.destroy_wrapper(self.wrapper)
            self.wrapper = None
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """with语句结束时自动清理"""
        self.cleanup()


def process_audio_for_session_concurrent(recognizer, session_id, audio_path, client_name="Client"):
    """并发处理特定会话的音频"""
    print(f"[{client_name}] Starting audio processing for session {session_id}")
    
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            chunk_duration = 0.1  # 0.1 second chunks
            frames_per_chunk = int(chunk_duration * wav_file.getframerate())
            
            total_chunks = 0
            start_time = time.time()
            
            while True:
                audio_chunk = wav_file.readframes(frames_per_chunk)
                if not audio_chunk:
                    # 文件结束，发送最后一块
                    result = recognizer.process_chunk_for_session(session_id, b'\x00' * 1600, is_last=True)
                    if result:
                        print(f"[{client_name}] Final: {result}")
                    break
                
                # 处理音频块
                result = recognizer.process_chunk_for_session(session_id, audio_chunk)
                if result:
                    elapsed_time = time.time() - start_time
                    print(f"[{client_name}] [{elapsed_time:.2f}s] {result}")
                
                total_chunks += 1
                
                # 模拟实时处理延迟
                time.sleep(0.02)
            
            processing_time = time.time() - start_time
            print(f"[{client_name}] Completed processing {total_chunks} chunks in {processing_time:.2f}s")
            
    except Exception as e:
        print(f"[{client_name}] Error processing audio: {e}")
    
    return f"{client_name} completed"


def simulate_concurrent_clients(recognizer, num_clients=3, audio_path="/root/sherpa-onnx/audios/girl-zh.wav"):
    """模拟多个并发客户端"""
    print(f"\n=== 模拟 {num_clients} 个并发客户端 ===")
    
    # 创建会话
    sessions = []
    for i in range(num_clients):
        session_id = f"client_{i+1}_session"
        try:
            session = recognizer.create_session(session_id)
            sessions.append((session, f"Client-{i+1}"))
            print(f"Created session for Client-{i+1}: {session}")
        except Exception as e:
            print(f"Failed to create session for Client-{i+1}: {e}")
            continue
    
    print(f"Active sessions: {recognizer.get_active_session_count()}")
    
    # 使用线程池并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
        # 提交所有任务
        futures = []
        for session_id, client_name in sessions:
            future = executor.submit(
                process_audio_for_session_concurrent,
                recognizer,
                session_id,
                audio_path,
                client_name
            )
            futures.append((future, session_id, client_name))
        
        print(f"\n所有 {len(futures)} 个客户端开始并发处理...")
        
        # 等待所有任务完成
        for future, session_id, client_name in futures:
            try:
                result = future.result(timeout=60)  # 60秒超时
                print(f"✓ {result}")
            except concurrent.futures.TimeoutError:
                print(f"✗ {client_name} 处理超时")
            except Exception as e:
                print(f"✗ {client_name} 处理出错: {e}")
    
    print(f"\n并发处理完成，当前活跃会话数: {recognizer.get_active_session_count()}")
    
    # 清理会话
    for session_id, client_name in sessions:
        try:
            recognizer.destroy_session(session_id)
            print(f"Cleaned up session for {client_name}")
        except Exception as e:
            print(f"Error cleaning up session for {client_name}: {e}")
    
    print(f"最终活跃会话数: {recognizer.get_active_session_count()}")


def test_concurrent_sessions():
    """测试并发会话功能"""
    with SenseVoiceMultiSession(debug_mode=True) as recognizer:
        # 初始化模型
        success = recognizer.init_model_with_params(
            vad_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/silero_vad.onnx",
            sense_voice_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
            tokens_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
            vad_threshold=0.45,
            language="zh",
            num_threads=4  # 增加线程数以支持并发
        )
        
        if not success:
            print("Failed to initialize model")
            return
        
        print("Model initialized successfully with custom parameters")
        
        # 测试不同数量的并发客户端
        test_cases = [
            {"num_clients": 2, "audio_path": "/root/sherpa-onnx/audios/girl-zh.wav"},
            {"num_clients": 3, "audio_path": "/root/sherpa-onnx/audios/girl-zh.wav"},
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"测试用例 {i+1}: {test_case['num_clients']} 个并发客户端")
            print(f"{'='*60}")
            
            simulate_concurrent_clients(
                recognizer,
                num_clients=test_case['num_clients'],
                audio_path=test_case['audio_path']
            )
            
            # 测试间隔
            if i < len(test_cases) - 1:
                print("\n等待 3 秒后进行下一个测试...")
                time.sleep(3)


def test_stress_concurrent():
    """压力测试 - 更多并发客户端"""
    with SenseVoiceMultiSession(debug_mode=False) as recognizer:  # 关闭调试模式减少输出
        success = recognizer.init_model_with_params(
            vad_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/silero_vad.onnx",
            sense_voice_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
            tokens_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
            vad_threshold=0.45,
            language="zh",
            num_threads=8  # 更多线程
        )
        
        if not success:
            print("Failed to initialize model")
            return
        
        print("=== 压力测试：10个并发客户端 ===")
        
        start_time = time.time()
        simulate_concurrent_clients(
            recognizer,
            num_clients=10,
            audio_path="/root/sherpa-onnx/audios/girl-zh.wav"
        )
        total_time = time.time() - start_time
        
        print(f"\n压力测试完成，总耗时: {total_time:.2f}秒")


if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 基础并发测试 (2-3个客户端)")
    print("2. 压力测试 (10个客户端)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "2":
        test_stress_concurrent()
    else:
        test_concurrent_sessions() 