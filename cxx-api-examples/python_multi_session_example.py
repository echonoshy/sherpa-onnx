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
        # Set library path environment variable
        build_lib_path = "/root/sherpa-onnx/build/lib"
        if os.path.exists(build_lib_path):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if build_lib_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{build_lib_path}:{current_ld_path}"
        
        # Find library file
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
            
            # Original simplified version
            self.lib.init_model.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]
            self.lib.init_model.restype = c_int
            
            # New parameterized version
            self.lib.init_model_with_params.argtypes = [
                c_void_p, c_char_p, c_char_p, c_char_p,  # wrapper, vad_model, sense_voice_model, tokens
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,  # VAD parameters
                c_int, c_int, c_char_p, c_int,  # sample_rate, use_itn, language, num_threads
                c_char_p, c_char_p, c_char_p  # HR parameters
            ]
            self.lib.init_model_with_params.restype = c_int
            
            # Multi-session API
            self.lib.create_session.argtypes = [c_void_p, c_char_p]
            self.lib.create_session.restype = c_char_p
            
            self.lib.process_chunk_for_session_safe.argtypes = [c_void_p, c_char_p, c_char_p, c_int, c_int, c_char_p, c_int]
            self.lib.process_chunk_for_session_safe.restype = c_int
            
            self.lib.destroy_session.argtypes = [c_void_p, c_char_p]
            self.lib.get_active_session_count.argtypes = [c_void_p]
            self.lib.get_active_session_count.restype = c_int
            self.lib.set_debug_mode.argtypes = [c_void_p, c_int]
            
            self.lib.cleanup_expired_sessions.argtypes = [c_void_p, c_int]
            
            # Add auto cleanup function signatures
            self.lib.start_auto_cleanup.argtypes = [c_void_p, c_int, c_int]
            self.lib.stop_auto_cleanup.argtypes = [c_void_p]
            self.lib.set_session_timeout.argtypes = [c_void_p, c_int]
            self.lib.get_session_timeout.argtypes = [c_void_p]
            self.lib.get_session_timeout.restype = c_int
            self.lib.is_auto_cleanup_enabled.argtypes = [c_void_p]
            self.lib.is_auto_cleanup_enabled.restype = c_int
            
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
        
        # Prepare result buffer
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
        """Clean up expired sessions"""
        self.lib.cleanup_expired_sessions(self.wrapper, timeout_seconds)
    
    def start_auto_cleanup(self, timeout_seconds=300, check_interval_seconds=30):
        """Start automatic cleanup of expired sessions"""
        self.lib.start_auto_cleanup(self.wrapper, timeout_seconds, check_interval_seconds)
        if hasattr(self, '_auto_cleanup_started'):
            return
        self._auto_cleanup_started = True
        print(f"🔧 Auto cleanup started (timeout: {timeout_seconds}s, check interval: {check_interval_seconds}s)")
    
    def stop_auto_cleanup(self):
        """Stop automatic cleanup"""
        self.lib.stop_auto_cleanup(self.wrapper)
        if hasattr(self, '_auto_cleanup_started'):
            print("🔧 Auto cleanup stopped")
            delattr(self, '_auto_cleanup_started')
    
    def set_session_timeout(self, timeout_seconds):
        """Set session timeout in seconds"""
        self.lib.set_session_timeout(self.wrapper, timeout_seconds)
        print(f"🔧 Session timeout set to {timeout_seconds}s")
    
    def get_session_timeout(self):
        """Get current session timeout setting"""
        return self.lib.get_session_timeout(self.wrapper)
    
    def is_auto_cleanup_enabled(self):
        """Check if auto cleanup is enabled"""
        return self.lib.is_auto_cleanup_enabled(self.wrapper) != 0
    
    def cleanup(self):
        """Actively clean up all resources"""
        # Stop auto cleanup first
        if hasattr(self, '_auto_cleanup_started'):
            self.stop_auto_cleanup()
        
        if hasattr(self, 'wrapper') and self.wrapper:
            self.lib.destroy_wrapper(self.wrapper)
            self.wrapper = None
    
    def __enter__(self):
        """Support with statement"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatic cleanup when with statement ends"""
        self.cleanup()


def process_audio_for_session_streaming(recognizer, session_id, audio_path, client_name="Client"):
    """Process audio for specific session using streaming approach"""
    print(f"[{client_name}] Starting streaming processing for session {session_id}")
    
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            # Check audio parameters
            assert wav_file.getnchannels() == 1, "Only mono audio is supported"
            assert wav_file.getsampwidth() == 2, "Only 16-bit audio is supported"
            file_sample_rate = wav_file.getframerate()
            
            print(f"[{client_name}] Audio: {file_sample_rate}Hz, starting streaming...")
            print(f"[{client_name}] " + "=" * 50)
            
            # Process audio chunks
            chunk_duration = 0.1  # 0.1 second chunks
            frames_per_chunk = int(chunk_duration * file_sample_rate)
            
            total_frames = 0
            current_intermediate = ""
            start_time = time.time()
            
            while True:
                audio_chunk = wav_file.readframes(frames_per_chunk)
                if not audio_chunk:
                    # End of file, send final chunk
                    final_result = recognizer.process_chunk_for_session(session_id, b'\x00' * 1600, is_last=True)
                    if final_result and "[final]" in final_result:
                        final_text = final_result.split("[final]")[-1].strip()
                        current_time = total_frames / file_sample_rate
                        if current_intermediate:
                            print()  # Clear intermediate line
                        print(f"[{client_name}] [{current_time:6.2f}s] (✓ Final): {final_text}")
                    break
                
                # Process current audio chunk
                result = recognizer.process_chunk_for_session(session_id, audio_chunk)
                
                if result:
                    current_time = total_frames / file_sample_rate
                    
                    # Parse result type
                    if "[intermediate]" in result:
                        # Intermediate result - real-time display
                        text = result.replace("[intermediate] ", "").strip()
                        if text != current_intermediate:
                            current_intermediate = text
                            print(f"\r[{client_name}] [{current_time:6.2f}s] (Recognizing): {text:<60}", end="", flush=True)
                    
                    elif "[final]" in result or result.strip():
                        # Final result - complete content with newline
                        if current_intermediate:
                            print()  # Clear intermediate line
                        
                        # Extract final text
                        if "[final]" in result:
                            final_text = result.split("[final]")[-1].strip()
                        else:
                            final_text = result.strip()
                        
                        if final_text:
                            print(f"[{client_name}] [{current_time:6.2f}s] (✓ Result): {final_text}")
                        
                        current_intermediate = ""  # Clear intermediate result
                
                total_frames += frames_per_chunk
                time.sleep(0.02)  # Simulate real-time processing delay
            
            # Ensure final newline
            if current_intermediate:
                print()
            
            processing_time = time.time() - start_time
            print(f"[{client_name}] " + "=" * 50)
            print(f"[{client_name}] Completed in {processing_time:.2f}s")
            
    except Exception as e:
        print(f"[{client_name}] Error processing audio: {e}")
    
    return f"{client_name} completed"


def test_concurrent_sessions(recognizer, num_clients=3, audio_path="/root/sherpa-onnx/audios/girl-zh.wav"):
    """Test concurrent sessions with simplified approach"""
    print(f"\n=== Testing {num_clients} Concurrent Clients ===")
    
    # Start auto cleanup before creating sessions
    recognizer.start_auto_cleanup(timeout_seconds=180, check_interval_seconds=20)
    print(f"🔧 Auto cleanup enabled with 180s timeout")
    
    # Create sessions
    sessions = []
    for i in range(num_clients):
        session_id = f"client_{i+1}_session"
        try:
            session = recognizer.create_session(session_id)
            sessions.append((session, f"Client-{i+1}"))
            print(f"✓ Created session for Client-{i+1}: {session}")
        except Exception as e:
            print(f"✗ Failed to create session for Client-{i+1}: {e}")
            continue
    
    print(f"Active sessions: {recognizer.get_active_session_count()}")
    
    # Process concurrently with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
        futures = []
        for session_id, client_name in sessions:
            future = executor.submit(
                process_audio_for_session_streaming,
                recognizer,
                session_id,
                audio_path,
                client_name
            )
            futures.append((future, session_id, client_name))
        
        print(f"\nAll {len(futures)} clients started concurrent processing...")
        
        # Wait for all tasks to complete
        for future, session_id, client_name in futures:
            try:
                result = future.result(timeout=60)  # 60 second timeout
                print(f"✓ {result}")
            except concurrent.futures.TimeoutError:
                print(f"✗ {client_name} processing timeout")
            except Exception as e:
                print(f"✗ {client_name} processing error: {e}")
    
    print(f"\nConcurrent processing completed. Active sessions: {recognizer.get_active_session_count()}")
    
    # Cleanup sessions
    for session_id, client_name in sessions:
        try:
            recognizer.destroy_session(session_id)
            print(f"✓ Cleaned up session for {client_name}")
        except Exception as e:
            print(f"✗ Error cleaning up session for {client_name}: {e}")
    
    print(f"Final active sessions: {recognizer.get_active_session_count()}")
    
    # Show auto cleanup status
    print(f"🔧 Auto cleanup enabled: {recognizer.is_auto_cleanup_enabled()}")
    print(f"🔧 Session timeout: {recognizer.get_session_timeout()}s")


def run_basic_test():
    """Run basic concurrent session test"""
    with SenseVoiceMultiSession(debug_mode=True) as recognizer:
        # Initialize model
        success = recognizer.init_model_with_params(
            vad_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/silero_vad.onnx",
            sense_voice_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
            tokens_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
            vad_threshold=0.45,
            language="zh",
            num_threads=4
        )
        
        if not success:
            print("Failed to initialize model")
            return
        
        print("✅ Model initialized successfully")
        
        # Enable auto cleanup for the entire test
        recognizer.start_auto_cleanup(timeout_seconds=120, check_interval_seconds=15)
        
        # Test with 2 and 3 clients
        for num_clients in [2, 3]:
            print(f"\n{'='*60}")
            print(f"Test Case: {num_clients} Concurrent Clients")
            print(f"{'='*60}")
            
            test_concurrent_sessions(
                recognizer,
                num_clients=num_clients,
                audio_path="/root/sherpa-onnx/audios/girl-zh.wav"
            )
            
            if num_clients < 3:
                print("\nWaiting 2 seconds before next test...")
                time.sleep(2)
        
        # Auto cleanup will be stopped automatically when exiting the 'with' block


def run_stress_test():
    """Run stress test with more concurrent clients"""
    with SenseVoiceMultiSession(debug_mode=False) as recognizer:
        success = recognizer.init_model_with_params(
            vad_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/silero_vad.onnx",
            sense_voice_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
            tokens_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
            vad_threshold=0.45,
            language="zh",
            num_threads=8
        )
        
        if not success:
            print("Failed to initialize model")
            return
        
        print("=== Stress Test: 10 Concurrent Clients ===")
        
        # Enable aggressive auto cleanup for stress test
        recognizer.start_auto_cleanup(timeout_seconds=60, check_interval_seconds=10)
        print("🔧 Aggressive auto cleanup enabled for stress test (60s timeout, 10s interval)")
        
        start_time = time.time()
        test_concurrent_sessions(
            recognizer,
            num_clients=10,
            audio_path="/root/sherpa-onnx/audios/girl-zh.wav"
        )
        total_time = time.time() - start_time
        
        print(f"\nStress test completed in {total_time:.2f} seconds")
        print(f"🔧 Final session count: {recognizer.get_active_session_count()}")


def run_concurrent_test(num_clients):
    """Run concurrent session test with specified number of clients"""
    with SenseVoiceMultiSession(debug_mode=False) as recognizer:
        # Initialize model
        success = recognizer.init_model_with_params(
            vad_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/silero_vad.onnx",
            sense_voice_model_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
            tokens_path="/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
            vad_threshold=0.45,
            language="zh",
            num_threads=min(8, num_clients + 2)  # Adjust threads based on client count
        )
        
        if not success:
            print("❌ Failed to initialize model")
            return
        
        print("✅ Model initialized successfully")
        
        # Configure auto cleanup based on number of clients
        if num_clients <= 5:
            timeout_seconds = 180
            check_interval = 20
        else:
            timeout_seconds = 120
            check_interval = 15
        
        recognizer.start_auto_cleanup(timeout_seconds=timeout_seconds, check_interval_seconds=check_interval)
        
        # Run test with specified number of clients
        print(f"\n{'='*60}")
        print(f"Running Test: {num_clients} Concurrent Clients")
        print(f"Auto Cleanup: {timeout_seconds}s timeout, {check_interval}s check interval")
        print(f"{'='*60}")
        
        start_time = time.time()
        test_concurrent_sessions(
            recognizer,
            num_clients=num_clients,
            audio_path="/root/sherpa-onnx/audios/girl-zh.wav"
        )
        total_time = time.time() - start_time
        
        print(f"\nTest completed in {total_time:.2f} seconds")
        print(f"🔧 Auto cleanup status: {'Enabled' if recognizer.is_auto_cleanup_enabled() else 'Disabled'}")
        print(f"🔧 Final active sessions: {recognizer.get_active_session_count()}")


if __name__ == "__main__":
    print("=" * 60)
    print("SenseVoice Multi-Session Concurrent Test")
    print("=" * 60)
    
    try:
        num_clients = int(input("Enter number of concurrent clients (1-20): ").strip())
        if num_clients < 1 or num_clients > 20:
            print("❌ Please enter a number between 1 and 20")
            sys.exit(1)
        
        print(f"Starting test with {num_clients} concurrent clients...")
        run_concurrent_test(num_clients)
        
    except ValueError:
        print("❌ Please enter a valid number")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)