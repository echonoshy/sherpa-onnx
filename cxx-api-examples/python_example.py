import ctypes
import numpy as np
import os
import wave
import time
import sys
from ctypes import c_char_p, c_int, c_void_p, POINTER

class SenseVoiceStreaming:
    def __init__(self, so_path=None):

        so_path = "./build/lib/libsense_voice_streaming_wrapper.so.1.0"
        
        print(f"Loading library from: {so_path}")
        
        # Load the shared library
        self.lib = ctypes.CDLL(so_path)
        
        # Define function signatures
        self.lib.create_wrapper.restype = c_void_p
        self.lib.destroy_wrapper.argtypes = [c_void_p]
        
        # Use detailed initialization function
        self.lib.init_model_with_params.restype = c_int

        # Session management functions
        self.lib.create_session.argtypes = [c_void_p, c_char_p]
        self.lib.create_session.restype = c_char_p
        self.lib.process_chunk_for_session_safe.argtypes = [c_void_p, c_char_p, c_char_p, c_int, c_int, c_char_p, c_int]
        self.lib.process_chunk_for_session_safe.restype = c_int
        self.lib.destroy_session.argtypes = [c_void_p, c_char_p]
        self.lib.get_active_session_count.argtypes = [c_void_p]
        self.lib.get_active_session_count.restype = c_int
        
        # Add auto cleanup function signatures
        self.lib.start_auto_cleanup.argtypes = [c_void_p, c_int, c_int]
        self.lib.stop_auto_cleanup.argtypes = [c_void_p]
        self.lib.set_session_timeout.argtypes = [c_void_p, c_int]
        self.lib.get_session_timeout.argtypes = [c_void_p]
        self.lib.get_session_timeout.restype = c_int
        self.lib.is_auto_cleanup_enabled.argtypes = [c_void_p]
        self.lib.is_auto_cleanup_enabled.restype = c_int
        self.lib.cleanup_expired_sessions.argtypes = [c_void_p, c_int]
        
        # Create wrapper instance
        self.wrapper = self.lib.create_wrapper()
        
        print("✅ Library loaded and initialized successfully")
    
    def __del__(self):
        if hasattr(self, 'wrapper'):
            self.lib.destroy_wrapper(self.wrapper)
    

    def init_model_with_params(self, vad_model_path, sense_voice_model_path, tokens_path,
                             vad_threshold=0.5,
                             min_silence_duration=0.5,      # 值越大，越不容易把句子切碎，字准会稍微高一点，但是计算量变大。
                             min_speech_duration=0.1,        # 值太大，会漏掉一些字。
                             max_speech_duration=8.0,        # 最大分句长度，目前看影响并不明显。
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
    
    def create_session(self, session_id):
        """Create a new session"""
        result = self.lib.create_session(self.wrapper, session_id.encode('utf-8'))
        return result.decode('utf-8') if result else ""
    
    def process_chunk_for_session(self, session_id, audio_bytes, is_last=False):
        """Process audio chunk for a specific session and return recognition result"""
        if not audio_bytes:
            return ""
        
        num_samples = len(audio_bytes) // 2
        
        # Create a buffer to receive the result
        buffer_size = 4096
        result_buffer = ctypes.create_string_buffer(buffer_size)
        
        result_length = self.lib.process_chunk_for_session_safe(
            self.wrapper,
            session_id.encode('utf-8'),
            audio_bytes,
            num_samples,
            1 if is_last else 0,
            result_buffer,
            buffer_size
        )
        
        if result_length > 0:
            return result_buffer.value.decode('utf-8')
        elif result_length == -1:
            return "Error: Buffer too small"
        else:
            return ""
    
    def destroy_session(self, session_id):
        """Destroy a session"""
        self.lib.destroy_session(self.wrapper, session_id.encode('utf-8'))
    
    def get_active_session_count(self):
        """Get number of active sessions"""
        return self.lib.get_active_session_count(self.wrapper)
    
    def cleanup_expired_sessions(self, timeout_seconds=300):
        """Manually cleanup expired sessions"""
        self.lib.cleanup_expired_sessions(self.wrapper, timeout_seconds)
    
    def start_auto_cleanup(self, timeout_seconds=300, check_interval_seconds=30):
        """Start automatic cleanup of expired sessions"""
        self.lib.start_auto_cleanup(self.wrapper, timeout_seconds, check_interval_seconds)
    
    def stop_auto_cleanup(self):
        """Stop automatic cleanup"""
        self.lib.stop_auto_cleanup(self.wrapper)
    
    def set_session_timeout(self, timeout_seconds):
        """Set session timeout in seconds"""
        self.lib.set_session_timeout(self.wrapper, timeout_seconds)
    
    def get_session_timeout(self):
        """Get current session timeout setting"""
        return self.lib.get_session_timeout(self.wrapper)
    
    def is_auto_cleanup_enabled(self):
        """Check if auto cleanup is enabled"""
        return self.lib.is_auto_cleanup_enabled(self.wrapper) != 0


def process_audio_streaming(recognizer, audio_path, session_id="default", print_decode=True):
    """Process audio using streaming approach with session management"""
    
    # Create session
    result = recognizer.create_session(session_id)
    if result != "OK":
        print(f"❌ Failed to create session: {result}")
        return
    
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            # Check audio parameters
            assert wav_file.getnchannels() == 1, "Only mono audio is supported"
            assert wav_file.getsampwidth() == 2, "Only 16-bit audio is supported"
            file_sample_rate = wav_file.getframerate()
            
            if print_decode:
                print(f"Audio parameters: sample_rate={file_sample_rate}Hz")
                print(f"Session ID: {session_id}")
                print("Starting streaming processing...")
                print("=" * 60)
            
            # Process audio chunks
            chunk_duration = 0.2  # 0.2 second chunks
            frames_per_chunk = int(chunk_duration * file_sample_rate)
            
            total_frames = 0
            current_intermediate = ""
            
            while True:
                audio_chunk = wav_file.readframes(frames_per_chunk)
                if not audio_chunk:
                    # End of file, get final result
                    final_result = recognizer.process_chunk_for_session(session_id, b'\x00\x00', is_last=True)
                    if final_result:
                        final_result = final_result.split("[final]")[-1].strip()
                        current_time = total_frames / file_sample_rate
                        if print_decode:
                            if current_intermediate:
                                print()  # Clear intermediate result line
                            print(f"[{current_time:6.2f}s] (✓ Chunk Result): {final_result}")
                    break
                
                # Process current audio chunk
                result = recognizer.process_chunk_for_session(session_id, audio_chunk)
                
                if result:
                    current_time = total_frames / file_sample_rate
                    
                    # Parse result type
                    if "[intermediate]" in result:
                        # Intermediate result - real-time update display
                        text = result.replace("[intermediate] ", "").strip()
                        if text != current_intermediate:
                            current_intermediate = text
                            if print_decode:
                                print(f"\r[{current_time:6.2f}s] (Recognizing): {text:<80}", end="", flush=True)
                    
                    elif "[final]" in result or result.strip():
                        # Final result - display complete content with newline
                        if current_intermediate and print_decode:
                            print()  # Clear intermediate result line
                        
                        # Extract final text, handle possible format issues
                        if "[final]" in result:
                            final_text = result.split("[final]")[-1].strip()
                        else:
                            final_text = result.strip()
                        if final_text and print_decode:
                            print(f"[{current_time:6.2f}s] (✓ Chunk Result): {final_text}")
                        
                        current_intermediate = ""  # Clear intermediate result
                
                total_frames += frames_per_chunk
                time.sleep(0.01)  # Simulate real-time processing delay
            
            # Ensure final newline
            if current_intermediate:
                print()
            
            if print_decode:
                print("=" * 60)
                print("Audio processing completed")
                print(f"Active sessions: {recognizer.get_active_session_count()}")
    
    finally:
        # Clean up session
        recognizer.destroy_session(session_id)
        if print_decode:
            print(f"Session {session_id} destroyed")


if __name__ == "__main__":
    print("=" * 60)
    print("SenseVoice Streaming Recognition Example")
    print("=" * 60)
    
    # Initialize recognizer
    try:
        recognizer = SenseVoiceStreaming()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Model paths
    model_paths = {
        "vad_model": "./sherpa-onnx-weights/silero_vad.onnx",
        "sense_voice_model": "./sherpa-onnx-weights/model.onnx",
        "tokens": "./sherpa-onnx-weights/tokens.txt"
    }
    
    # Check model files
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"❌ Missing {name}: {path}")
            sys.exit(1)
    

    success = recognizer.init_model_with_params(
            vad_model_path="sherpa-onnx-weights/silero_vad.onnx",
            sense_voice_model_path="sherpa-onnx-weights/model.onnx",
            tokens_path="sherpa-onnx-weights/tokens.txt",
            vad_threshold=0.5,
            language="auto",
            num_threads=1
        )

    if not success:
        print("❌ Failed to initialize model")
        sys.exit(1)
    
    print("✅ Model initialized successfully")
    
    # Start auto cleanup for session management
    print("🔧 Starting auto cleanup (timeout: 300s, check interval: 30s)")
    recognizer.start_auto_cleanup(timeout_seconds=300, check_interval_seconds=30)
    
    # Process audio file
    # audio_path = "exps/audios/Audiodata_2025_3/地铁-英文现场录音-15条/地铁-英文现场录音-1-2.wav"
    audio_path = "/home/lake/gitcodes/asr-sherpa-onnx/4022654511881686103_16k.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"Processing audio file: {audio_path}")
    process_audio_streaming(recognizer, audio_path, session_id="test_session_001")
    
    # Stop auto cleanup before exit
    print("🔧 Stopping auto cleanup")
    recognizer.stop_auto_cleanup()