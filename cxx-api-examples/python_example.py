import ctypes
import numpy as np
import os
import wave
import time
import sys
from ctypes import c_char_p, c_int, c_void_p, POINTER

class SenseVoiceStreaming:
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
        
        # Load the shared library
        self.lib = ctypes.CDLL(so_path)
        
        # Define function signatures
        self.lib.create_wrapper.restype = c_void_p
        self.lib.destroy_wrapper.argtypes = [c_void_p]
        
        # Use simplified initialization function
        self.lib.init_model.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]
        self.lib.init_model.restype = c_int
        
        # Processing functions
        self.lib.process_chunk.argtypes = [c_void_p, c_char_p, c_int, c_int]
        self.lib.process_chunk.restype = c_char_p
        self.lib.flush_remaining.argtypes = [c_void_p]
        self.lib.flush_remaining.restype = c_char_p
        self.lib.reset_wrapper.argtypes = [c_void_p]
        self.lib.set_debug_mode.argtypes = [c_void_p, c_int]
        
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
        self.lib.set_debug_mode(self.wrapper, 1 if debug_mode else 0)
        
        print("✅ Library loaded and initialized successfully")
    
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
    
    def process_chunk(self, audio_bytes, is_last=False):
        """Process audio chunk and return recognition result"""
        if not audio_bytes:
            return ""
        
        num_samples = len(audio_bytes) // 2
        result = self.lib.process_chunk(
            self.wrapper,
            audio_bytes,
            num_samples,
            1 if is_last else 0
        )
        
        return result.decode('utf-8') if result else ""
    
    def flush_remaining(self):
        """Flush any remaining audio and return final results"""
        result = self.lib.flush_remaining(self.wrapper)
        return result.decode('utf-8') if result else ""
    
    def reset(self):
        """Reset the internal state"""
        self.lib.reset_wrapper(self.wrapper)
    
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


def process_audio_streaming(recognizer, audio_path, print_decode=True):
    """Process audio using streaming approach"""
    
    with wave.open(audio_path, 'rb') as wav_file:
        # Check audio parameters
        assert wav_file.getnchannels() == 1, "Only mono audio is supported"
        assert wav_file.getsampwidth() == 2, "Only 16-bit audio is supported"
        file_sample_rate = wav_file.getframerate()
        
        if print_decode:
            print(f"Audio parameters: sample_rate={file_sample_rate}Hz")
            print("Starting streaming processing...")
            print("=" * 60)
        
        # Process audio chunks
        chunk_duration = 0.1  # 0.1 second chunks
        frames_per_chunk = int(chunk_duration * file_sample_rate)
        
        total_frames = 0
        current_intermediate = ""
        
        while True:
            audio_chunk = wav_file.readframes(frames_per_chunk)
            if not audio_chunk:
                # End of file, get final result
                final_result = recognizer.flush_remaining()
                if final_result:
                    final_result = final_result.split("[final]")[-1].strip()
                    current_time = total_frames / file_sample_rate
                    if print_decode:
                        if current_intermediate:
                            print()  # Clear intermediate result line
                        print(f"[{current_time:6.2f}s] (✓ Chunk Result): {final_result}")
                break
            
            # Process current audio chunk
            result = recognizer.process_chunk(audio_chunk)
            
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
            time.sleep(0.1)  # Simulate real-time processing delay
        
        # Ensure final newline
        if current_intermediate:
            print()
        
        if print_decode:
            print("=" * 60)
            print("Audio processing completed")


if __name__ == "__main__":
    print("=" * 60)
    print("SenseVoice Streaming Recognition Example")
    print("=" * 60)
    
    # Initialize recognizer
    try:
        recognizer = SenseVoiceStreaming(debug_mode=False)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Model paths
    model_paths = {
        "vad_model": "/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/silero_vad.onnx",
        "sense_voice_model": "/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
        "tokens": "/root/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"
    }
    
    # Check model files
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"❌ Missing {name}: {path}")
            sys.exit(1)
    
    # Initialize model
    success = recognizer.init_model(
        model_paths["vad_model"],
        model_paths["sense_voice_model"], 
        model_paths["tokens"]
    )
    
    if not success:
        print("❌ Failed to initialize model")
        sys.exit(1)
    
    print("✅ Model initialized successfully")
    
    # Start auto cleanup for session management
    print("🔧 Starting auto cleanup (timeout: 300s, check interval: 30s)")
    recognizer.start_auto_cleanup(timeout_seconds=300, check_interval_seconds=30)
    
    # Process audio file
    audio_path = "/root/sherpa-onnx/audios/girl-zh.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"Processing audio file: {audio_path}")
    process_audio_streaming(recognizer, audio_path)
    
    # Stop auto cleanup before exit
    print("🔧 Stopping auto cleanup")
    recognizer.stop_auto_cleanup()