import ctypes
import numpy as np
import os
import wave
import time
import sys
import json
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
        
        # Add function signatures for new methods
        self.lib.set_intermediate_decode_samples.argtypes = [c_void_p, c_int]
        self.lib.get_intermediate_decode_samples.argtypes = [c_void_p]
        self.lib.get_intermediate_decode_samples.restype = c_int
        self.lib.set_vad_process_threshold.argtypes = [c_void_p, c_int]
        self.lib.get_vad_process_threshold.argtypes = [c_void_p]
        self.lib.get_vad_process_threshold.restype = c_int
        
        # Create wrapper instance
        self.wrapper = self.lib.create_wrapper()
        
        print("✅ Library loaded and initialized successfully")
    
    def __del__(self):
        if hasattr(self, 'wrapper'):
            self.lib.destroy_wrapper(self.wrapper)
    

    def init_model_with_params(self, vad_model_path, sense_voice_model_path, tokens_path,
                             vad_threshold=0.5,
                             min_silence_duration=0.5,       # 值越大，句子越完整，字准会稍微高一点，但是计算量变大。
                             min_speech_duration=0.25,       # 值太大，会漏掉一些字，数值太小，计算量变大。
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
    
    def set_intermediate_decode_samples(self, samples):
        """Set the number of samples for intermediate decoding"""
        self.lib.set_intermediate_decode_samples(self.wrapper, samples)
    
    def get_intermediate_decode_samples(self):
        """Get the current number of samples for intermediate decoding"""
        return self.lib.get_intermediate_decode_samples(self.wrapper)
    
    def set_vad_process_threshold(self, min_samples):
        """Set the minimum samples threshold for VAD processing"""
        self.lib.set_vad_process_threshold(self.wrapper, min_samples)
    
    def get_vad_process_threshold(self):
        """Get the current VAD processing threshold"""
        return self.lib.get_vad_process_threshold(self.wrapper)
    
    def set_intermediate_decode_interval(self, seconds):
        """Set intermediate decode interval in seconds (convenience method)"""
        samples = int(seconds * 16000)  # Assuming 16kHz sample rate
        self.set_intermediate_decode_samples(samples)
    
    def get_intermediate_decode_interval(self):
        """Get intermediate decode interval in seconds (convenience method)"""
        samples = self.get_intermediate_decode_samples()
        return samples / 16000.0  # Assuming 16kHz sample rate


def parse_result(result_str):
    """Parse JSON result from C++ wrapper"""
    if not result_str or result_str.startswith("Error:"):
        return None
    
    try:
        # Handle multiple results separated by " | "
        if " | " in result_str:
            results = []
            for part in result_str.split(" | "):
                parsed = json.loads(part.strip())
                results.append(parsed)
            return results
        else:
            return json.loads(result_str)
    except json.JSONDecodeError:
        # Fallback for old format compatibility
        if "[intermediate]" in result_str:
            return {"type": "intermediate", "text": result_str.replace("[intermediate] ", "").strip(), "timestamps": []}
        elif "[final]" in result_str:
            return {"type": "final", "text": result_str.replace("[final] ", "").strip(), "timestamps": []}
        else:
            return {"type": "unknown", "text": result_str.strip(), "timestamps": []}


def format_timestamps(timestamps):
    """Format timestamps for display as time intervals"""
    if not timestamps or len(timestamps) < 2:
        return ""
    
    # Convert single timestamps to time intervals [start, end]
    intervals = []
    for i in range(len(timestamps) - 1):
        intervals.append([timestamps[i], timestamps[i + 1]])
    
    # Format as string
    interval_strs = [f"[{start:.2f}, {end:.2f}]" for start, end in intervals]
    return f" [timestamps: {', '.join(interval_strs)}]"


def process_audio_streaming(recognizer, audio_path, session_id="default", print_decode=True):
    """Process audio using streaming approach with session management"""
    
    # Create session
    result = recognizer.create_session(session_id)
    if result != "OK":
        print(f"❌ Failed to create session: {result}")
        return
    
    # Record start time for RTF calculation (only processing time, not I/O)
    total_processing_time = 0
    
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            # Check audio parameters
            assert wav_file.getnchannels() == 1, "Only mono audio is supported"
            assert wav_file.getsampwidth() == 2, "Only 16-bit audio is supported"
            file_sample_rate = wav_file.getframerate()
            
            # Calculate total audio duration for RTF calculation
            total_audio_frames = wav_file.getnframes()
            audio_duration = total_audio_frames / file_sample_rate
            
            if print_decode:
                print(f"Audio parameters: sample_rate={file_sample_rate}Hz, duration={audio_duration:.2f}s")
                print(f"Session ID: {session_id}")
                print("Starting streaming processing...")
                print("=" * 60)
            
            # Reset file position to beginning
            wav_file.rewind()
            
            # Process audio chunks
            chunk_duration = 1  # 0.2 second chunks
            frames_per_chunk = int(chunk_duration * file_sample_rate)
            
            total_frames = 0
            current_intermediate = ""
            
            while True:
                audio_chunk = wav_file.readframes(frames_per_chunk)
                if not audio_chunk:
                    # End of file, get Chunk Result
                    processing_start = time.time()
                    final_result = recognizer.process_chunk_for_session(session_id, b'\x00\x00', is_last=True)
                    processing_end = time.time()
                    total_processing_time += (processing_end - processing_start)
                    
                    if final_result:
                        parsed_result = parse_result(final_result)
                        if parsed_result:
                            current_time = total_frames / file_sample_rate
                            if isinstance(parsed_result, list):
                                for result_item in parsed_result:
                                    if result_item["type"] == "final" and result_item["text"]:
                                        timestamps_str = format_timestamps(result_item["timestamps"]) if result_item["timestamps"] else ""
                                        if print_decode:
                                            if current_intermediate:
                                                print()  # Clear intermediate result line
                                            print(f"[{current_time:6.2f}s] (✓ Chunk Result): {result_item['text']}{timestamps_str}")
                            else:
                                if parsed_result["type"] == "final" and parsed_result["text"]:
                                    timestamps_str = format_timestamps(parsed_result["timestamps"]) if parsed_result["timestamps"] else ""
                                    if print_decode:
                                        if current_intermediate:
                                            print()  # Clear intermediate result line
                                        print(f"[{current_time:6.2f}s] (✓ Chunk Result): {parsed_result['text']}{timestamps_str}")
                    break
                
                # Process current audio chunk
                processing_start = time.time()
                result = recognizer.process_chunk_for_session(session_id, audio_chunk)
                processing_end = time.time()
                total_processing_time += (processing_end - processing_start)
                
                if result:
                    parsed_result = parse_result(result)
                    if parsed_result:
                        current_time = total_frames / file_sample_rate
                        
                        # Handle multiple results
                        results_to_process = parsed_result if isinstance(parsed_result, list) else [parsed_result]
                        
                        for result_item in results_to_process:
                            if result_item["type"] == "intermediate":
                                # Intermediate result - real-time update display (no timestamps)
                                text = result_item["text"]
                                if text != current_intermediate:
                                    current_intermediate = text
                                    if print_decode:
                                        print(f"\r[{current_time:6.2f}s] (Recognizing): {text:<80}", end="", flush=True)
                            
                            elif result_item["type"] == "final":
                                # Chunk Result - display with timestamps if available
                                if current_intermediate and print_decode:
                                    print()  # Clear intermediate result line
                                
                                final_text = result_item["text"]
                                if final_text and print_decode:
                                    # Only show timestamps for Chunk Results (VAD endpoints)
                                    timestamps_str = format_timestamps(result_item["timestamps"]) if result_item["timestamps"] else ""
                                    print(f"[{current_time:6.2f}s] (✓ Chunk Result): {final_text}{timestamps_str}")
                                
                                current_intermediate = ""  # Clear intermediate result
                
                total_frames += frames_per_chunk
                # time.sleep(0.02)#   Simulate real-time processing delay
            
            # Calculate RTF (Real Time Factor) - only based on actual processing time
            rtf = total_processing_time / audio_duration if audio_duration > 0 else 0
            
            # Ensure final newline
            if current_intermediate:
                print()
            
            if print_decode:
                print("=" * 60)
                print("Audio processing completed")
                print(f"Audio duration: {audio_duration:.2f}s")
                print(f"Total processing time: {total_processing_time:.2f}s")
                print(f"RTF (Real Time Factor): {rtf:.3f}")
                if rtf < 1.0:
                    print("✅ Real-time processing achieved (RTF < 1.0)")
                else:
                    print("⚠️  Processing slower than real-time (RTF > 1.0)")
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
        recognizer.set_intermediate_decode_samples(16000) # 中间解码的最小样本数
        recognizer.set_vad_process_threshold(1024)  # vad 处理的最小样本数
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    

    success = recognizer.init_model_with_params(
            vad_model_path="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/silero_vad.onnx",
            sense_voice_model_path="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
            tokens_path="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
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
    # audio_path = "/root/sherpa-onnx/audios/girl-zh.wav"
    audio_path = "/root/sherpa-onnx/audios/meeting-zh.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"Processing audio file: {audio_path}")
    process_audio_streaming(recognizer, audio_path, session_id="test_session_001")
    
    # Stop auto cleanup before exit
    print("🔧 Stopping auto cleanup")
    recognizer.stop_auto_cleanup()