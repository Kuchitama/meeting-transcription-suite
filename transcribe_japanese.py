#!/usr/bin/env python3
import whisper
import sys
from datetime import timedelta
import torch
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import subprocess
import tempfile
import shutil
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf
import gc
import time

def format_timestamp(seconds):
    """Convert seconds to timestamp format (HH:MM:SS)"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_audio_duration(audio_path):
    """Get audio duration in seconds using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return 0

def detect_silence_intervals(audio_path, min_silence_duration=2.0, silence_threshold=-40):
    """
    Detect silence intervals in audio file using librosa with optimizations
    
    Args:
        audio_path: Path to audio file
        min_silence_duration: Minimum duration of silence in seconds
        silence_threshold: Silence threshold in dB
        
    Returns:
        List of (start_time, end_time) tuples for silence intervals
    """
    try:
        print("Analyzing audio for silence detection...")
        
        # Get duration first to check if we should skip
        duration = get_audio_duration(audio_path)
        if duration > 3600:  # Skip for files > 1 hour
            print(f"Audio too long ({duration/60:.1f} minutes), skipping silence detection")
            return []
        
        # Load audio with reduced sample rate for faster processing
        target_sr = 16000  # Lower sample rate for faster processing
        print(f"Loading audio at {target_sr}Hz for analysis...")
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Use larger frame and hop lengths for faster processing
        frame_length = 4096  # Increased from 2048
        hop_length = 1024    # Increased from 512
        
        print("Calculating audio energy levels...")
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms)
        
        # Find silence frames
        silence_frames = rms_db < silence_threshold
        
        # Convert frame indices to time
        times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)
        
        # Find continuous silence intervals
        print("Detecting silence intervals...")
        silence_intervals = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                # Start of silence
                silence_start = times[i]
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence
                silence_duration = times[i] - silence_start
                if silence_duration >= min_silence_duration:
                    silence_intervals.append((silence_start, times[i]))
                in_silence = False
        
        # Handle case where audio ends in silence
        if in_silence:
            silence_duration = times[-1] - silence_start
            if silence_duration >= min_silence_duration:
                silence_intervals.append((silence_start, times[-1]))
        
        print(f"Found {len(silence_intervals)} silence intervals")
        
        # Clear memory
        del y, rms, rms_db
        gc.collect()
        
        return silence_intervals
        
    except Exception as e:
        print(f"Error detecting silence: {e}")
        return []

def detect_speech_boundaries(audio_path, min_speech_duration=30.0, max_speech_duration=900.0):
    """
    Detect optimal speech boundaries for chunking
    
    Args:
        audio_path: Path to audio file
        min_speech_duration: Minimum chunk duration in seconds
        max_speech_duration: Maximum chunk duration in seconds
        
    Returns:
        List of boundary timestamps
    """
    try:
        # Get silence intervals
        silence_intervals = detect_silence_intervals(audio_path)
        
        if not silence_intervals:
            print("No silence intervals detected, using time-based splitting")
            return []
        
        # Find good split points
        duration = get_audio_duration(audio_path)
        boundaries = [0]  # Start with beginning
        
        current_time = 0
        
        while current_time < duration:
            # Look for next good split point
            target_time = current_time + min_speech_duration
            max_time = current_time + max_speech_duration
            
            # Find silence intervals in the target range
            candidates = []
            for start, end in silence_intervals:
                if target_time <= start <= max_time:
                    candidates.append((start, end))
            
            if candidates:
                # Choose the silence interval closest to target time
                best_silence = min(candidates, key=lambda x: abs(x[0] - target_time))
                split_point = best_silence[0] + (best_silence[1] - best_silence[0]) / 2
                boundaries.append(split_point)
                current_time = split_point
            else:
                # No good silence found, use time-based split
                current_time += max_speech_duration
                if current_time < duration:
                    boundaries.append(current_time)
        
        # Add end time
        if boundaries[-1] < duration:
            boundaries.append(duration)
        
        return boundaries
        
    except Exception as e:
        print(f"Error detecting speech boundaries: {e}")
        return []

def split_audio_dynamic(audio_path, min_duration=30.0, max_duration=900.0, use_dynamic=True):
    """Split audio file using dynamic or fixed chunking"""
    temp_dir = tempfile.mkdtemp()
    chunk_files = []
    chunk_boundaries = []
    
    try:
        duration = get_audio_duration(audio_path)
        print(f"Audio duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        if duration <= max_duration:
            # No need to split
            return [audio_path], temp_dir, [0.0]
        
        if use_dynamic:
            # Try dynamic splitting first
            print("Analyzing audio for optimal split points...")
            
            # Skip dynamic chunking for very long files (>20 minutes) to save time
            if duration > 1200:  # 20 minutes
                print(f"Audio too long ({duration/60:.1f} minutes), using fixed chunking for speed")
                use_dynamic = False
            else:
                boundaries = detect_speech_boundaries(audio_path, min_duration, max_duration)
                
                if boundaries and len(boundaries) > 2:  # More than just start and end
                    print(f"Using dynamic splitting with {len(boundaries)-1} chunks")
                    
                    # Create chunks in parallel for faster processing
                    from concurrent.futures import ThreadPoolExecutor
                    import threading
                    
                    def create_chunk(chunk_info):
                        i, start_time, end_time = chunk_info
                        chunk_duration = end_time - start_time
                        chunk_file = os.path.join(temp_dir, f"chunk_{i:03d}.m4a")
                        
                        print(f"Creating chunk {i+1}/{len(boundaries)-1} (duration: {chunk_duration:.1f}s)...")
                        
                        # Use ffmpeg with fast seek
                        result = subprocess.run([
                            'ffmpeg', '-ss', str(start_time), '-i', audio_path,
                            '-t', str(chunk_duration), '-c', 'copy', '-avoid_negative_ts', 'make_zero',
                            '-y', chunk_file
                        ], capture_output=True)
                        
                        if result.returncode == 0 and os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
                            print(f"  Chunk {i+1} created successfully")
                            return i, chunk_file, start_time
                        else:
                            print(f"  Error creating chunk {i+1}: {result.stderr.decode()}")
                            return None
                    
                    # Prepare chunk info
                    chunk_infos = [(i, boundaries[i], boundaries[i+1]) for i in range(len(boundaries) - 1)]
                    
                    # Process chunks in parallel (use more workers for faster processing)
                    max_workers = min(8, len(chunk_infos), multiprocessing.cpu_count())
                    print(f"Using {max_workers} parallel workers for chunk creation...")
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        futures = [executor.submit(create_chunk, info) for info in chunk_infos]
                        
                        # Process with progress bar
                        results = []
                        for future in tqdm(as_completed(futures), total=len(futures), desc="Creating chunks"):
                            result = future.result()
                            if result:
                                results.append(result)
                    
                    # Collect successful chunks from async results
                    chunk_data = []
                    for result in results:
                        if result:
                            i, chunk_file, start_time = result
                            chunk_data.append((i, chunk_file, start_time))
                    
                    # Sort by index to maintain order
                    chunk_data.sort(key=lambda x: x[0])
                    
                    # Extract file paths and boundaries
                    chunk_files = [cd[1] for cd in chunk_data]
                    chunk_boundaries = [cd[2] for cd in chunk_data]
                    
                    if chunk_files:
                        print(f"Created {len(chunk_files)} dynamic chunks")
                        return chunk_files, temp_dir, chunk_boundaries
        
        # Fallback to fixed chunking
        print(f"Using fixed chunking with {max_duration//60}-minute chunks...")
        chunk_files, temp_dir = split_audio_fixed(audio_path, max_duration, temp_dir)
        
        # Generate boundaries for fixed chunking
        for i in range(len(chunk_files)):
            chunk_boundaries.append(i * max_duration)
        
        return chunk_files, temp_dir, chunk_boundaries
        
    except Exception as e:
        print(f"Error in dynamic splitting: {e}")
        print("Falling back to fixed chunking...")
        chunk_files, temp_dir = split_audio_fixed(audio_path, max_duration, temp_dir)
        
        # Generate boundaries for fixed chunking fallback
        for i in range(len(chunk_files)):
            chunk_boundaries.append(i * max_duration)
        
        return chunk_files, temp_dir, chunk_boundaries

def split_audio_fixed(audio_path, chunk_duration, temp_dir):
    """Fixed-time audio splitting with parallel processing"""
    chunk_files = []
    
    try:
        duration = get_audio_duration(audio_path)
        num_chunks = int(duration // chunk_duration) + 1
        
        print(f"Creating {num_chunks} fixed chunks of {chunk_duration}s each...")
        
        # Function to create a single chunk
        def create_fixed_chunk(chunk_info):
            i, start_time = chunk_info
            chunk_file = os.path.join(temp_dir, f"chunk_{i:03d}.m4a")
            
            # Calculate actual chunk duration (last chunk might be shorter)
            actual_duration = min(chunk_duration, duration - start_time)
            
            print(f"Creating chunk {i+1}/{num_chunks} (start: {start_time:.1f}s, duration: {actual_duration:.1f}s)...")
            
            # Use ffmpeg with fast seek (-ss before -i)
            result = subprocess.run([
                'ffmpeg', '-ss', str(start_time), '-i', audio_path,
                '-t', str(actual_duration), '-c', 'copy', '-avoid_negative_ts', 'make_zero',
                '-y', chunk_file
            ], capture_output=True)
            
            if result.returncode == 0 and os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
                print(f"  Chunk {i+1} created successfully")
                return chunk_file
            else:
                print(f"  Error creating chunk {i+1}: {result.stderr.decode() if result.stderr else 'Unknown error'}")
                return None
        
        # Prepare chunk info
        chunk_infos = [(i, i * chunk_duration) for i in range(num_chunks)]
        
        # Process chunks in parallel (limit to 4 workers)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, num_chunks)) as executor:
            results = list(executor.map(create_fixed_chunk, chunk_infos))
        
        # Collect successful chunks
        chunk_files = [f for f in results if f is not None]
        
        print(f"Created {len(chunk_files)} fixed chunks")
        return chunk_files, temp_dir
        
    except Exception as e:
        print(f"Error in fixed splitting: {e}")
        return [audio_path], temp_dir

def split_audio(audio_path, chunk_duration=600):
    """Legacy wrapper for backward compatibility"""
    return split_audio_fixed(audio_path, chunk_duration, tempfile.mkdtemp())

def get_gpu_memory_info():
    """Get GPU memory information for batch size optimization"""
    if not torch.cuda.is_available():
        return None, None
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        free_memory = total_memory - allocated_memory
        return total_memory, free_memory
    except:
        return None, None

def calculate_optimal_batch_size(chunk_count, model_size="base"):
    """Calculate optimal batch size based on GPU memory and model size"""
    total_memory, free_memory = get_gpu_memory_info()
    
    if total_memory is None:
        # CPU fallback
        return min(multiprocessing.cpu_count(), chunk_count, 4)
    
    # Memory requirements per model (rough estimates in MB)
    model_memory_requirements = {
        "tiny": 200,
        "base": 400,
        "small": 800,
        "medium": 1500,
        "large": 3000,
        "large-v2": 3000,
        "large-v3": 3000
    }
    
    model_memory = model_memory_requirements.get(model_size, 400) * 1024 * 1024  # Convert to bytes
    safety_factor = 0.7  # Use only 70% of available memory for safety
    
    max_batch_size = int((free_memory * safety_factor) / model_memory)
    optimal_batch_size = min(max_batch_size, chunk_count, 8)  # Cap at 8 for stability
    
    return max(1, optimal_batch_size)

def transcribe_chunk(chunk_info):
    """Transcribe a single audio chunk"""
    chunk_file, chunk_index, model_size, temperature = chunk_info
    
    # Load model for this chunk
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    
    try:
        # Transcribe chunk
        result = model.transcribe(
            chunk_file,
            language="ja",
            verbose=False,
            temperature=temperature,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.8,  # Increase threshold to handle silence better
            condition_on_previous_text=True,
            initial_prompt=None,  # Remove initial_prompt to avoid repetition
            word_timestamps=False,
            beam_size=5
        )
        
        return chunk_index, result
    
    finally:
        # Clean up model to free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def transcribe_chunks_batch(chunk_batch, model_size, temperature):
    """Transcribe multiple chunks in a batch for better GPU utilization"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model once for the batch
    model = whisper.load_model(model_size, device=device)
    batch_results = {}
    
    try:
        for chunk_file, chunk_index, _, _ in chunk_batch:
            try:
                result = model.transcribe(
                    chunk_file,
                    language="ja",
                    verbose=False,
                    temperature=temperature,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.8,
                    condition_on_previous_text=True,
                    initial_prompt=None,
                    word_timestamps=False,
                    beam_size=5
                )
                batch_results[chunk_index] = result
            except Exception as e:
                print(f"Error processing chunk {chunk_index}: {e}")
                # Create empty result for failed chunk
                batch_results[chunk_index] = {"text": "", "segments": []}
        
        return batch_results
    
    finally:
        # Clean up model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class SpeculativeDecoder:
    """Speculative decoding using a fast model to predict and a slow model to verify"""
    
    def __init__(self, fast_model_size="tiny", slow_model_size="base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fast_model_size = fast_model_size
        self.slow_model_size = slow_model_size
        self.fast_model = None
        self.slow_model = None
        
    def load_models(self):
        """Load both fast and slow models"""
        print(f"Loading fast model ({self.fast_model_size}) and slow model ({self.slow_model_size})...")
        self.fast_model = whisper.load_model(self.fast_model_size, device=self.device)
        self.slow_model = whisper.load_model(self.slow_model_size, device=self.device)
        
    def unload_models(self):
        """Clean up models to free memory"""
        if self.fast_model:
            del self.fast_model
            self.fast_model = None
        if self.slow_model:
            del self.slow_model
            self.slow_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def transcribe_speculative(self, audio_file, temperature=0.0):
        """Perform speculative decoding transcription"""
        if not self.fast_model or not self.slow_model:
            self.load_models()
        
        try:
            # Fast prediction
            fast_result = self.fast_model.transcribe(
                audio_file,
                language="ja",
                verbose=False,
                temperature=temperature,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.8,
                condition_on_previous_text=True,
                initial_prompt=None,
                word_timestamps=False,
                beam_size=3  # Use smaller beam for speed
            )
            
            # If fast model confidence is high, use it directly
            fast_avg_logprob = sum(seg.get('avg_logprob', -1.0) for seg in fast_result['segments']) / max(len(fast_result['segments']), 1)
            
            if fast_avg_logprob > -0.5:  # High confidence threshold
                return fast_result
            
            # Low confidence, use slow model for verification/improvement
            slow_result = self.slow_model.transcribe(
                audio_file,
                language="ja",
                verbose=False,
                temperature=temperature,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.8,
                condition_on_previous_text=True,
                initial_prompt=fast_result['text'][:100] if fast_result['text'] else None,  # Use fast result as context
                word_timestamps=False,
                beam_size=5
            )
            
            return slow_result
            
        except Exception as e:
            print(f"Error in speculative decoding: {e}")
            # Fallback to slow model only
            return self.slow_model.transcribe(
                audio_file,
                language="ja",
                verbose=False,
                temperature=temperature,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.8,
                condition_on_previous_text=True,
                initial_prompt=None,
                word_timestamps=False,
                beam_size=5
            )
    
    def transcribe_chunks_speculative(self, chunk_batch, temperature=0.0):
        """Transcribe multiple chunks using speculative decoding"""
        if not self.fast_model or not self.slow_model:
            self.load_models()
        
        batch_results = {}
        
        for chunk_file, chunk_index, _, _ in chunk_batch:
            try:
                result = self.transcribe_speculative(chunk_file, temperature)
                batch_results[chunk_index] = result
            except Exception as e:
                print(f"Error processing chunk {chunk_index} with speculative decoding: {e}")
                batch_results[chunk_index] = {"text": "", "segments": []}
        
        return batch_results

def transcribe_video(video_path, model_size="base", temperature=0.0, fast_mode=False, 
                    chunk_duration=600, use_dynamic_chunking=True, use_batch_processing=True, use_speculative_decoding=True):
    print(f"Loading Whisper model ({model_size})...")
    
    # Check if CUDA is available for faster processing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get audio duration first
    duration = get_audio_duration(video_path)
    print(f"Audio duration: {duration/60:.1f} minutes")
    
    if fast_mode and duration > chunk_duration:
        print("Using fast mode with audio chunking...")
        return transcribe_with_chunking(video_path, model_size, temperature, chunk_duration, use_dynamic_chunking, use_batch_processing, use_speculative_decoding)
    
    # Standard single-pass transcription
    model = whisper.load_model(model_size, device=device)
    
    print(f"Transcribing audio/video: {video_path}")
    print("This may take a few minutes...")
    
    # Transcribe with optimized parameters for Japanese
    result = model.transcribe(
        video_path, 
        language="ja",
        verbose=False,
        temperature=temperature,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.8,  # Increase threshold to handle silence better
        condition_on_previous_text=True,
        initial_prompt=None,  # Remove initial_prompt to avoid repetition
        word_timestamps=False,
        beam_size=5
    )
    
    return result

def transcribe_with_chunking(video_path, model_size="base", temperature=0.0, chunk_duration=600, use_dynamic_chunking=True, use_batch_processing=True, use_speculative_decoding=True):
    """Transcribe long audio by splitting into chunks and processing in parallel"""
    
    # Split audio into chunks
    if use_dynamic_chunking:
        chunk_files, temp_dir, chunk_boundaries = split_audio_dynamic(video_path, min_duration=30.0, max_duration=chunk_duration, use_dynamic=True)
    else:
        chunk_files, temp_dir = split_audio_fixed(video_path, chunk_duration, tempfile.mkdtemp())
        # Generate boundaries for fixed chunking
        chunk_boundaries = [i * chunk_duration for i in range(len(chunk_files))]
    
    if len(chunk_files) == 1:
        # No chunking needed, use standard method
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(
            video_path,
            language="ja",
            verbose=False,
            temperature=temperature,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.8,  # Increase threshold to handle silence better
            condition_on_previous_text=True,
            initial_prompt=None,  # Remove initial_prompt to avoid repetition
            word_timestamps=False,
            beam_size=5
        )
        return result
    
    try:
        # Prepare chunk information for parallel processing
        chunk_info_list = [
            (chunk_file, i, model_size, temperature) 
            for i, chunk_file in enumerate(chunk_files)
        ]
        
        chunk_results = {}
        
        # Choose processing method based on parameters
        if use_speculative_decoding:
            print("Using speculative decoding for enhanced speed...")
            speculative_decoder = SpeculativeDecoder(
                fast_model_size="tiny", 
                slow_model_size=model_size
            )
            
            # Calculate optimal batch size
            batch_size = calculate_optimal_batch_size(len(chunk_files), model_size)
            print(f"Processing {len(chunk_files)} chunks with batch size {batch_size} using speculative decoding...")
            
            # Process in batches
            for i in range(0, len(chunk_info_list), batch_size):
                batch = chunk_info_list[i:i + batch_size]
                batch_results = speculative_decoder.transcribe_chunks_speculative(batch, temperature)
                chunk_results.update(batch_results)
                
                # Update progress
                progress = min(i + batch_size, len(chunk_info_list))
                print(f"Completed {progress}/{len(chunk_info_list)} chunks")
            
            speculative_decoder.unload_models()
            
        elif use_batch_processing:
            # Calculate optimal batch size for GPU/CPU
            batch_size = calculate_optimal_batch_size(len(chunk_files), model_size)
            print(f"Processing {len(chunk_files)} chunks with batch size {batch_size}...")
            
            # Process chunks in batches
            for i in range(0, len(chunk_info_list), batch_size):
                batch = chunk_info_list[i:i + batch_size]
                
                try:
                    batch_results = transcribe_chunks_batch(batch, model_size, temperature)
                    chunk_results.update(batch_results)
                except Exception as e:
                    print(f"Error processing batch starting at chunk {i}: {e}")
                    # Fallback to individual processing for this batch
                    for chunk_info in batch:
                        try:
                            chunk_index, result = transcribe_chunk(chunk_info)
                            chunk_results[chunk_index] = result
                        except Exception as chunk_e:
                            print(f"Error processing chunk {chunk_info[1]}: {chunk_e}")
                
                # Update progress
                progress = min(i + batch_size, len(chunk_info_list))
                print(f"Completed {progress}/{len(chunk_info_list)} chunks")
                
        else:
            # Original parallel processing method
            max_workers = min(multiprocessing.cpu_count(), len(chunk_files))
            print(f"Processing {len(chunk_files)} chunks with {max_workers} workers (original method)...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks for processing
                future_to_chunk = {
                    executor.submit(transcribe_chunk, chunk_info): chunk_info[1] 
                    for chunk_info in chunk_info_list
                }
                
                # Process completed chunks with progress bar
                for future in tqdm(as_completed(future_to_chunk), total=len(chunk_files), desc="Processing chunks"):
                    chunk_index = future_to_chunk[future]
                    try:
                        chunk_index, result = future.result()
                        chunk_results[chunk_index] = result
                    except Exception as e:
                        print(f"Error processing chunk {chunk_index}: {e}")
        
        # Combine results from all chunks
        print("Combining results from all chunks...")
        
        combined_result = combine_chunk_results(chunk_results, chunk_boundaries, chunk_duration)
        
        return combined_result
        
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def combine_chunk_results(chunk_results, chunk_boundaries=None, chunk_duration=600):
    """Combine transcription results from multiple chunks"""
    combined_segments = []
    full_text = ""
    
    # Sort chunks by index
    sorted_chunks = sorted(chunk_results.items())
    
    for chunk_index, result in sorted_chunks:
        # Calculate time offset
        if chunk_boundaries and len(chunk_boundaries) > chunk_index:
            # Use actual boundaries from dynamic chunking
            time_offset = chunk_boundaries[chunk_index]
        else:
            # Fallback to estimated offset (for fixed chunking)
            time_offset = chunk_index * chunk_duration
        
        # Add text
        if result["text"].strip():
            full_text += result["text"].strip() + " "
        
        # Add segments with time offset
        for segment in result["segments"]:
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += time_offset
            adjusted_segment["end"] += time_offset
            combined_segments.append(adjusted_segment)
    
    # Create combined result
    combined_result = {
        "text": full_text.strip(),
        "segments": combined_segments,
        "language": "ja"
    }
    
    return combined_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Japanese audio transcription using Whisper")
    parser.add_argument("audio_file", type=str, help="Path to audio/video file")
    parser.add_argument("--model", type=str, default="base", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for sampling (default: 0.0)")
    parser.add_argument("--fast", action="store_true",
                       help="Use fast mode with audio chunking for long files")
    parser.add_argument("--chunk-duration", type=int, default=600,
                       help="Duration of each audio chunk in seconds (default: 600)")
    parser.add_argument("--no-dynamic-chunking", action="store_true",
                       help="Disable dynamic chunking and use fixed-time splitting")
    parser.add_argument("--silence-threshold", type=float, default=-40,
                       help="Silence threshold in dB for dynamic chunking (default: -40)")
    parser.add_argument("--min-silence-duration", type=float, default=2.0,
                       help="Minimum silence duration in seconds (default: 2.0)")
    parser.add_argument("--no-batch-processing", action="store_true",
                       help="Disable batch processing optimization")
    parser.add_argument("--no-speculative-decoding", action="store_true",
                       help="Disable speculative decoding (enabled by default)")
    
    args = parser.parse_args()
    
    use_dynamic_chunking = not args.no_dynamic_chunking
    use_batch_processing = not args.no_batch_processing
    use_speculative_decoding = not args.no_speculative_decoding
    
    result = transcribe_video(
        args.audio_file, 
        args.model, 
        args.temperature, 
        args.fast, 
        args.chunk_duration,
        use_dynamic_chunking,
        use_batch_processing,
        use_speculative_decoding
    )
    
    print("\n=== Transcription Results ===\n")
    
    # Print full transcription
    print("Full Transcription:")
    print(result["text"])
    print("\n")
    
    # Print with timestamps
    print("Transcription with Timestamps:")
    for segment in result["segments"]:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        if text:
            print(f"[{start_time} - {end_time}] {text}")
    
    # Save to file
    video_path = args.audio_file
    if video_path.endswith(".mp4"):
        output_file = video_path.replace(".mp4", "_transcription.txt")
    elif video_path.endswith(".m4a"):
        output_file = video_path.replace(".m4a", "_transcription.txt")
    else:
        output_file = video_path + "_transcription.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Japanese Audio Transcription ===\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Audio/Video: {video_path}\n")
        f.write(f"Fast Mode: {args.fast}\n")
        f.write(f"Dynamic Chunking: {use_dynamic_chunking}\n")
        f.write(f"Batch Processing: {use_batch_processing}\n")
        f.write(f"Speculative Decoding: {use_speculative_decoding}\n")
        f.write(f"Chunk Duration: {args.chunk_duration}s\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Full Transcription:\n")
        f.write(result["text"] + "\n\n")
        
        f.write("Transcription with Timestamps:\n")
        for segment in result["segments"]:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            if text:
                f.write(f"[{start_time} - {end_time}] {text}\n")
    
    print(f"\nTranscription saved to: {output_file}")
    
    # Print statistics
    print(f"\nTranscription Statistics:")
    print(f"Total segments: {len(result['segments'])}")
    print(f"Total duration: {format_timestamp(result['segments'][-1]['end'] if result['segments'] else 0)}")