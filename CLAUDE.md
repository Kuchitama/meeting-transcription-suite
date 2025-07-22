# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meeting Transcription Suite is a Python-based audio transcription tool built on OpenAI's Whisper, optimized for Japanese language transcription and long-form audio processing. The project uses a single-file architecture with `transcribe_japanese.py` as the main entry point.

## Development Commands

### Installation
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt install -y ffmpeg libsndfile1

# Install Python dependencies
pip install -r requirements.txt
```

### Running Transcription
```bash
# Basic usage
python transcribe_japanese.py /path/to/audio.mp3

# With specific model and options
python transcribe_japanese.py audio.mp3 --model medium --fast --chunk-duration 300
```

### System Requirements Verification
```bash
# Check FFmpeg installation (required dependency)
ffmpeg -version

# Check CUDA availability (optional, for GPU acceleration)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install PyTorch with CUDA support (if NVIDIA GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Performance Optimization Commands

#### Alternative Faster Implementations
```bash
# Install faster-whisper (CTranslate2-based, significantly faster)
pip install faster-whisper

# Install whisper.cpp (C++ implementation for maximum speed)
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
./main -m models/ggml-base.bin -l ja -f input.m4a
```

#### Large File Processing
```bash
# Split large audio files for parallel processing (10-minute chunks)
ffmpeg -i input.m4a -f segment -segment_time 600 -c copy output_%03d.m4a

# Split by 30-minute chunks for very long recordings
ffmpeg -i input.m4a -f segment -segment_time 1800 -c copy chunk_%03d.m4a
```

### Audio Preprocessing Commands

#### Audio Quality Improvement
```bash
# Comprehensive preprocessing: noise reduction, normalization, format optimization
ffmpeg -i input.m4a -af "highpass=f=200,lowpass=f=3000,loudnorm" -ar 16000 -ac 1 -c:a pcm_s16le output.wav

# Basic noise reduction and volume normalization
ffmpeg -i input.m4a -af "highpass=f=200,lowpass=f=3000,volume=2" output.m4a

# Convert to mono (faster processing, smaller files)
ffmpeg -i input.m4a -ac 1 output_mono.m4a

# Resample to 16kHz (optimal for Whisper)
ffmpeg -i input.m4a -ar 16000 output_16k.m4a
```

#### Batch Preprocessing
```bash
# Process multiple files with noise reduction
for file in *.m4a; do
  ffmpeg -i "$file" -af "highpass=f=200,lowpass=f=3000,loudnorm" -ar 16000 "${file%.*}_processed.wav"
done
```

### Post-processing Commands

#### Text Refinement
```bash
# Japanese text post-processing with sed (basic cleanup)
sed -e 's/[。、]{2,}/。/g' -e 's/[。、]\s\+/。/g' transcription.txt > cleaned_transcription.txt

# Remove excessive whitespace
sed -e 's/\s\+/ /g' -e 's/^\s\+//g' -e 's/\s\+$//g' transcription.txt > trimmed_transcription.txt
```

## Architecture Overview

### Single-File Design
The entire application is contained in `transcribe_japanese.py` with modular functions:

- **Audio Processing**: `detect_silence_intervals()`, `split_audio_dynamic()`, `split_audio_fixed()`
- **Transcription Engine**: `transcribe_video()`, `transcribe_with_chunking()`, `transcribe_chunk()`
- **Performance Optimization**: `SpeculativeDecoder` class, `transcribe_chunks_batch()`
- **GPU Management**: `get_gpu_memory_info()`, automatic batch sizing

### Key Features
- **Dynamic Chunking**: Splits audio at silence intervals to preserve speech continuity
- **Speculative Decoding**: Uses fast+slow model combination for 4-6x speedup
- **Batch Processing**: Groups chunks for optimal GPU utilization
- **Memory Management**: Automatic cleanup and garbage collection

### Technology Stack
- **Core**: OpenAI Whisper, PyTorch, TorchAudio
- **Audio Processing**: librosa, soundfile, FFmpeg
- **Optimization**: NumPy, numba (JIT compilation)
- **UI**: tqdm (progress bars), argparse (CLI)

## File Structure
```
transcribe_japanese.py    # Main application (contains all functionality)
requirements.txt          # Python dependencies
README.md                # User documentation
docs/                    # Japanese documentation
├── transcribe_japanese_improvements.md
└── whisper_accuracy_tips.md
```

## Command-Line Interface

### Required Arguments
- `audio_path`: Path to audio or video file

### Optional Arguments
- `--model`: Whisper model (tiny, base, small, medium, large, large-v2, large-v3)
- `--temperature`: Sampling temperature (default: 0.0 for accuracy)
- `--chunk-duration`: Chunk size in seconds (default: 600)
- `--fast`: Enable speculative decoding
- `--no-dynamic-chunking`: Disable intelligent chunking
- `--no-batch-processing`: Disable batch optimization
- `--no-speculative-decoding`: Disable speculative decoding when using --fast

## Model Performance Comparison

| Model | Accuracy | Speed | Memory Usage | Use Case |
|-------|----------|-------|--------------|----------|
| tiny | ★☆☆ | ★★★ | ~1GB | Quick drafts, fast testing |
| base | ★★☆ | ★★☆ | ~1GB | Balanced speed/accuracy |
| small | ★★☆ | ★★☆ | ~2GB | Good general purpose |
| medium | ★★★ | ★☆☆ | ~5GB | High accuracy needs |
| large | ★★★ | ★☆☆ | ~10GB | Best accuracy, production use |

### Recommended Settings by Use Case

#### Long Meeting Recordings (1+ hours)
```bash
python transcribe_japanese.py meeting.m4a --model small --chunk-duration 1800 --fast
```

#### High Accuracy Transcription
```bash
python transcribe_japanese.py audio.m4a --model large --temperature 0.0 --no-batch-processing
```

#### Quick Draft/Testing
```bash
python transcribe_japanese.py audio.m4a --model tiny --fast
```

## Troubleshooting Commands

### Memory Issues
```bash
# Check GPU memory usage
nvidia-smi

# Force CPU-only processing if GPU memory insufficient
CUDA_VISIBLE_DEVICES="" python transcribe_japanese.py audio.m4a --model small

# Use smaller model for memory-constrained environments
python transcribe_japanese.py audio.m4a --model base
```

### Performance Issues
```bash
# Check if CUDA is being used
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# Monitor GPU utilization during transcription
watch -n 1 nvidia-smi

# Use multiple smaller chunks for very large files
python transcribe_japanese.py large_audio.m4a --chunk-duration 300
```

### Audio Format Issues
```bash
# Check audio file information
ffprobe -v quiet -print_format json -show_format -show_streams audio.m4a

# Convert unsupported formats to supported ones
ffmpeg -i input.format -c:a libmp3lame output.mp3
ffmpeg -i input.format -c:a aac output.m4a
ffmpeg -i input.format -c:a pcm_s16le output.wav
```

### Librosa PySoundFile Warning
If you see "PySoundFile failed. Trying audioread instead" warning:
```bash
# Install libsndfile1 (Ubuntu/Debian)
sudo apt install -y libsndfile1

# Or use the installation script
./install_ffmpeg.sh
```

## Output Format
- Creates `{input_filename}_transcription.txt`
- Includes full transcription, timestamped segments, and processing metadata
- Statistics: segment count, duration, model used, processing time

## Development Notes

### External Dependencies
- **FFmpeg**: Required for audio/video processing and format conversion
- **CUDA**: Optional for GPU acceleration (automatically detected)
- **libsndfile1**: Required for librosa audio loading functionality

### Performance Considerations
- GPU memory is automatically managed with dynamic batch sizing
- The `SpeculativeDecoder` class provides significant speedup for large models
- Dynamic chunking improves accuracy by respecting speech boundaries
- Parallel processing uses ThreadPoolExecutor for multi-threaded transcription

#### Recent Performance Optimizations (2025-01-22)
- **Parallel Chunk Creation**: Audio chunks are now created in parallel using up to 8 workers
- **Optimized Silence Detection**: Uses 16kHz sample rate and larger frame sizes for faster processing
- **Fast-Seek FFmpeg**: Commands use `-ss` before `-i` for significantly faster chunk creation
- **Smart Thresholds**: Dynamic chunking disabled for files >20 minutes, silence detection skipped for files >1 hour
- **Progress Indicators**: Added tqdm progress bars for chunk creation visibility
- **Memory Cleanup**: Explicit garbage collection after audio processing to free memory

### Japanese Language Optimization
- **Temperature**: 0.0 for deterministic, accurate results (most critical setting)
- **Beam Search**: beam_size=5 for optimal Japanese transcription quality
- **Initial Prompt**: Include meeting context or domain-specific terminology
- **Condition on Previous Text**: Enabled to maintain conversation context
- **VAD Filter**: Voice Activity Detection removes silence automatically
- **Punctuation**: Optimized for Japanese sentence structure (。、)
- **Silence Threshold**: Tuned for Japanese speech patterns and pauses

#### Japanese-Specific Parameters in Code
Key parameters optimized for Japanese in `transcribe_japanese.py`:
- `language='ja'` - Forces Japanese language detection
- `temperature=0.0` - Eliminates randomness for consistency
- `beam_size=5` - Balances accuracy vs speed for Japanese
- `condition_on_previous_text=True` - Maintains conversation flow
- `vad_filter=True` - Removes silent segments automatically

#### Specialized Term Handling
For technical meetings or specific domains, consider post-processing with term dictionaries:
- Database terms: "ビックエリー" → "BigQuery"
- Technical terms: "クロードコード" → "Claude Code"  
- Role abbreviations: "エスエー" → "SA (Solutions Architect)"

### Memory Management
- Automatic GPU memory detection and batch size calculation
- Explicit garbage collection after processing chunks
- Temporary file cleanup for audio extraction from video files