# Meeting Transcription Suite

A powerful audio transcription tool built on OpenAI's Whisper, optimized for long-form audio processing with advanced features.

## Features

- 🎯 **Multi-model support**: tiny, base, small, medium, large models
- 🚀 **Optimized for long audio**: Automatic chunking with configurable duration
- 🎵 **Dynamic chunk splitting**: Intelligent splitting based on silence detection
- ⚡ **Parallel processing**: Multi-threaded transcription for faster results
- 🧠 **GPU optimization**: Automatic batch size calculation based on available VRAM
- 🔄 **Speculative decoding**: 4-6x speedup using draft-and-verify approach
- 🌏 **Multi-language support**: Optimized for Japanese, but works with many languages
- 📊 **Progress tracking**: Real-time progress bars with tqdm
- 🎞️ **Video support**: Extracts audio from video files automatically

## Quick Start

### 1. Install Dependencies

#### Option A: Automated Setup (Recommended)
```bash
# Install FFmpeg (required)
./install_ffmpeg.sh

# Install Python dependencies
pip install -r requirements.txt
```

#### Option B: Manual Setup
```bash
# Install FFmpeg
sudo apt update && sudo apt install -y ffmpeg  # Ubuntu/Debian
brew install ffmpeg                            # macOS/Homebrew

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
# Check FFmpeg
ffmpeg -version

# Test the transcription tool
python transcribe_japanese.py --help
```

## Usage

### Basic usage
```bash
python transcribe_japanese.py /path/to/audio.mp3
```

### Advanced options
```bash
# Use a specific model
python transcribe_japanese.py audio.mp3 --model medium

# Custom chunk duration
python transcribe_japanese.py audio.mp3 --chunk-duration 300

# Enable dynamic chunking
python transcribe_japanese.py audio.mp3 --dynamic-chunking

# Adjust temperature for accuracy
python transcribe_japanese.py audio.mp3 --temperature 0.2

# Fast mode (uses speculative decoding)
python transcribe_japanese.py audio.mp3 --fast
```

### Command-line arguments
- `audio_path`: Path to the audio or video file
- `--model`: Whisper model size (default: base)
- `--temperature`: Temperature for sampling (default: 0.0)
- `--chunk-duration`: Duration of each chunk in seconds (default: 600)
- `--dynamic-chunking`: Enable intelligent chunk splitting based on silence
- `--fast`: Enable speculative decoding for faster transcription

## System Requirements

### Essential Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **FFmpeg**: Required for audio/video processing
- **Operating System**: Linux, macOS, Windows (WSL recommended)

### Recommended Requirements
- **GPU**: CUDA-capable NVIDIA GPU (for faster processing)
- **Memory**: 8GB+ RAM (16GB+ for large models)
- **Storage**: 5GB+ free space (for model downloads)

### Python Dependencies
See `requirements.txt` for complete list:
- `openai-whisper>=20230314`
- `torch>=2.0.0`
- `librosa>=0.10.0`
- `numpy>=1.24.0`
- And more...

## Detailed Installation Guide

### Prerequisites Setup

#### For Ubuntu/Debian Systems
```bash
# Update system packages
sudo apt update

# Install Python development tools
sudo apt install -y python3-pip python3-dev python3-venv

# Install FFmpeg and audio libraries
sudo apt install -y ffmpeg libffi-dev libssl-dev libsndfile1

# Install additional dependencies for Python compilation (if using pyenv)
sudo apt install -y build-essential zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev
```

#### For macOS Systems
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python, FFmpeg and audio libraries
brew install python ffmpeg libsndfile
```

#### For Windows (WSL2 Recommended)
```bash
# Enable WSL2 and install Ubuntu from Microsoft Store
# Then follow Ubuntu/Debian instructions above
```

### Python Environment Setup

#### Option 1: Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option 2: System-wide Installation
```bash
# Install dependencies globally
pip3 install -r requirements.txt
```

### GPU Support (Optional but Recommended)

#### CUDA Installation
```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Install CUDA-enabled PyTorch (replace cu118 with your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### FFmpeg Installation Methods

Use the provided installation script for automatic setup:
```bash
# Make script executable
chmod +x install_ffmpeg.sh

# Run installation script
./install_ffmpeg.sh

# Or specify installation method
./install_ffmpeg.sh apt      # Ubuntu/Debian
./install_ffmpeg.sh brew     # Homebrew
./install_ffmpeg.sh snap     # Snap packages
./install_ffmpeg.sh static   # Static build (universal)
```

## Performance Tips

1. **GPU Memory**: The tool automatically adjusts batch size based on available VRAM
2. **Model Selection**: 
   - `tiny`/`base`: Fast, good for quick drafts
   - `small`/`medium`: Balanced speed and accuracy
   - `large`: Best accuracy, requires significant GPU memory
3. **Dynamic Chunking**: Improves accuracy by avoiding cuts in the middle of speech
4. **Speculative Decoding**: Use `--fast` flag for 4-6x speedup with minimal accuracy loss

## Output

The transcription is saved as a text file with the same name as the input file, with `_transcription.txt` suffix.

Example: `audio.mp3` → `audio_transcription.txt`

## Transcription Cleanup (Post-Processing)

Clean up transcriptions by removing filler words, repetitions, and improving readability using LLMs.

### Installation

```bash
# Install cleanup dependencies
pip install -r requirements-cleanup.txt
```

### Basic Usage

```bash
# Clean up using Claude API (recommended)
export CLAUDE_API_KEY="your-api-key"
python cleanup_transcription.py audio_transcription.txt

# Clean up using OpenAI
export OPENAI_API_KEY="your-api-key"
python cleanup_transcription.py audio_transcription.txt --provider openai

# Local cleanup (no API needed)
python cleanup_transcription.py audio_transcription.txt --local-only
```

### Advanced Options

```bash
# Business style (formal, concise)
python cleanup_transcription.py transcript.txt --style business

# Casual style (natural conversation flow)
python cleanup_transcription.py transcript.txt --style casual

# Academic style (formal, structured)
python cleanup_transcription.py transcript.txt --style academic

# Preserve speaker labels
python cleanup_transcription.py transcript.txt --preserve-speakers

# Custom output file
python cleanup_transcription.py input.txt -o cleaned_output.txt
```

### Cleanup Features

1. **Filler Word Removal**: Removes "um", "uh", "えー", "あの" etc.
2. **Repetition Elimination**: Detects and removes repeated words/phrases
3. **Grammar Correction**: Fixes punctuation and grammar errors
4. **Style Formatting**: Adapts text to business, casual, or academic style
5. **Speaker Preservation**: Optionally maintains speaker labels
6. **Chunk Processing**: Handles long transcriptions efficiently

### API Setup

#### Method 1: Using .env File (Recommended)

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit .env file with your API keys**:
   ```bash
   # Open .env in your editor
   nano .env
   
   # Add your actual API keys
   CLAUDE_API_KEY=sk-ant-api03-your-actual-key-here
   OPENAI_API_KEY=sk-your-actual-openai-key-here
   ```

3. **Install additional dependencies**:
   ```bash
   pip install -r requirements-cleanup.txt
   ```

#### Method 2: Environment Variables

##### Claude API
1. Get API key from [Anthropic Console](https://console.anthropic.com/)
2. Set environment variable:
   ```bash
   export CLAUDE_API_KEY="sk-ant-api03-your-key"
   ```

##### OpenAI API
1. Get API key from [OpenAI Platform](https://platform.openai.com/)
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-your-key"
   ```

#### Method 3: Command Line

```bash
# Specify API key directly in command
python cleanup_transcription.py transcript.txt --api-key "sk-ant-api03-your-key"
```

#### Security Best Practices

- ✅ **Use .env files** for local development
- ✅ **Never commit API keys** to Git (they're in .gitignore)
- ✅ **Use environment variables** in production
- ✅ **Rotate keys regularly** (monthly recommended)
- ❌ **Never hardcode keys** in source code
- ❌ **Don't share keys** in chat/email

### Example Workflow

```bash
# 1. Transcribe audio
python transcribe_japanese.py meeting.m4a --model medium

# 2. Clean up transcription
python cleanup_transcription.py meeting_transcription.txt --style business

# Result: meeting_transcription_cleaned.txt
```

## Troubleshooting

### Common Issues and Solutions

#### 1. `ModuleNotFoundError: No module named '_ctypes'`
**Problem**: Python was compiled without libffi support.

**Solution**:
```bash
# Install libffi development library
sudo apt install -y libffi-dev

# Reinstall Python (if using pyenv)
pyenv uninstall 3.13.5
pyenv install 3.13.5
```

#### 2. `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
**Problem**: FFmpeg is not installed or not in PATH.

**Solution**:
```bash
# Use the installation script
./install_ffmpeg.sh

# Or install manually
sudo apt install -y ffmpeg  # Ubuntu/Debian
brew install ffmpeg         # macOS/Homebrew
```

#### 3. CUDA/GPU Issues
**Problem**: GPU not detected or CUDA errors.

**Solutions**:
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU-only mode if needed
CUDA_VISIBLE_DEVICES="" python transcribe_japanese.py audio.mp3
```

#### 4. Memory Issues
**Problem**: Out of memory errors during transcription.

**Solutions**:
```bash
# Use smaller model
python transcribe_japanese.py audio.mp3 --model base

# Reduce chunk duration
python transcribe_japanese.py audio.mp3 --chunk-duration 300

# Disable batch processing
python transcribe_japanese.py audio.mp3 --no-batch-processing
```

#### 5. Audio Format Issues
**Problem**: Unsupported audio format or corruption.

**Solutions**:
```bash
# Check file information
ffprobe audio_file

# Convert to supported format
ffmpeg -i input.format -c:a libmp3lame output.mp3
ffmpeg -i input.format -c:a aac output.m4a
```

#### 6. Performance Issues
**Problem**: Slow transcription speed.

**Solutions**:
```bash
# Enable fast mode
python transcribe_japanese.py audio.mp3 --fast

# Use smaller model for speed
python transcribe_japanese.py audio.mp3 --model small --fast

# Preprocess audio for better performance
ffmpeg -i input.m4a -af "highpass=f=200,lowpass=f=3000,loudnorm" -ar 16000 -ac 1 output.wav
```

### Getting Help

If you encounter issues not covered here:

1. **Check System Requirements**: Ensure your system meets all prerequisites
2. **Update Dependencies**: Run `pip install --upgrade -r requirements.txt`
3. **Check CLAUDE.md**: See the project guide for detailed configuration
4. **Create an Issue**: Report bugs on the project repository

### Environment Verification Script

Create a verification script to check your setup:
```bash
# Create verify_setup.py
cat > verify_setup.py << 'EOF'
#!/usr/bin/env python3
import sys
import subprocess
import importlib

def check_command(cmd):
    try:
        subprocess.run([cmd, '--version'], capture_output=True, check=True)
        return True
    except:
        return False

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except:
        return False

print("=== Environment Verification ===")
print(f"Python: {sys.version}")
print(f"FFmpeg: {'✓' if check_command('ffmpeg') else '✗'}")
print(f"FFprobe: {'✓' if check_command('ffprobe') else '✗'}")
print(f"Whisper: {'✓' if check_module('whisper') else '✗'}")
print(f"Torch: {'✓' if check_module('torch') else '✗'}")
print(f"Librosa: {'✓' if check_module('librosa') else '✗'}")

if check_module('torch'):
    import torch
    print(f"CUDA Available: {'✓' if torch.cuda.is_available() else '✗'}")
EOF

# Run verification
python verify_setup.py
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built on [OpenAI Whisper](https://github.com/openai/whisper)
- Uses [librosa](https://librosa.org/) for audio analysis
- Powered by [PyTorch](https://pytorch.org/)