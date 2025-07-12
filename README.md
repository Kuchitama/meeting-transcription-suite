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

## Installation

```bash
pip install -r requirements.txt
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

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for larger models)
- FFmpeg (for audio processing)

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

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built on [OpenAI Whisper](https://github.com/openai/whisper)
- Uses [librosa](https://librosa.org/) for audio analysis
- Powered by [PyTorch](https://pytorch.org/)