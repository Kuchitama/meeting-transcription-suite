#!/usr/bin/env python3
"""
Transcription Cleanup Tool using LLMs (Claude, OpenAI, etc.)

This script cleans up transcriptions by:
- Removing filler words and meaningless utterances
- Eliminating repetitions
- Fixing grammar and punctuation
- Improving readability while preserving meaning
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Environment variable support
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    # dotenv not installed, continue without it
    pass

# Third-party imports
try:
    import anthropic
except ImportError:
    anthropic = None
    print("Warning: anthropic package not installed. Install with: pip install anthropic")

try:
    import openai
except ImportError:
    openai = None
    print("Warning: openai package not installed. Install with: pip install openai")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class TranscriptionCleaner:
    """Clean and refine transcription text using various methods"""
    
    # Japanese filler words and expressions
    JAPANESE_FILLERS = [
        "あの", "えーと", "えー", "その", "ま", "まあ", "なんか", 
        "ちょっと", "やっぱり", "とりあえず", "一応", "なんていうか",
        "えっと", "あー", "うーん", "そうですね", "ですね"
    ]
    
    # English filler words
    ENGLISH_FILLERS = [
        "um", "uh", "er", "ah", "like", "you know", "I mean", 
        "actually", "basically", "literally", "sort of", "kind of"
    ]
    
    def __init__(self, provider: str = "claude", api_key: Optional[str] = None):
        """
        Initialize the transcription cleaner
        
        Args:
            provider: LLM provider ("claude", "openai", "local")
            api_key: API key for the provider
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        
        # Initialize provider client
        if self.provider == "claude" and anthropic:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "openai" and openai:
            openai.api_key = self.api_key
            self.client = openai
        else:
            self.client = None
    
    def clean_with_llm(self, text: str, style: str = "business", 
                       preserve_speaker: bool = True) -> str:
        """
        Clean transcription using LLM
        
        Args:
            text: Raw transcription text
            style: Cleanup style ("business", "casual", "academic")
            preserve_speaker: Whether to preserve speaker labels
            
        Returns:
            Cleaned text
        """
        # Prepare prompt based on style
        prompts = {
            "business": self._get_business_prompt(text, preserve_speaker),
            "casual": self._get_casual_prompt(text, preserve_speaker),
            "academic": self._get_academic_prompt(text, preserve_speaker)
        }
        
        prompt = prompts.get(style, prompts["business"])
        
        # Call appropriate LLM
        if self.provider == "claude" and self.client:
            return self._call_claude(prompt)
        elif self.provider == "openai" and self.client:
            return self._call_openai(prompt)
        else:
            print(f"Warning: LLM provider '{self.provider}' not available. Using local cleanup.")
            return self.clean_locally(text)
    
    def _get_business_prompt(self, text: str, preserve_speaker: bool) -> str:
        """Generate business-style cleanup prompt"""
        speaker_instruction = "Preserve speaker labels if present." if preserve_speaker else "Remove speaker labels."
        
        return f"""Please clean up this meeting transcription for business use. 

Instructions:
1. Remove filler words (um, uh, えー, あの, etc.)
2. Remove meaningless repetitions
3. Fix grammar and punctuation
4. Keep technical terms and important details
5. Make it concise but preserve all key information
6. {speaker_instruction}
7. If the text is in Japanese, maintain formal business Japanese style
8. Format the output for easy reading

Transcription:
{text}

Cleaned version:"""
    
    def _get_casual_prompt(self, text: str, preserve_speaker: bool) -> str:
        """Generate casual-style cleanup prompt"""
        speaker_instruction = "Keep speaker labels if present." if preserve_speaker else "Remove speaker labels."
        
        return f"""Please clean up this transcription while keeping a natural conversational tone.

Instructions:
1. Remove excessive filler words but keep some for natural flow
2. Fix obvious errors but maintain conversational style
3. Remove only severe repetitions
4. {speaker_instruction}
5. Keep the personality and tone of speakers
6. Make it readable but not overly formal

Transcription:
{text}

Cleaned version:"""
    
    def _get_academic_prompt(self, text: str, preserve_speaker: bool) -> str:
        """Generate academic-style cleanup prompt"""
        speaker_instruction = "Maintain speaker attribution." if preserve_speaker else "Remove speaker labels."
        
        return f"""Please clean up this transcription for academic/research purposes.

Instructions:
1. Remove all filler words and verbal tics
2. Correct grammar to academic standards
3. Structure into clear, logical paragraphs
4. {speaker_instruction}
5. Preserve all factual content and technical details
6. Use formal academic language
7. Add appropriate punctuation for clarity

Transcription:
{text}

Cleaned version:"""
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API for text cleanup"""
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return ""
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API for text cleanup"""
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are a professional transcription editor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""
    
    def clean_locally(self, text: str) -> str:
        """
        Local cleanup without LLM
        Basic rule-based cleaning for when LLM is not available
        """
        # Remove common filler words
        cleaned = text
        
        # Remove Japanese fillers
        for filler in self.JAPANESE_FILLERS:
            # Remove filler at start of sentences or surrounded by punctuation
            cleaned = re.sub(f'(^|[。、\\s]){filler}([。、\\s]|$)', r'\1\2', cleaned)
        
        # Remove English fillers (case-insensitive)
        for filler in self.ENGLISH_FILLERS:
            cleaned = re.sub(f'\\b{filler}\\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove repeated words
        cleaned = self._remove_repetitions(cleaned)
        
        # Clean up extra spaces and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([。、,.])', r'\1', cleaned)
        cleaned = re.sub(r'([。、,.])\1+', r'\1', cleaned)
        
        return cleaned.strip()
    
    def _remove_repetitions(self, text: str) -> str:
        """Remove repeated words or phrases - enhanced version"""
        # First pass: Remove exact repeated phrases
        # Pattern: "認識をしてお願いいたします" repeated many times
        import re
        
        # Remove patterns that repeat more than 3 times consecutively
        # This handles cases like "認識をしてお願いいたします" repeated 100+ times
        cleaned = text
        
        # Find all phrases between Japanese punctuation or spaces
        segments = re.split(r'([。、\s]+)', cleaned)
        result_segments = []
        last_segment = ""
        repeat_count = 0
        
        for segment in segments:
            if segment.strip() and segment == last_segment:
                repeat_count += 1
                if repeat_count <= 2:  # Allow up to 2 repetitions
                    result_segments.append(segment)
            else:
                result_segments.append(segment)
                if segment.strip():
                    last_segment = segment
                    repeat_count = 0
        
        cleaned = ''.join(result_segments)
        
        # Second pass: Remove sentence-level repetitions
        # Split by Japanese sentence endings
        sentences = re.split(r'([。！？])', cleaned)
        cleaned_sentences = []
        last_sentence = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            ending = sentences[i + 1] if i + 1 < len(sentences) else ""
            
            if sentence.strip() and sentence.strip() != last_sentence:
                cleaned_sentences.append(sentence + ending)
                last_sentence = sentence.strip()
            elif not sentence.strip():
                cleaned_sentences.append(sentence + ending)
        
        cleaned = ''.join(cleaned_sentences)
        
        # Third pass: Remove word-level repetitions
        words = cleaned.split()
        cleaned_words = []
        
        i = 0
        while i < len(words):
            # Check for repeated sequences
            repeated = False
            for length in range(10, 0, -1):  # Check longer sequences first
                if i + 2 * length <= len(words):
                    seq1 = words[i:i+length]
                    seq2 = words[i+length:i+2*length]
                    if seq1 == seq2 and len(' '.join(seq1)) > 3:  # Ignore very short sequences
                        # Check if this pattern repeats even more
                        repeat_count = 2
                        j = i + 2 * length
                        while j + length <= len(words) and words[j:j+length] == seq1:
                            repeat_count += 1
                            j += length
                        
                        # Keep only first occurrence if repeated more than twice
                        if repeat_count > 2:
                            cleaned_words.extend(seq1)
                            i = j
                        else:
                            cleaned_words.extend(seq1 * 2)
                            i += 2 * length
                        repeated = True
                        break
            
            if not repeated:
                cleaned_words.append(words[i])
                i += 1
        
        return ' '.join(cleaned_words)
    
    def process_chunks(self, text: str, chunk_size: int = 2000) -> str:
        """
        Process long text in chunks
        
        Args:
            text: Long transcription text
            chunk_size: Size of each chunk in characters
            
        Returns:
            Cleaned text
        """
        # Split text into chunks at sentence boundaries
        chunks = self._split_into_chunks(text, chunk_size)
        cleaned_chunks = []
        
        # Process each chunk
        iterator = tqdm(chunks, desc="Processing chunks") if tqdm else chunks
        for chunk in iterator:
            cleaned = self.clean_with_llm(chunk)
            if cleaned:
                cleaned_chunks.append(cleaned)
            else:
                # Fallback to local cleaning if LLM fails
                cleaned_chunks.append(self.clean_locally(chunk))
            
            # Rate limiting
            if self.provider in ["claude", "openai"]:
                time.sleep(1)  # Avoid rate limits
        
        return '\n\n'.join(cleaned_chunks)
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks at sentence boundaries"""
        # Split by Japanese and English sentence endings
        sentences = re.split(r'([。！？\.\!\?]\s*)', text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            ending = sentences[i + 1] if i + 1 < len(sentences) else ""
            
            if len(current_chunk) + len(sentence) + len(ending) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ending
            else:
                current_chunk += sentence + ending
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Clean up transcription files using LLMs or local processing"
    )
    parser.add_argument("input_file", help="Input transcription file")
    parser.add_argument("-o", "--output", help="Output file (default: input_cleaned.txt)")
    parser.add_argument("--provider", choices=["claude", "openai", "local"], 
                       default="claude", help="LLM provider to use")
    parser.add_argument("--api-key", help="API key for LLM provider")
    parser.add_argument("--style", choices=["business", "casual", "academic"], 
                       default="business", help="Cleanup style")
    parser.add_argument("--preserve-speakers", action="store_true",
                       help="Preserve speaker labels in output")
    parser.add_argument("--chunk-size", type=int, default=2000,
                       help="Chunk size for processing long texts")
    parser.add_argument("--local-only", action="store_true",
                       help="Use only local processing (no LLM)")
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Initialize cleaner
    if args.local_only:
        cleaner = TranscriptionCleaner(provider="local")
    else:
        cleaner = TranscriptionCleaner(provider=args.provider, api_key=args.api_key)
    
    print(f"Processing transcription using {cleaner.provider}...")
    
    # Process text
    if args.local_only:
        cleaned_text = cleaner.clean_locally(text)
    elif len(text) > args.chunk_size * 2:
        # Process in chunks for long texts
        cleaned_text = cleaner.process_chunks(text, args.chunk_size)
    else:
        cleaned_text = cleaner.clean_with_llm(text, args.style, args.preserve_speakers)
    
    # Fallback to local if LLM fails
    if not cleaned_text and not args.local_only:
        print("LLM processing failed, falling back to local cleanup...")
        cleaned_text = cleaner.clean_locally(text)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"{input_path.stem}_cleaned.txt"
    
    # Write output
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"Cleaned transcription saved to: {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    
    # Show statistics
    original_length = len(text)
    cleaned_length = len(cleaned_text)
    reduction = (1 - cleaned_length / original_length) * 100 if original_length > 0 else 0
    
    print(f"\nStatistics:")
    print(f"Original length: {original_length:,} characters")
    print(f"Cleaned length: {cleaned_length:,} characters")
    print(f"Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()