# Whisper文字起こし精度向上のためのTips

## 処理速度の改善方法

### 1. GPUの利用
```bash
# NVIDIA GPUがある場合、PyTorchのCUDA版をインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 音声ファイルの分割処理
```bash
# ffmpegで音声を10分ごとに分割
ffmpeg -i input.m4a -f segment -segment_time 600 -c copy output_%03d.m4a
```

### 3. より高速な代替手段
- **whisper.cpp**: C++実装版（より高速）
  ```bash
  git clone https://github.com/ggerganov/whisper.cpp
  cd whisper.cpp
  make
  ./main -m models/ggml-base.bin -l ja -f input.m4a
  ```

- **faster-whisper**: CTranslate2を使用した高速版
  ```bash
  pip install faster-whisper
  ```

## 精度向上のための前処理

### 1. 音声品質の改善
```bash
# ノイズ除去と音量正規化
ffmpeg -i input.m4a -af "highpass=f=200,lowpass=f=3000,loudnorm" -ar 16000 output.wav

# モノラルに変換（処理速度向上）
ffmpeg -i input.m4a -ac 1 output_mono.m4a
```

### 2. 音声の前処理スクリプト
```python
import subprocess

def preprocess_audio(input_file, output_file):
    """音声ファイルの前処理"""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', 'highpass=f=200,lowpass=f=3000,loudnorm',
        '-ar', '16000',  # サンプリングレート16kHz
        '-ac', '1',      # モノラル
        '-c:a', 'pcm_s16le',  # 16bit PCM
        output_file
    ]
    subprocess.run(cmd)
```

## 後処理による精度向上

### 1. 専門用語の置換辞書
```python
# 専門用語辞書の例
term_dict = {
    "DV": "データベース",
    "DEP": "DEP",
    "ビックエリー": "BigQuery",
    "クロードコード": "Claude Code",
    "エスエー": "SA (Solutions Architect)",
    # ... その他の専門用語
}

def post_process_text(text):
    """後処理で専門用語を修正"""
    for wrong, correct in term_dict.items():
        text = text.replace(wrong, correct)
    return text
```

### 2. 句読点の修正
```python
import re

def fix_punctuation(text):
    """日本語の句読点を修正"""
    # 文末の修正
    text = re.sub(r'([。、])\s+', r'\1', text)
    # 重複する句読点の削除
    text = re.sub(r'[。、]{2,}', '。', text)
    return text
```

## 実用的な改善策

### 1. 録音時の工夫
- マイクを話者に近づける
- 静かな環境で録音
- 高品質な録音設定を使用（最低44.1kHz/16bit）

### 2. 段階的な処理
1. まずtinyモデルで全体を把握
2. 重要な部分だけlargeモデルで再処理
3. 結果を統合

### 3. 複数モデルの併用
```python
# 複数モデルで処理して最良の結果を選択
results = []
for model_size in ['base', 'small', 'medium']:
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file, language='ja')
    results.append(result)
# 結果を比較・統合
```

## 推奨設定

長時間の会議録音（1時間以上）の場合：
1. **モデル**: small または medium
2. **前処理**: 音声を30分ごとに分割
3. **temperature**: 0.0（最も確実な予測）
4. **beam_size**: 5以上
5. **initial_prompt**: 会議の文脈を含める

これらの方法を組み合わせることで、文字起こしの精度と処理速度を大幅に改善できます。