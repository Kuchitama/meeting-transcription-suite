# Whisper文字起こし精度改善ガイド

## 改善版スクリプトの主な変更点

### 1. モデルサイズの変更
- `base` → `large` モデルに変更（より高精度）
- モデルサイズオプション: tiny < base < small < medium < large < large-v2 < large-v3
- 大きいモデルほど精度が高いが、処理時間も長くなる

### 2. 追加パラメータの設定
- **temperature**: 0.0に設定（より確実な予測）
- **initial_prompt**: 日本語会話であることを明示
- **condition_on_previous_text**: 前の文脈を考慮
- **vad_filter**: 音声活動検出で無音部分を除去
- **punctuations**: 日本語の句読点を適切に設定

### 3. GPU対応
- CUDAが利用可能な場合は自動的にGPUを使用
- 処理速度が大幅に向上

## 使用方法

### 基本的な使用方法
```bash
# デフォルト（largeモデル）で実行
python3 transcribe_japanese_improved.py

# モデルサイズを指定
python3 transcribe_japanese_improved.py --model medium

# ファイルパスを指定
python3 transcribe_japanese_improved.py --path /path/to/audio.m4a
```

### さらなる精度向上のための提案

1. **音声品質の改善**
   - 録音時のノイズを減らす
   - 適切なマイク距離を保つ
   - 音量レベルを一定に保つ

2. **前処理の追加**
   ```bash
   # ffmpegで音声を前処理（ノイズ除去、正規化）
   ffmpeg -i input.m4a -af "highpass=f=200,lowpass=f=3000,volume=2" output.m4a
   ```

3. **後処理の実装**
   - 専門用語辞書による置換
   - 文脈に基づく修正
   - 句読点の調整

4. **複数モデルの併用**
   - 異なるモデルで複数回実行
   - 結果を比較・統合

5. **ファインチューニング**
   - 特定の話者や専門分野に特化したモデルの作成
   - ただし、これには追加の学習データが必要

## パフォーマンス比較

| モデル | 精度 | 速度 | メモリ使用量 |
|--------|------|------|--------------|
| tiny   | ★☆☆ | ★★★ | ~1GB        |
| base   | ★★☆ | ★★☆ | ~1GB        |
| small  | ★★☆ | ★★☆ | ~2GB        |
| medium | ★★★ | ★☆☆ | ~5GB        |
| large  | ★★★ | ★☆☆ | ~10GB       |

## トラブルシューティング

### メモリ不足エラー
- より小さいモデルを使用
- バッチサイズを調整
- GPUメモリが不足の場合はCPUモードで実行

### 処理が遅い
- GPUの使用を確認
- より小さいモデルを検討
- 音声ファイルを分割して処理

### 特定の単語が認識されない
- initial_promptに専門用語を含める
- 後処理で辞書置換を実装
- より大きいモデルを使用