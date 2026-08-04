#!/bin/bash
apt-get update && apt-get install -y ffmpeg

set -e

echo "Cloning HotpotQA dataset..."
git clone https://github.com/hotpotqa/hotpot || echo "HotpotQA already exists."

echo "Cloning LongMemEval dataset..."
git clone https://github.com/xiaowu0162/LongMemEval || echo "LongMemEval already exists."

echo "Dataset setup complete."

# PyTorch with CUDA
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Python packages
pip install -r requirements.txt

echo "Environment setup complete!"