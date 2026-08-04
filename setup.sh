#!/bin/bash
set -e

echo "Cloning HotpotQA dataset..."
git clone https://github.com/hotpotqa/hotpot.git datasets/hotpotqa || echo "HotpotQA already exists."

echo "Cloning LongMemEval dataset..."
git clone https://github.com/salesforce/LongMemEval.git datasets/longmemeval || echo "LongMemEval already exists."

echo "Dataset setup complete."

# PyTorch with CUDA
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Python packages
pip install -r requirements.txt

echo "Environment setup complete!"