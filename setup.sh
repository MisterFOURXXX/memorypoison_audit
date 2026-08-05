#!/bin/bash
sed -i 's/archive.ubuntu.com/mirrors.kernel.org/g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y libaio-dev -qq
apt-get update && apt-get install -y ffmpeg

set -e
# PyTorch with CUDA
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Python packages
pip install -r requirements.txt

pip uninstall -y sentence-transformers huggingface_hub torchcodec

pip install --upgrade sentence-transformers huggingface_hub datasets chromadb scikit-learn

# Upgrade Python packages
# pip install --upgrade sentence-transformers huggingface_hub datasets chromadb scikit-learn numpy pandas plotly umap-learn pyyaml tqdm seaborn matplotlib

echo "Environment setup complete!"

cd /kaggle/working/memorypoison_audit/datasets

echo "Downloading LongMemEval dataset from HuggingFace..."
wget -nc https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json -P longmemeval/data/
wget -nc https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -P longmemeval/data/
wget -nc https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json -P longmemeval/data/

echo "Dataset setup complete."