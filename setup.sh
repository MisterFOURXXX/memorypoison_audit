#!/bin/bash
set -e

echo "Cloning HotpotQA dataset..."
git clone https://github.com/hotpotqa/hotpot.git datasets/hotpotqa || echo "HotpotQA already exists."

echo "Cloning LongMemEval dataset..."
git clone https://github.com/salesforce/LongMemEval.git datasets/longmemeval || echo "LongMemEval already exists."

echo "Downloading HotpotQA fullwiki data (if not present)..."
cd datasets/hotpotqa
wget -nc http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget -nc http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
cd ../..

echo "Dataset setup complete."