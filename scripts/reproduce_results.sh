#!/bin/bash
# Run the full benchmark and ablations in under 2 hours (using reduced sample ratios)
echo "Running main benchmark..."
python experiments/run_attack_suite.py

echo "Running ablation studies..."
python experiments/run_ablation.py

echo "All results saved in experiments/results/"