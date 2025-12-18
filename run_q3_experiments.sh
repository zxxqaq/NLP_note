#!/bin/bash

# Script to run all Q3 experiments with updated prompt
# Make sure MISTRAL_API_KEY is set before running

echo "=========================================="
echo "Running Q3 Experiments with Updated Prompt"
echo "=========================================="
echo ""

# Check if API key is set
if [ -z "$MISTRAL_API_KEY" ]; then
    echo "ERROR: MISTRAL_API_KEY environment variable is not set."
    echo "Please set it using: export MISTRAL_API_KEY='your_api_key'"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

echo "Experiment 1: Top-1 + Random 1"
echo "----------------------------------------"
python q3_evaluate_retrieval_with_random.py \
    --top_k 1 \
    --random_k 1 \
    --output_path data/q3_top_1_mix_random_1_results.json

echo ""
echo "Experiment 2: Top-3 + Random 1"
echo "----------------------------------------"
python q3_evaluate_retrieval_with_random.py \
    --top_k 3 \
    --random_k 1 \
    --output_path data/q3_top_3_mix_random_1_results.json

echo ""
echo "Experiment 3: Top-3 + Random 3"
echo "----------------------------------------"
python q3_evaluate_retrieval_with_random.py \
    --top_k 3 \
    --random_k 3 \
    --output_path data/q3_top_3_mix_random_3_results.json

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

