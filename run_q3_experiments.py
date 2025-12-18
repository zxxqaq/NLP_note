#!/usr/bin/env python3
"""
Script to run all Q3 experiments with updated prompt (same as Q1/Q2).
Runs three experiments:
1. Top-1 + Random 1
2. Top-3 + Random 1
3. Top-3 + Random 3
"""

import os
import sys
import subprocess

def check_api_key():
    """Check if MISTRAL_API_KEY is set."""
    api_key = os.environ.get('MISTRAL_API_KEY')
    if not api_key:
        print("ERROR: MISTRAL_API_KEY environment variable is not set.")
        print("Please set it using: export MISTRAL_API_KEY='your_api_key'")
        return False
    print(f"MISTRAL_API_KEY is set (length: {len(api_key)})")
    return True

def run_experiment(top_k, random_k, output_path):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Running: Top-{top_k} + Random {random_k}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        'q3_evaluate_retrieval_with_random.py',
        '--top_k', str(top_k),
        '--random_k', str(random_k),
        '--output_path', output_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Experiment completed: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed: {output_path}")
        print(f"Error: {e}")
        return False

def main():
    print("="*80)
    print("Q3 Experiments with Updated Prompt (Same as Q1/Q2)")
    print("="*80)
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    experiments = [
        (1, 1, 'data/q3_top_1_mix_random_1_results.json'),
        (3, 1, 'data/q3_top_3_mix_random_1_results.json'),
        (3, 3, 'data/q3_top_3_mix_random_3_results.json'),
    ]
    
    results = []
    for top_k, random_k, output_path in experiments:
        success = run_experiment(top_k, random_k, output_path)
        results.append((top_k, random_k, output_path, success))
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for top_k, random_k, output_path, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{status}: Top-{top_k} + Random {random_k} -> {output_path}")
    
    all_success = all(success for _, _, _, success in results)
    if all_success:
        print("\n✓ All experiments completed successfully!")
    else:
        print("\n✗ Some experiments failed. Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()

