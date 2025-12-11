"""
Re-evaluate results using fixed CoverExactMatch and ExactMatch metrics.

This script:
1. Loads existing evaluation results JSON files
2. Re-calculates exact_match and cover_exact_match using fixed metrics
3. Updates overall statistics
4. Saves updated results
"""

import json
import sys
import os

# Add DEXTER path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DEXTER-macos'))

from dexter.utils.metrics.ExactMatch import ExactMatch
from dexter.utils.metrics.CoverExactMatch import CoverExactMatch


def reevaluate_json_file(input_path: str, output_path: str = None):
    """Re-evaluate a JSON results file with fixed metrics.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save updated JSON file (if None, overwrites input)
    """
    if output_path is None:
        output_path = input_path
    
    print(f"Loading results from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize metrics
    exact_match_metric = ExactMatch()
    cover_exact_match_metric = CoverExactMatch()
    
    # Re-evaluate detailed results
    print("Re-evaluating detailed results...")
    total_exact_match = 0
    total_cover_exact_match = 0
    total_processed = 0
    
    detailed_results = data.get("detailed_results", [])
    
    for result in detailed_results:
        predicted_answer = result.get("predicted_answer", "")
        gold_answer = result.get("gold_answer", "")
        
        if not predicted_answer or not gold_answer:
            continue
        
        # Re-calculate metrics
        exact_match = exact_match_metric.evaluate(predicted_answer, gold_answer)
        cover_exact_match = cover_exact_match_metric.evaluate(predicted_answer, gold_answer)
        
        # Update result
        result["exact_match"] = bool(exact_match)
        result["cover_exact_match"] = bool(cover_exact_match)
        
        # Update statistics
        total_processed += 1
        if exact_match:
            total_exact_match += 1
        if cover_exact_match:
            total_cover_exact_match += 1
    
    # Update overall statistics
    if "results" in data:
        results = data["results"]
        results["exact_match_count"] = total_exact_match
        results["cover_exact_match_count"] = total_cover_exact_match
        
        if total_processed > 0:
            results["exact_match"] = float(total_exact_match) / total_processed
            results["cover_exact_match"] = float(total_cover_exact_match) / total_processed
        else:
            results["exact_match"] = 0.0
            results["cover_exact_match"] = 0.0
        
        results["processed_questions"] = total_processed
    
    # Save updated results
    print(f"Saving updated results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Re-evaluation Summary for: {os.path.basename(input_path)}")
    print(f"{'=' * 80}")
    if "results" in data:
        results = data["results"]
        print(f"Total questions: {results.get('total_questions', 'N/A')}")
        print(f"Processed questions: {total_processed}")
        print(f"\nUpdated Metrics:")
        print(f"  Exact Match: {results['exact_match']:.4f} ({total_exact_match}/{total_processed})")
        print(f"  Cover Exact Match: {results['cover_exact_match']:.4f} ({total_cover_exact_match}/{total_processed})")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-evaluate results using fixed metrics")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output JSON file (default: overwrites input)")
    
    args = parser.parse_args()
    
    reevaluate_json_file(args.input, args.output)

