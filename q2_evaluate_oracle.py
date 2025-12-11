"""
Evaluate using context from dev_1200.json.

This script:
1. Loads dev_1200.json and extracts question, context, and answer
2. Formats context from dev_1200.json (list format) to text
3. Generates answer using Mistral API with the context
4. Evaluates using Exact Match and Cover Exact Match metrics
5. Outputs detailed results report
"""

import json
import os
import sys
from typing import Dict, List, Optional
from tqdm import tqdm

# Add DEXTER path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DEXTER-macos'))

from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
from dexter.utils.metrics.ExactMatch import ExactMatch
from dexter.utils.metrics.CoverExactMatch import CoverExactMatch


def load_dev_data(dev_path: str) -> List[Dict]:
    """Load dev data from JSON file.
    
    Args:
        dev_path: Path to dev_1200.json
        
    Returns:
        List of dev data items
    """
    print(f"Loading dev data from: {dev_path}")
    with open(dev_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from dev data")
    return data


def format_context_from_dev(context_data: List) -> str:
    """Format context from dev_1200.json format to text string.
    
    Args:
        context_data: Context in dev format: [[title, [sentences...]], ...]
        
    Returns:
        Formatted context string
    """
    if not context_data:
        return ""
    
    formatted_parts = []
    for item in context_data:
        if isinstance(item, list) and len(item) >= 2:
            title = item[0] if item[0] else ""
            sentences = item[1] if isinstance(item[1], list) else [str(item[1])]
            
            if title:
                # Format: "Title: sentence1. sentence2. ..."
                text = ". ".join(str(s).strip() for s in sentences if s)
                if text:
                    formatted_parts.append(f"{title}: {text}")
            else:
                # No title, just sentences
                text = ". ".join(str(s).strip() for s in sentences if s)
                if text:
                    formatted_parts.append(text)
    
    return "\n\n".join(formatted_parts)


def generate_answer(llm_engine, question: str, context: str) -> str:
    """Generate answer using Mistral API.
    
    Args:
        llm_engine: Initialized Mistral engine
        question: Question text
        context: Context text
        
    Returns:
        Generated answer
    """
    system_prompt = "You are a helpful assistant. Answer the question based on the given context. Provide only the answer without additional explanation."
    user_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        response = llm_engine.get_mistral_completion(system_prompt, user_prompt)
        # Extract answer (remove any extra formatting)
        answer = response.strip()
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""


def evaluate_dev_context(
    dev_path: str = "data/dev_1200.json",
    output_path: str = "dev_context_evaluation_results.json",
    model_name: str = "open-mistral-7b"
):
    """Main evaluation function using context from dev_1200.json.
    
    Args:
        dev_path: Path to dev_1200.json
        output_path: Path to save evaluation results
        model_name: Mistral model name
    """
    print("=" * 80)
    print("Dev Context Evaluation")
    print("=" * 80)
    
    # Load data
    dev_data = load_dev_data(dev_path)
    
    # Initialize LLM engine
    print("\nInitializing Mistral API engine...")
    config_instance = LLMEngineOrchestrator()
    llm_engine = config_instance.get_llm_engine(
        data="",
        llm_class="mistral",
        model_name=model_name
    )
    print("Mistral engine initialized successfully")
    
    # Initialize evaluation metrics
    exact_match_metric = ExactMatch()
    cover_exact_match_metric = CoverExactMatch()
    
    # Process each question
    print(f"\nProcessing {len(dev_data)} questions...")
    print("-" * 80)
    
    results = []
    total_exact_match = 0
    total_cover_exact_match = 0
    total_processed = 0
    total_skipped = 0
    
    for item in tqdm(dev_data, desc="Processing questions"):
        question_id = item.get("_id", "")
        question = item.get("question", "")
        context_data = item.get("context", [])
        gold_answer = item.get("answer", "")
        
        # Skip if essential data is missing
        if not question:
            print(f"Warning: Question is empty for {question_id}, skipping")
            total_skipped += 1
            continue
        
        if not context_data:
            print(f"Warning: Context is empty for {question_id}, skipping")
            total_skipped += 1
            continue
        
        if not gold_answer:
            print(f"Warning: Answer is empty for {question_id}, skipping")
            total_skipped += 1
            continue
        
        # Format context from dev format
        context_text = format_context_from_dev(context_data)
        
        if not context_text:
            print(f"Warning: Formatted context is empty for {question_id}, skipping")
            total_skipped += 1
            continue
        
        # Generate answer
        predicted_answer = generate_answer(llm_engine, question, context_text)
        
        if not predicted_answer:
            print(f"Warning: Failed to generate answer for question {question_id}, skipping")
            total_skipped += 1
            continue
        
        # Evaluate
        exact_match = exact_match_metric.evaluate(predicted_answer, gold_answer)
        cover_exact_match = cover_exact_match_metric.evaluate(predicted_answer, gold_answer)
        
        # Update statistics
        total_processed += 1
        if exact_match:
            total_exact_match += 1
        if cover_exact_match:
            total_cover_exact_match += 1
        
        # Store result
        result = {
            "question_id": question_id,
            "question": question,
            "context": context_text,
            "predicted_answer": predicted_answer,
            "gold_answer": gold_answer,
            "exact_match": bool(exact_match),
            "cover_exact_match": bool(cover_exact_match)
        }
        results.append(result)
        
        # Print progress every 10 questions
        if total_processed % 10 == 0:
            current_em = total_exact_match / total_processed if total_processed > 0 else 0
            current_cem = total_cover_exact_match / total_processed if total_processed > 0 else 0
            print(f"\nProgress: {total_processed} questions processed")
            print(f"  Current Exact Match: {current_em:.4f}")
            print(f"  Current Cover Exact Match: {current_cem:.4f}")
    
    # Calculate final metrics
    final_exact_match = total_exact_match / total_processed if total_processed > 0 else 0.0
    final_cover_exact_match = total_cover_exact_match / total_processed if total_processed > 0 else 0.0
    
    # Prepare output
    output = {
        "experiment_name": "Dev Context Evaluation",
        "experiment_description": "Evaluate using context from dev_1200.json",
        "model": {
            "name": model_name,
            "type": "Mistral API"
        },
        "data_source": {
            "dev_file": dev_path
        },
        "results": {
            "total_questions": len(dev_data),
            "processed_questions": total_processed,
            "skipped_questions": total_skipped,
            "exact_match": float(final_exact_match),
            "cover_exact_match": float(final_cover_exact_match),
            "exact_match_count": total_exact_match,
            "cover_exact_match_count": total_cover_exact_match
        },
        "detailed_results": results
    }
    
    # Save results
    print(f"\n{'=' * 80}")
    print("Saving results...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("Evaluation Summary")
    print(f"{'=' * 80}")
    print(f"Total questions in dev data: {len(dev_data)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Skipped: {total_skipped}")
    print(f"\nMetrics:")
    print(f"  Exact Match: {final_exact_match:.4f} ({total_exact_match}/{total_processed})")
    print(f"  Cover Exact Match: {final_cover_exact_match:.4f} ({total_cover_exact_match}/{total_processed})")
    print(f"\nResults saved to: {output_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate using context from dev_1200.json")
    parser.add_argument("--dev_path", type=str, default="data/dev_1200.json",
                        help="Path to dev_1200.json")
    parser.add_argument("--output_path", type=str, default="dev_context_evaluation_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--model_name", type=str, default="open-mistral-7b",
                        help="Mistral model name (e.g., 'open-mistral-7b' for Mistral API or 'mistralai/Mistral-7B-Instruct-v0.1' for Hugging Face)")
    
    args = parser.parse_args()
    
    evaluate_dev_context(
        dev_path=args.dev_path,
        output_path=args.output_path,
        model_name=args.model_name
    )

