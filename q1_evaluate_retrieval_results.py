"""
Evaluate retrieval results using Mistral API.

This script:
1. Loads retrieval results (question_id -> context_id with scores)
2. For each question, finds the top-k contexts (default: top-3, highest scores)
3. Retrieves question from dev_1200.json and context from wiki_musique_corpus.json
4. Combines top-k contexts and generates answer using Mistral API
5. Evaluates using Exact Match and Cover Exact Match metrics
6. Outputs detailed results report
"""

import json
import os
import sys
from typing import Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(desc)
        return iterable

# Add DEXTER path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DEXTER-macos'))

from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
from dexter.utils.metrics.ExactMatch import ExactMatch
from dexter.utils.metrics.CoverExactMatch import CoverExactMatch


def load_retrieval_results(retrieval_path: str) -> Dict[str, Dict[str, float]]:
    """Load retrieval results from JSON file.
    
    Args:
        retrieval_path: Path to retrieval_results.json
        
    Returns:
        Dictionary mapping question_id to {context_id: score}
    """
    print(f"Loading retrieval results from: {retrieval_path}")
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} question-retrieval pairs")
    return data


def load_dev_data(dev_path: str) -> Dict[str, Dict]:
    """Load dev data and create a mapping from _id to question/answer.
    
    Args:
        dev_path: Path to dev_1200.json
        
    Returns:
        Dictionary mapping _id to {question, answer, ...}
    """
    print(f"Loading dev data from: {dev_path}")
    with open(dev_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mapping from _id to data
    id_to_data = {}
    for item in data:
        question_id = item.get("_id")
        if question_id:
            id_to_data[question_id] = {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "full_data": item
            }
    
    print(f"Loaded {len(id_to_data)} questions from dev data")
    return id_to_data


def load_corpus(corpus_path: str) -> Dict[str, Dict]:
    """Load corpus and create a mapping from context_id to context text.
    
    Args:
        corpus_path: Path to wiki_musique_corpus.json
        
    Returns:
        Dictionary mapping context_id to {title, text}
    """
    print(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mapping from context_id to context data
    id_to_context = {}
    for context_id, context_data in data.items():
        id_to_context[context_id] = {
            "title": context_data.get("title", ""),
            "text": context_data.get("text", "")
        }
    
    print(f"Loaded {len(id_to_context)} contexts from corpus")
    return id_to_context


def get_top_k_context(question_id: str, retrieval_data: Dict[str, Dict[str, float]], k: int = 1) -> List[tuple]:
    """Get top-k context IDs for a question based on retrieval scores.
    
    Args:
        question_id: Question ID
        retrieval_data: Retrieval results dictionary
        k: Number of top contexts to return (default: 1)
        
    Returns:
        List of tuples (context_id, score) sorted by score (descending)
    """
    if question_id not in retrieval_data:
        return []
    
    # Get all context-score pairs for this question
    context_scores = retrieval_data[question_id]
    
    # Sort by score (descending, higher score is better)
    sorted_contexts = sorted(
        context_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return top-k
    return sorted_contexts[:k]


def format_context_for_llm(context_data: Dict) -> str:
    """Format context data for LLM input.
    
    Args:
        context_data: Dictionary with 'title' and 'text' keys
        
    Returns:
        Formatted context string
    """
    title = context_data.get("title", "")
    text = context_data.get("text", "")
    
    if title:
        return f"{title}: {text}"
    else:
        return text


def generate_answer(llm_engine, question: str, context: str) -> str:
    """Generate answer using Mistral API.
    
    Args:
        llm_engine: Initialized Mistral engine
        question: Question text
        context: Context text
        
    Returns:
        Generated answer
    """
    system_prompt = "You are a helpful assistant. Answer the question using the given context and your knowledge. Provide only the answer without additional explanation."
    user_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        response = llm_engine.get_mistral_completion(system_prompt, user_prompt)
        # Extract answer (remove any extra formatting)
        answer = response.strip()
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""


def evaluate_retrieval_results(
    retrieval_path: str = "data/retrieval_results.json",
    dev_path: str = "data/dev_1200.json",
    corpus_path: str = "data/wiki_musique_corpus.json",
    output_path: str = "retrieval_evaluation_results.json",
    model_name: str = "open-mistral-7b",
    top_k: int = 3
):
    """Main evaluation function.
    
    Args:
        retrieval_path: Path to retrieval_results.json
        dev_path: Path to dev_1200.json
        corpus_path: Path to wiki_musique_corpus.json
        output_path: Path to save evaluation results
        model_name: Mistral model name
        top_k: Number of top contexts to use (default: 1)
    """
    print("=" * 80)
    print("Retrieval Results Evaluation")
    print("=" * 80)
    
    # Load data
    retrieval_data = load_retrieval_results(retrieval_path)
    dev_data = load_dev_data(dev_path)
    corpus_data = load_corpus(corpus_path)
    
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
    print(f"\nProcessing {len(retrieval_data)} questions...")
    print(f"Using top-{top_k} context(s) per question")
    print("-" * 80)
    
    results = []
    total_exact_match = 0
    total_cover_exact_match = 0
    total_processed = 0
    
    for question_id, context_scores in tqdm(retrieval_data.items(), desc="Processing questions"):
        # Get question and answer from dev data
        if question_id not in dev_data:
            print(f"Warning: Question ID {question_id} not found in dev data, skipping")
            continue
        
        question = dev_data[question_id]["question"]
        gold_answer = dev_data[question_id]["answer"]
        
        if not question or not gold_answer:
            print(f"Warning: Question or answer is empty for {question_id}, skipping")
            continue
        
        # Get top-k contexts
        top_contexts = get_top_k_context(question_id, retrieval_data, k=top_k)
        
        if not top_contexts:
            print(f"Warning: No contexts found for question {question_id}, skipping")
            continue
        
        # Choose logic based on top_k value
        if top_k == 1:
            # ===== TOP-1 LOGIC =====
            top_context_id, top_score = top_contexts[0]
            
            # Get context from corpus
            if top_context_id not in corpus_data:
                print(f"Warning: Context ID {top_context_id} not found in corpus, skipping")
                continue
            
            context_data = corpus_data[top_context_id]
            context_text = format_context_for_llm(context_data)
            top_context_ids = [top_context_id]
            top_scores = [top_score]
            # ===== END TOP-1 LOGIC =====
        else:
            # ===== TOP-K LOGIC (for k > 1) =====
            # Combine top-k contexts
            context_parts = []
            top_context_ids = []
            top_scores = []
            
            for context_id, score in top_contexts:
                if context_id not in corpus_data:
                    print(f"Warning: Context ID {context_id} not found in corpus, skipping this context")
                    continue
                
                context_data_item = corpus_data[context_id]
                formatted_context = format_context_for_llm(context_data_item)
                context_parts.append(formatted_context)
                top_context_ids.append(context_id)
                top_scores.append(score)
            
            if not context_parts:
                print(f"Warning: No valid contexts found for question {question_id}, skipping")
                continue
            
            # Combine all contexts with separator
            context_text = "\n\n---\n\n".join(context_parts)
            top_context_id = top_context_ids[0]  # Use first context ID for reference
            top_score = top_scores[0]  # Use first score for reference
            context_data = corpus_data[top_context_id]  # For backward compatibility
            # ===== END TOP-K LOGIC =====
        
        # Generate answer
        predicted_answer = generate_answer(llm_engine, question, context_text)
        
        if not predicted_answer:
            print(f"Warning: Failed to generate answer for question {question_id}, skipping")
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
            "context_id": top_context_id,  # Primary context ID (for backward compatibility)
            "context_ids": top_context_ids,  # All context IDs used (top-3)
            "context_titles": [corpus_data[cid].get("title", "") for cid in top_context_ids],
            "context_text": context_text,
            "retrieval_scores": top_scores,  # All scores (top-3)
            "retrieval_score": top_score,  # Primary score (for backward compatibility)
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
        "experiment_name": "Retrieval Results Evaluation",
        "experiment_description": f"Evaluate top-{top_k} retrieved contexts using Mistral API",
        "model": {
            "name": model_name,
            "type": "Mistral API"
        },
        "retrieval_config": {
            "top_k": top_k,
            "retrieval_file": retrieval_path
        },
        "results": {
            "total_questions": len(retrieval_data),
            "processed_questions": total_processed,
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
    print(f"Total questions in retrieval results: {len(retrieval_data)}")
    print(f"Successfully processed: {total_processed}")
    print(f"\nMetrics:")
    print(f"  Exact Match: {final_exact_match:.4f} ({total_exact_match}/{total_processed})")
    print(f"  Cover Exact Match: {final_cover_exact_match:.4f} ({total_cover_exact_match}/{total_processed})")
    print(f"\nResults saved to: {output_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval results using Mistral API")
    parser.add_argument("--retrieval_path", type=str, default="data/retrieval_results.json",
                        help="Path to retrieval_results.json")
    parser.add_argument("--dev_path", type=str, default="data/dev_1200.json",
                        help="Path to dev_1200.json")
    parser.add_argument("--corpus_path", type=str, default="data/wiki_musique_corpus.json",
                        help="Path to wiki_musique_corpus.json")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save evaluation results (default: auto-generated based on top_k)")
    parser.add_argument("--model_name", type=str, default="open-mistral-7b",
                        help="Mistral model name (e.g., 'open-mistral-7b' for Mistral API or 'mistralai/Mistral-7B-Instruct-v0.1' for Hugging Face)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top contexts to use (default: 3)")
    
    args = parser.parse_args()
    
    # Auto-generate output path if not specified
    if args.output_path is None:
        args.output_path = f"data/q1_top_{args.top_k}_results.json"
    
    evaluate_retrieval_results(
        retrieval_path=args.retrieval_path,
        dev_path=args.dev_path,
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        model_name=args.model_name,
        top_k=args.top_k
    )



