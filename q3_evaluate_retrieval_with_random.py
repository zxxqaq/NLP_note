"""
Evaluate retrieval results with top-1 context + random context.

This script:
1. Loads retrieval results (question_id -> context_id with scores)
2. For each question, finds the top-1 context (highest score)
3. Randomly selects one unrelated context from corpus (different from top-1)
4. Combines top-1 context + random context as input
5. Retrieves question from dev_1200.json
6. Generates answer using Mistral API
7. Evaluates using Exact Match and Cover Exact Match metrics
8. Outputs detailed results report
"""

import json
import os
import sys
import random
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

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


def get_random_context(corpus_data: Dict[str, Dict], exclude_context_id: str, seed: Optional[int] = None) -> Tuple[str, Dict]:
    """Get a random context from corpus, excluding the specified context_id.
    
    Args:
        corpus_data: Dictionary mapping context_id to context data
        exclude_context_id: Context ID to exclude from random selection
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Tuple of (random_context_id, context_data)
    """
    if seed is not None:
        random.seed(seed)
    
    # Get all context IDs except the excluded one
    available_contexts = [cid for cid in corpus_data.keys() if cid != exclude_context_id]
    
    if not available_contexts:
        # If no other contexts available, return None
        return None, None
    
    # Randomly select one
    random_context_id = random.choice(available_contexts)
    random_context_data = corpus_data[random_context_id]
    
    return random_context_id, random_context_data


def combine_contexts(top1_context: str, random_context: str) -> str:
    """Combine top-1 context and random context for LLM input.
    
    Args:
        top1_context: Formatted top-1 context text
        random_context: Formatted random context text
        
    Returns:
        Combined context string
    """
    return f"Relevant Context: {top1_context}\n\nUnrelated Context: {random_context}"


def generate_answer(llm_engine, question: str, context: str) -> str:
    """Generate answer using Mistral API.
    
    Args:
        llm_engine: Initialized Mistral engine
        question: Question text
        context: Context text (combined top-1 + random)
        
    Returns:
        Generated answer
    """
    system_prompt = "You are a helpful assistant. Answer the question based on the given context. Focus on the relevant context and ignore unrelated information. Provide only the answer without additional explanation."
    user_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        response = llm_engine.get_mistral_completion(system_prompt, user_prompt)
        # Extract answer (remove any extra formatting)
        answer = response.strip()
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""


def evaluate_retrieval_with_random(
    retrieval_path: str = "data/retrieval_results.json",
    dev_path: str = "data/dev_1200.json",
    corpus_path: str = "data/wiki_musique_corpus.json",
    output_path: str = "retrieval_with_random_evaluation_results.json",
    model_name: str = "open-mistral-7b",
    random_seed: Optional[int] = None
):
    """Main evaluation function with top-1 + random context.
    
    Args:
        retrieval_path: Path to retrieval_results.json
        dev_path: Path to dev_1200.json
        corpus_path: Path to wiki_musique_corpus.json
        output_path: Path to save evaluation results
        model_name: Mistral model name
        random_seed: Random seed for reproducibility (optional)
    """
    print("=" * 80)
    print("Retrieval Results Evaluation (Top-1 + Random Context)")
    print("=" * 80)
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Random seed set to: {random_seed}")
    
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
    print("Using top-1 context + 1 random unrelated context per question")
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
        
        # Get top-1 context
        top_contexts = get_top_k_context(question_id, retrieval_data, k=1)
        
        if not top_contexts:
            print(f"Warning: No contexts found for question {question_id}, skipping")
            continue
        
        top_context_id, top_score = top_contexts[0]
        
        # Get top-1 context from corpus
        if top_context_id not in corpus_data:
            print(f"Warning: Context ID {top_context_id} not found in corpus, skipping")
            continue
        
        top1_context_data = corpus_data[top_context_id]
        top1_context_text = format_context_for_llm(top1_context_data)
        
        # Get random unrelated context
        random_context_id, random_context_data = get_random_context(
            corpus_data, 
            exclude_context_id=top_context_id,
            seed=random_seed
        )
        
        if random_context_id is None:
            print(f"Warning: Could not find random context for question {question_id}, skipping")
            continue
        
        random_context_text = format_context_for_llm(random_context_data)
        
        # Combine contexts
        combined_context = combine_contexts(top1_context_text, random_context_text)
        
        # Generate answer
        predicted_answer = generate_answer(llm_engine, question, combined_context)
        
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
            "top1_context_id": top_context_id,
            "top1_context_title": top1_context_data.get("title", ""),
            "top1_context_text": top1_context_text,
            "top1_retrieval_score": top_score,
            "random_context_id": random_context_id,
            "random_context_title": random_context_data.get("title", ""),
            "random_context_text": random_context_text,
            "combined_context": combined_context,
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
        "experiment_name": "Retrieval Results Evaluation (Top-1 + Random Context)",
        "experiment_description": "Evaluate top-1 retrieved context + random unrelated context using Mistral API",
        "model": {
            "name": model_name,
            "type": "Mistral API"
        },
        "retrieval_config": {
            "top_k": 1,
            "random_context": True,
            "random_seed": random_seed,
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
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval results with top-1 + random context using Mistral API")
    parser.add_argument("--retrieval_path", type=str, default="data/retrieval_results.json",
                        help="Path to retrieval_results.json")
    parser.add_argument("--dev_path", type=str, default="data/dev_1200.json",
                        help="Path to dev_1200.json")
    parser.add_argument("--corpus_path", type=str, default="data/wiki_musique_corpus.json",
                        help="Path to wiki_musique_corpus.json")
    parser.add_argument("--output_path", type=str, default="retrieval_with_random_evaluation_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--model_name", type=str, default="open-mistral-7b",
                        help="Mistral model name (e.g., 'open-mistral-7b' for Mistral API or 'mistralai/Mistral-7B-Instruct-v0.1' for Hugging Face)")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility (optional)")
    
    args = parser.parse_args()
    
    evaluate_retrieval_with_random(
        retrieval_path=args.retrieval_path,
        dev_path=args.dev_path,
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        model_name=args.model_name,
        random_seed=args.random_seed
    )

