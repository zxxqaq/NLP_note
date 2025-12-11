# Scripts Usage Guide

This guide explains how to run the three evaluation scripts and documents the fix made to the Cover Exact Match metric.

## Scripts Overview

1. **q1_evaluate_retrieval_results.py**: Evaluates using top-1 retrieved context
2. **q2_evaluate_oracle.py**: Evaluates using Oracle context from dev_1200.json
3. **q3_evaluate_retrieval_with_random.py**: Evaluates using top-1 context + random unrelated context

## Cover Exact Match Fix

### Problem

The original `CoverExactMatch` implementation only checked if the predicted answer was contained in the gold answer:

```python
# Original (incorrect) implementation
return (self.normalize_answer(answers1) in self.normalize_answer(answers2))
```

This caused false negatives when the gold answer was contained in the predicted answer.

**Example:**
- Predicted: `"The Mask Of Fu Manchu came out first."`
- Gold: `"The Mask Of Fu Manchu"`
- Original result: `False` (incorrect)
- Expected result: `True` (gold answer is contained in predicted answer)

### Solution

The fix implements **bidirectional containment checking**:

```python
# Fixed implementation
norm1 = self.normalize_answer(answers1)
norm2 = self.normalize_answer(answers2)
return (norm1 in norm2) or (norm2 in norm1)
```

Now it checks:
- If predicted answer contains gold answer, OR
- If gold answer contains predicted answer

### Impact

After the fix:
- **top_1_results.json**: Cover Exact Match improved from 3.25% (39/1200) to 42.92% (515/1200)
- **dev_context_evaluation_results.json**: Cover Exact Match improved from 9.00% (108/1200) to 60.92% (731/1200)

## Prerequisites

### 1. Install Dependencies

```bash
source .venv/bin/activate
pip install tqdm requests mistralai transformers torch openai huggingface_hub
```

### 2. Set Environment Variables

**For Mistral API:**
```bash
export MISTRAL_API_KEY='your_mistral_api_key'
export MISTRAL_RPM_LIMIT=60  # Optional, default is 60 requests/minute
```

### 3. Required Data Files

- `data/retrieval_results.json` - Retrieval results with question_id -> context_id scores
- `data/dev_1200.json` - Development set with questions, answers, and contexts
- `data/wiki_musique_corpus.json` - Corpus with context documents

## Script 1: q1_evaluate_retrieval_results.py

Evaluates model performance using **top-1 retrieved context** from retrieval results.

### Usage

```bash
source .venv/bin/activate
export MISTRAL_API_KEY='your_mistral_api_key'
export MISTRAL_RPM_LIMIT=60

python3 q1_evaluate_retrieval_results.py
```

### With Custom Arguments

```bash
python3 q1_evaluate_retrieval_results.py \
    --retrieval_path data/retrieval_results.json \
    --dev_path data/dev_1200.json \
    --corpus_path data/wiki_musique_corpus.json \
    --output_path q1_results.json \
    --model_name open-mistral-7b \
    --top_k 1
```

### Arguments

- `--retrieval_path`: Path to retrieval_results.json (default: `data/retrieval_results.json`)
- `--dev_path`: Path to dev_1200.json (default: `data/dev_1200.json`)
- `--corpus_path`: Path to wiki_musique_corpus.json (default: `data/wiki_musique_corpus.json`)
- `--output_path`: Path to save results (default: `retrieval_evaluation_results.json`)
- `--model_name`: Model name (default: `open-mistral-7b`)
- `--top_k`: Number of top contexts to use (default: `1`)

### Output

Saves results to JSON file with:
- Overall metrics (Exact Match, Cover Exact Match)
- Detailed results for each question

## Script 2: q2_evaluate_oracle.py

Evaluates model performance using **Oracle context** directly from dev_1200.json (gold standard context).

### Usage

```bash
source .venv/bin/activate
export MISTRAL_API_KEY='your_mistral_api_key'
export MISTRAL_RPM_LIMIT=60

python3 q2_evaluate_oracle.py
```

### With Custom Arguments

```bash
python3 q2_evaluate_oracle.py \
    --dev_path data/dev_1200.json \
    --output_path q2_oracle_results.json \
    --model_name open-mistral-7b
```

### Arguments

- `--dev_path`: Path to dev_1200.json (default: `data/dev_1200.json`)
- `--output_path`: Path to save results (default: `dev_context_evaluation_results.json`)
- `--model_name`: Model name (default: `open-mistral-7b`)

### Output

Saves results to JSON file with:
- Overall metrics (Exact Match, Cover Exact Match)
- Detailed results for each question
- Context formatted from dev_1200.json format

## Script 3: q3_evaluate_retrieval_with_random.py

Evaluates model performance using **top-1 retrieved context + random unrelated context** to test robustness to noise.

### Usage

```bash
source .venv/bin/activate
export MISTRAL_API_KEY='your_mistral_api_key'
export MISTRAL_RPM_LIMIT=60

python3 q3_evaluate_retrieval_with_random.py
```

### With Custom Arguments

```bash
python3 q3_evaluate_retrieval_with_random.py \
    --retrieval_path data/retrieval_results.json \
    --dev_path data/dev_1200.json \
    --corpus_path data/wiki_musique_corpus.json \
    --output_path q3_random_results.json \
    --model_name open-mistral-7b \
    --random_seed 42
```

### Arguments

- `--retrieval_path`: Path to retrieval_results.json (default: `data/retrieval_results.json`)
- `--dev_path`: Path to dev_1200.json (default: `data/dev_1200.json`)
- `--corpus_path`: Path to wiki_musique_corpus.json (default: `data/wiki_musique_corpus.json`)
- `--output_path`: Path to save results (default: `retrieval_with_random_evaluation_results.json`)
- `--model_name`: Model name (default: `open-mistral-7b`)
- `--random_seed`: Random seed for reproducibility (optional, default: `None`)

### Output

Saves results to JSON file with:
- Overall metrics (Exact Match, Cover Exact Match)
- Detailed results including both top-1 and random context information

## Comparison of Experiments

| Experiment | Context Source | Expected Performance | Use Case |
|-----------|---------------|---------------------|----------|
| **q1: Retrieval** | Top-1 retrieved context | Baseline | Evaluate retrieval + generation |
| **q2: Oracle** | Oracle context from dev_1200.json | Highest (perfect context) | Evaluate generation only |
| **q3: Retrieval + Random** | Top-1 + Random unrelated context | Lower (due to noise) | Test robustness to noise |

## Evaluation Metrics

### Exact Match (EM)

Checks if the predicted answer exactly matches the gold answer after normalization:
- Lowercase conversion
- Punctuation removal
- Article removal (a, an, the)
- Whitespace normalization

**Returns:** `True` if exactly equal, `False` otherwise

### Cover Exact Match (CEM) - Fixed Version

Checks if either answer contains the other after normalization:
- Checks if `predicted_answer` contains `gold_answer`, OR
- Checks if `gold_answer` contains `predicted_answer`

**Returns:** `True` if either containment is true, `False` otherwise

**Examples:**
- Predicted: `"The Mask Of Fu Manchu came out first."`
- Gold: `"The Mask Of Fu Manchu"`
- Result: `True` ✓ (gold is contained in predicted)

- Predicted: `"Maheen Khan"`
- Gold: `"Maheen Khan is a designer"`
- Result: `True` ✓ (predicted is contained in gold)

## Re-evaluating Existing Results

If you have existing result files that were generated with the old CoverExactMatch rule, you can re-evaluate them using:

```bash
python3 reevaluate_results.py --input <result_file.json> --output <updated_result_file.json>
```

This will:
1. Load the existing results
2. Re-calculate exact_match and cover_exact_match using the fixed metrics
3. Update overall statistics
4. Save to a new file

## Notes

- All scripts use the **fixed CoverExactMatch** metric (bidirectional containment)
- Progress is printed every 10 questions
- Failed questions are skipped and counted
- Scripts respect RPM limits to avoid API rate limiting
- Results are saved in JSON format for further analysis
- The fixed CoverExactMatch significantly improves accuracy for cases where answers contain additional information

## Troubleshooting

### API Key Not Set

```
ValueError: MISTRAL_API_KEY environment variable is not set
```

**Solution:** Set the environment variable:
```bash
export MISTRAL_API_KEY='your_api_key'
```

### Module Not Found

```
ModuleNotFoundError: No module named 'tqdm'
```

**Solution:** Install dependencies:
```bash
pip install tqdm requests mistralai transformers torch openai huggingface_hub
```

### Rate Limiting

If you encounter rate limit errors, reduce the RPM limit:
```bash
export MISTRAL_RPM_LIMIT=30  # Lower rate limit
```

## File Locations

- **Scripts**: `q1_evaluate_retrieval_results.py`, `q2_evaluate_oracle.py`, `q3_evaluate_retrieval_with_random.py`
- **Fixed Metric**: `DEXTER-macos/dexter/utils/metrics/CoverExactMatch.py`
- **Re-evaluation Tool**: `reevaluate_results.py`

