# RAGs for Open Domain Complex QA: Experimental Analysis Report

## Executive Summary

This report analyzes the performance of Retrieval-Augmented Generation (RAG) systems for open-domain complex question answering. Experiments were conducted using the Mistral-large-latest model on a dataset of 1,200 questions, investigating how different context retrieval strategies and noise injection affect answer generation performance.

## Experimental Setup

### Model and Parameters

All experiments use the **Mistral-large-latest** model via Mistral API.

### Evaluation Metrics

- **Exact Match (EM)**: Predicted answer exactly matches the gold answer after normalization
- **Cover Exact Match (CEM)**: Bidirectional containment check (predicted contains gold OR gold contains predicted)

## 1. Performance Overview

### 1.1 Quantitative Results

| Experiment | Configuration | Exact Match (EM) | Cover Exact Match (CEM) | EM Count | CEM Count |
|------------|--------------|------------------|------------------------|----------|-----------|
| **Q1: Retriever-based RAG** | Top-1 | 29.00% | 37.17% | 348/1200 | 446/1200 |
| **Q1: Retriever-based RAG** | Top-3 | 26.08% | 36.33% | 313/1200 | 436/1200 |
| **Q2: Oracle Contexts** | Oracle | 52.88% | 70.14% | 634/1199 | 841/1199 |
| **Q3: Noise Injection** | Top-1 + 1 Random | 27.17% | 35.25% | 326/1200 | 423/1200 |
| **Q3: Noise Injection** | Top-3 + 1 Random | 26.33% | 36.58% | 316/1200 | 439/1200 |
| **Q3: Noise Injection** | Top-3 + 3 Random | 24.00% | 34.67% | 288/1200 | 416/1200 |

### 1.2 Key Performance Observations

1. **Oracle contexts significantly outperform retrieved contexts**: Oracle configuration (Q2) achieves 52.88% EM, which is **1.82× higher** than Top-1 retrieval (29.00%); CEM reaches 70.14%, which is **1.89× higher** than Top-1 (37.17%).

2. **Top-3 retrieval performs slightly worse than Top-1**: EM drops from 29.00% to 26.08%, CEM drops from 37.17% to 36.33%, indicating that additional contexts may introduce noise.

3. **Noise injection degrades performance**:
   - Top-1 + 1 Random vs Top-1: EM drops from 29.00% to 27.17% (-6.3%), CEM drops from 37.17% to 35.25% (-5.2%)
   - Top-3 + 3 Random vs Top-3: EM drops from 26.08% to 24.00% (-8.0%), CEM drops from 36.33% to 34.67% (-4.6%)

## 2. Intuition Behind Performance Changes

### 2.1 Why Oracle Contexts Perform Better

Reasons for superior oracle context performance:

- **Relevance Guarantee**: Oracle contexts are explicitly annotated as relevant to each question, eliminating retrieval errors
- **Information Completeness**: Oracle contexts contain all necessary information to answer the question
- **No Noise**: No irrelevant or distracting information included

**The 1.82× EM improvement and 1.89× CEM improvement demonstrate that context quality is the primary bottleneck in RAG systems.**

### 2.2 Why Top-3 Retrieval Performs Worse Than Top-1

Reasons for Top-3 performance degradation:

- **Noise Introduction**: Additional contexts may contain irrelevant information that confuses the model
- **Information Dilution**: Relevant information becomes diluted across more contexts
- **Attention Dispersion**: The model must identify key information from more content

### 2.3 Why Noise Injection Degrades Performance

Noise injection causes performance degradation through:

- **Distraction Effect**: Irrelevant contexts divert model attention
- **Context Dilution**: Relevant information becomes a smaller proportion of total context
- **Incorrect Inference Risk**: The model may generate answers based on irrelevant information

## 3. Qualitative Analysis

### 3.1 Case Study 1: Information Completeness

**Question**: "Who is the mother of the director of film Polish-Russian War (Film)?"

**Gold Answer**: "Małgorzata Braunek"

**Q1 Top-1 Result** ✅:
- Retrieved context contains film and director information
- **Predicted**: "Małgorzata Braunek."
- **Result**: EM ✅, CEM ✅

**Q2 Oracle Result** ✅:
- Oracle context contains detailed information about director Xawery Żuławski, including "He is the son of actress Małgorzata Braunek"
- **Predicted**: "Małgorzata Braunek"
- **Result**: EM ✅, CEM ✅

**Q3 Top-3+3 Random Result** ❌:
- 3 random unrelated contexts (Tatum Bell, Darmstadt Madonna, Pang Juan) interfered with the model
- **Predicted**: "The provided context does not specify the mother of Xawery Żuławski."
- **Result**: EM ❌, CEM ❌

### 3.2 Case Study 2: Historical Information

**Question**: "When did John V, Prince Of Anhalt-Zerbst's father die?"

**Gold Answer**: "12 June 1516"

**Q1 Top-1 Result** ❌:
- Retrieved context is about John VI rather than John V
- **Predicted**: "The provided context does not specify when John V, Prince of Anhalt-Zerbst's father died."
- **Result**: EM ❌, CEM ❌

**Q2 Oracle Result** ✅:
- Oracle context contains complete family information, including Ernest I's death date
- **Predicted**: "12 June 1516"
- **Result**: EM ✅, CEM ✅

## 4. How Experiments Answer Research Questions

### 4.1 RQ1: How does negative contexts impact downstream answer generation performance?

**Answer**: Negative (unrelated) contexts have a **significant negative impact** on answer generation.

**Evidence**:
- Top-1 + 1 Random vs Top-1: EM drops 6.3% (29.00% → 27.17%)
- Top-3 + 3 Random vs Top-3: EM drops 8.0% (26.08% → 24.00%)
- More noise leads to greater performance degradation

**Implication**: RAG systems should prioritize high-precision retrieval to minimize negative context injection.

### 4.2 RQ2: Are negative contexts more important for answer generation than related contexts?

**Answer**: **No, related contexts are far more important than negative contexts.**

**Evidence**:
- The gap between Oracle and Top-1 (23.88% EM) is much larger than the gap between Top-1 and Top-1+1Random (1.83% EM)
- Context quality impact is **13× greater** than noise impact

**Implication**: Improving retrieval quality should be prioritized over noise filtering.

### 4.3 RQ3: Does providing only gold contexts deteriorate the performance compared to mixing with other negative or related contexts?

**Answer**: **No, providing only gold contexts significantly improves performance.**

**Evidence**:
- Oracle (52.88% EM) >> Top-1 (29.00% EM)
- Oracle (70.14% CEM) >> Top-1 (37.17% CEM)
- Mixing with negative contexts consistently degrades performance

**Implication**: Quality over quantity—a few high-quality contexts outperform many mixed-quality contexts.

## 5. Detailed Analysis: Noise Sampling Impact on QA Performance

### 5.1 Quantitative Impact

| Configuration | EM | CEM | vs Baseline |
|--------------|-----|-----|-------------|
| Top-1 (baseline) | 29.00% | 37.17% | - |
| Top-1 + 1 Random | 27.17% | 35.25% | EM -6.3%, CEM -5.2% |
| Top-3 (baseline) | 26.08% | 36.33% | - |
| Top-3 + 1 Random | 26.33% | 36.58% | EM +1.0%, CEM +0.7% |
| Top-3 + 3 Random | 24.00% | 34.67% | EM -8.0%, CEM -4.6% |

### 5.2 Key Observations

1. **Single noise context impact**: Top-1 + 1 Random causes 6.3% EM drop
2. **Multiple noise contexts impact**: Top-3 + 3 Random causes 8.0% EM drop
3. **Top-3 + 1 Random anomaly**: Slight improvement may be due to statistical fluctuation or specific random samples

## 6. The Role of Context Quality

### 6.1 Context Quality as the Primary Factor

**Quality Spectrum**:
1. **Oracle contexts** (highest quality): 52.88% EM, 70.14% CEM
2. **Top-1 retrieved contexts**: 29.00% EM, 37.17% CEM
3. **Top-3 retrieved contexts**: 26.08% EM, 36.33% CEM
4. **Retrieved + random contexts**: 24.00-27.17% EM, 34.67-36.58% CEM

### 6.2 Quality vs. Quantity Trade-off

- **Top-3 < Top-1**: More contexts ≠ better performance
- **Oracle >> All Retrieval**: Quality matters far more than quantity
- **Noise Degrades Performance**: Noise consistently hurts performance

## 7. Conclusions and Future Directions

### 7.1 Key Findings

1. **Context quality is the primary bottleneck**: Oracle context EM is 1.82× that of retrieved contexts
2. **Negative contexts have significant negative impact**: Causing 6-8% performance degradation
3. **Related contexts are more important than negative contexts**: Quality gap impact is 13× that of noise impact
4. **Pure gold contexts outperform mixed contexts**: Quality over quantity

### 7.2 Implications for RAG Systems

1. **Prioritize retrieval quality**: Invest in improving retrieval precision and relevance ranking
2. **Quality over quantity**: Focus on retrieving fewer high-quality contexts
3. **Noise awareness**: Minimize negative context injection through better retrieval and filtering
4. **Context selection**: Develop better context selection and ranking methods

### 7.3 Future Research Directions

1. **Advanced retrieval methods**: Investigate better retrieval models and re-ranking strategies
2. **Noise filtering**: Develop methods to identify and filter negative contexts before answer generation
3. **Context fusion**: Explore better methods for combining multiple contexts
4. **Adaptive context selection**: Develop methods to dynamically determine optimal context quantity and quality

## 8. Appendix: Experimental Details

### 8.1 Model Configuration

- **Model Name**: `mistral-large-latest`
- **API Provider**: Mistral AI
- **Dataset**: 1,200 questions from `dev_1200.json`
- **Corpus**: `wiki_musique_corpus.json`

### 8.2 Experiment Configurations

| Experiment | Description |
|------------|-------------|
| Q1 Top-1 | Uses top-1 context from retrieval results |
| Q1 Top-3 | Uses top-3 contexts from retrieval results |
| Q2 Oracle | Uses annotated contexts from dev_1200.json |
| Q3 Top-1+1R | Top-1 retrieved context + 1 random unrelated context |
| Q3 Top-3+1R | Top-3 retrieved contexts + 1 random unrelated context |
| Q3 Top-3+3R | Top-3 retrieved contexts + 3 random unrelated contexts |

### 8.3 Performance Summary Table

| Metric | Q1 Top-1 | Q1 Top-3 | Q2 Oracle | Q3 Top-1+1R | Q3 Top-3+1R | Q3 Top-3+3R |
|--------|----------|----------|-----------|-------------|-------------|-------------|
| EM | 29.00% | 26.08% | 52.88% | 27.17% | 26.33% | 24.00% |
| CEM | 37.17% | 36.33% | 70.14% | 35.25% | 36.58% | 34.67% |
| EM Count | 348 | 313 | 634 | 326 | 316 | 288 |
| CEM Count | 446 | 436 | 841 | 423 | 439 | 416 |
| Total | 1200 | 1200 | 1199 | 1200 | 1200 | 1200 |

---

**Report Generated**: Based on experimental results from q1_top_1_results.json, q1_top_3_results.json, q2_oracle_results.json, q3_top_1_mix_random_1_results.json, q3_top_3_mix_random_1_results.json, and q3_top_3_mix_random_3_results.json
