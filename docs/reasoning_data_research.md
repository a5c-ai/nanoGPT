# Reasoning Training Data Research for 1.5B GPT-2 XL

## Current State
- Model: GPT-2 XL 1.5B with modern arch (RoPE, RMSNorm, SwiGLU)
- GPU: Quadro RTX 8000, 46GB VRAM
- Existing data: 18,681 train / 9,017 val examples across 4 domains (math 40%, logic 25%, science 25%, physical 10%)
- Format: `{prompt, thinking, answer, domain}`
- Target benchmarks: GSM8K, ARC, BoolQ, MATH

## 1. Available Open Reasoning Datasets on HuggingFace

### 1.1 Math Reasoning

| Dataset | Size | Source | License | Notes |
|---------|------|--------|---------|-------|
| **nvidia/OpenMathInstruct-2** | 14M pairs (~600K unique questions) | Llama3.1-405B generated solutions for GSM8K/MATH train problems | CC-BY-4.0 | Best open math reasoning dataset. Has 1M/2M/5M subsamples. |
| **nvidia/OpenMathReasoning** | 306K unique problems, ~3.2M CoT solutions | DeepSeek-R1 + QwQ-32B generations on AoPS problems | CC-BY-4.0 | Higher difficulty (olympiad level). Long CoT traces. |
| **AI-MO/NuminaMath-1.5** | ~900K competition problems | Chinese HS math + olympiad, from PDFs/forums | Apache-2.0 | Competition-level, may be too hard for 1.5B cold-start |
| **meta-math/MetaMathQA** | ~395K | GPT-3.5 augmented GSM8K/MATH | MIT | Good for bootstrapping, moderate difficulty |
| **TIGER-Lab/MathInstruct** | ~262K | Compilation from 13 math datasets with CoT | MIT | Multi-source, diverse difficulty |
| **camel-ai/math** | ~50K | GPT-4 generated | CC-BY-NC-4.0 | Smaller but high quality |
| **microsoft/orca-math-word-problems-200k** | 200K | Synthetic word problems | MIT | GSM8K-style, good for baseline math |

### 1.2 Science Reasoning

| Dataset | Size | Source | License | Notes |
|---------|------|--------|---------|-------|
| **allenai/ai2_arc** (ARC-Challenge) | 2,590 train | Standardized science exams | CC-BY-SA-4.0 | Direct benchmark target |
| **allenai/sciq** | 13,679 train | Science exam questions | CC-BY-NC-3.0 | With supporting paragraphs |
| **allenai/openbookqa** | 4,957 | Elementary science | Apache-2.0 | With science facts |

### 1.3 General Reasoning / CoT

| Dataset | Size | Source | License | Notes |
|---------|------|--------|---------|-------|
| **Open-Orca/OpenOrca** | ~4.2M total (~75K CoT subset) | GPT-3.5/4 augmented FLAN | MIT | Large but CoT subset is small |
| **google/FLAN Collection (CoT submix)** | ~75K CoT entries | Human-curated | Apache-2.0 | Gold standard general CoT |
| **boolq** (Google) | 15,942 | Yes/no reading comprehension | CC-BY-SA-3.0 | Direct benchmark target |

### 1.4 Recommended Primary Datasets

For a 1.5B model targeting GSM8K, ARC, BoolQ, and MATH:

1. **OpenMathInstruct-2 (1M subsample)** - Primary math source. Use `train_1M` split.
2. **microsoft/orca-math-word-problems-200k** - GSM8K-style word problems for baseline math capability.
3. **MetaMathQA (subset ~100K)** - Augmented GSM8K/MATH with diverse rephrasings.
4. **ARC-Challenge + SciQ + OpenBookQA** - Science reasoning (~20K combined with CoT generation).
5. **BoolQ with generated CoT** - Use a teacher model to add reasoning traces to BoolQ train.
6. **FLAN-CoT subset (~75K)** - General reasoning diversity.

## 2. DeepSeek-R1 Distillation Approach

### 2.1 Key Finding: Distillation > Direct RL for Small Models

The DeepSeek-R1 paper (Jan 2025) demonstrated conclusively that **distilling reasoning traces from large models into small models outperforms training small models with RL directly**. This is the single most important finding for our 1.5B model.

### 2.2 DeepSeek-R1 Distillation Recipe

1. **Generate traces**: Use DeepSeek-R1 (or R1-Distill-Qwen-32B) to generate chain-of-thought solutions for training problems.
2. **Filter by correctness**: Only keep traces where the final answer is correct (verified against ground truth).
3. **Fine-tune**: SFT the 1.5B model on ~800K filtered traces.

### 2.3 Practical Implementation for Our Setup

Since we cannot run DeepSeek-R1 locally (requires 8x H200), recommended approach:
- Use **DeepSeek-R1 API** or **Together AI API** to generate traces at ~$0.50-2/M tokens
- Alternatively, use pre-generated datasets: **OpenMathInstruct-2** (Llama-405B traces) or **OpenMathReasoning** (R1/QwQ traces)
- The pre-generated datasets are effectively "distillation-ready"

### 2.4 Re-Distillation (Dropbox Research, 2025)

Dropbox researchers showed that **logit alignment as a second-stage distillation** improved GSM8K by 4% for Qwen models and 14% for Llama models. This suggests a two-phase approach:
1. Phase 1: SFT on filtered reasoning traces
2. Phase 2: Logit-level KD from a 7B/14B teacher on the same data

### 2.5 DeepScaleR Precedent

DeepScaleR-1.5B achieved **43.1% on AIME 2024** (surpassing o1-preview) starting from DeepSeek-R1-Distill-Qwen-1.5B:
- Training data: Only ~40K unique problem-answer pairs
- Method: GRPO with iterative context lengthening (8K -> 16K -> 24K)
- Key insight: Small, high-quality data + RL scaling is extremely effective at 1.5B scale

## 3. STaR (Self-Taught Reasoner) Strategy

### 3.1 Core Algorithm

```
for iteration in range(N):
    1. Generate rationales for all training questions using current model
    2. Filter: keep only rationales leading to correct answers
    3. Rationalize failures: for wrong answers, provide correct answer and re-generate rationale
    4. Fine-tune model on collected correct rationales
    5. Repeat with improved model
```

### 3.2 Applicability to 1.5B Model

**Challenges:**
- **Cold start problem**: STaR requires the base model to have minimum reasoning ability. A raw GPT-2 XL cannot bootstrap from zero.
- **Solution**: First do SFT distillation (Section 2), then use STaR for iterative improvement.

**Recommended STaR Schedule:**
1. SFT on distilled data first (get to ~30-40% GSM8K baseline)
2. STaR iteration 1: Generate rationales, filter correct ones, retrain
3. STaR iteration 2-3: Repeat with increasingly difficult problems
4. Expected improvement: 5-15% accuracy gain over 3 iterations

### 3.3 2025 STaR Variants to Consider

- **HS-STaR**: Hierarchical sampling with difficulty estimation and budget reallocation. Allocates more generation budget to problems near the model's capability frontier.
- **CARE-STaR**: Constraint-aware reasoning, better for structured problems.
- **STaR + Verifier**: Train an ORM (outcome reward model) alongside to improve filtering.

## 4. Data Quality for 1.5B Model

### 4.1 Optimal CoT Length

**Key research finding (CMU L1-1.5B, March 2025):**
- Longer CoT is NOT always better for small models
- L1-1.5B with Length-Controlled Policy Optimization (LCPO) showed that **short, focused CoT** can outperform longer reasoning at 1.5B scale
- **Recommended**: Target 150-400 tokens for CoT traces
  - Simple problems (GSM8K easy): 100-200 tokens
  - Medium problems (GSM8K hard, ARC): 200-400 tokens
  - Hard problems (MATH): 300-600 tokens
- **Truncation strategy**: Filter out traces > 800 tokens for initial SFT; the model can learn to reason longer via RL later

### 4.2 Difficulty Curriculum

**E2H Reasoner (2025)** demonstrated that curriculum learning (easy-to-hard) significantly improves reasoning in 1.5B-3B models:

**Recommended 3-phase curriculum:**

| Phase | Duration | Data Mix | Difficulty |
|-------|----------|----------|------------|
| Phase 1: Foundation | 40% of training | GSM8K-style word problems, simple BoolQ, basic ARC | Easy (>70% solvable by base model) |
| Phase 2: Growth | 40% of training | MetaMathQA, harder ARC-Challenge, multi-step science | Medium (30-70% solvable) |
| Phase 3: Challenge | 20% of training | MATH competition, NuminaMath easy subset, complex reasoning | Hard (<30% solvable) |

**Counterpoint**: Some 2025 papers (e.g., "On the Limits of Curriculum Learning") found that random sampling is competitive. **Recommendation**: Implement curriculum but A/B test against random baseline.

### 4.3 Domain Balancing

Current data is heavily skewed toward math (40%) and logic (25%). For target benchmarks:

**Recommended domain proportions for training:**

| Domain | Proportion | Target Benchmark | Rationale |
|--------|-----------|-----------------|-----------|
| Math (arithmetic/word problems) | 35% | GSM8K | Core reasoning capability |
| Math (competition/algebra) | 15% | MATH | Harder math subset |
| Science | 20% | ARC | Science reasoning + factual |
| Boolean/Reading Comprehension | 15% | BoolQ | Binary classification reasoning |
| General CoT / Logic | 15% | All | Transfer learning, diversity |

**Key principle**: Over-representing a domain by >40% leads to catastrophic forgetting on others. Under-representing by <10% yields negligible improvement.

### 4.4 Decontamination

**Critical for credible results.** 2025 research shows decontamination is essential:

1. **N-gram overlap is insufficient**: MegaScience (2025) found that n-gram methods miss 60-80% of contamination that LLM-based methods catch.
2. **Small models overfit benchmarks easily**: A 13B model can achieve GPT-4-level scores on GSM8K just from contaminated training data.

**Recommended decontamination pipeline:**
```
Step 1: Exact match removal
  - Remove any training example with >80% token overlap with test set questions

Step 2: N-gram overlap (baseline)
  - 13-gram overlap check between training questions and test questions
  - Remove matches with >0.5 Jaccard similarity

Step 3: Semantic decontamination
  - Embed all training and test questions with a sentence transformer
  - Remove training examples with cosine similarity > 0.85 to any test question

Step 4: LLM-based verification (for final data)
  - Use a small LLM to check if training examples are paraphrases of test examples
  - Most thorough but most expensive

Target benchmarks to decontaminate against:
  - GSM8K test set (1,319 problems)
  - MATH test set (5,000 problems)
  - ARC-Challenge test set (1,172 questions)
  - BoolQ validation set (3,270 questions)
```

### 4.5 Data Quality Filtering

Beyond decontamination, apply quality filters:

1. **Correctness verification**: For math, verify final numerical answer matches ground truth. Discard wrong solutions.
2. **Format compliance**: Ensure CoT follows `<think>...</think>` format with extractable `<answer>` tags.
3. **Length filtering**: Remove traces < 50 tokens (too short to reason) or > 1000 tokens (too long for 1.5B).
4. **Deduplication**: Remove near-duplicate solutions (same problem, almost identical reasoning path).
5. **Response length selection**: MegaScience (2025) found this is the single most effective quality signal for science reasoning data.

## 5. Recommended Strategy: Phased Approach

### Phase 1: Data Collection & Curation (Recommended First)

**Target: ~200K high-quality examples**

| Source | Examples | Processing |
|--------|----------|------------|
| OpenMathInstruct-2 (1M subsample) | 80K (filtered) | Filter by correctness, length 150-400 tokens, deduplicate |
| orca-math-word-problems-200k | 40K (filtered) | Add CoT via teacher model or select with existing CoT |
| MetaMathQA | 30K (filtered) | Select GSM8K/MATH augmentations, length filter |
| ARC + SciQ + OpenBookQA | 15K | Generate CoT traces with DeepSeek-R1 API or use existing |
| BoolQ train | 10K | Generate CoT traces with teacher model |
| FLAN-CoT | 20K | Select reasoning-heavy examples |
| Existing multi_cot data | 5K (filtered from 18K) | Keep highest quality, decontaminate |
| **Total** | **~200K** | |

### Phase 2: SFT Training

1. Decontaminate against all target benchmark test sets
2. Format all data to match existing `{prompt, thinking, answer, domain}` schema
3. Train with difficulty curriculum (easy -> medium -> hard over 3 epochs)
4. Use context length 1024-2048 tokens (fits 46GB GPU comfortably)

### Phase 3: RL Fine-Tuning (After SFT)

Following the DeepScaleR recipe:
1. Select ~40K hard math problems where model gets <50% accuracy
2. Apply GRPO with iterative context lengthening (1K -> 2K -> 4K)
3. Use accuracy reward (binary: correct/incorrect answer)
4. Expected boost: 10-20% on GSM8K, 5-15% on MATH

### Phase 4: STaR Iteration (Optional, After RL)

1. Generate rationales for remaining unsolved problems
2. Rationalize failures (provide answer, re-generate reasoning)
3. Retrain on expanded correct-answer dataset
4. 2-3 iterations expected to yield 3-8% additional improvement

## 6. Key References

1. **OpenMathInstruct-2** (NVIDIA, 2024) - [Paper](https://arxiv.org/pdf/2410.01560) | [Dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
2. **OpenMathReasoning** (NVIDIA, 2025) - [Paper](https://arxiv.org/pdf/2504.16891) | [Dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
3. **DeepSeek-R1** (DeepSeek, 2025) - [Paper](https://arxiv.org/html/2501.12948v1) | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
4. **DeepScaleR-1.5B** (Berkeley/Together AI, 2025) - [Model](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
5. **STaR: Bootstrapping Reasoning** (Zelikman et al., 2022) - [Paper](https://arxiv.org/abs/2203.14465)
6. **L1-1.5B / LCPO** (CMU, 2025) - Length-controlled reasoning optimization
7. **E2H Reasoner** (2025) - [Paper](https://arxiv.org/abs/2506.06632) - Curriculum RL for 1.5B-3B models
8. **Teach Small Models by Curriculum Distillation** (EMNLP 2025) - [Paper](https://aclanthology.org/2025.emnlp-main.376.pdf)
9. **NuminaMath-1.5** (AI-MO, 2024) - [Dataset](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)
10. **Re-Distilling DeepSeek R1** (Dropbox, 2025) - [Blog](https://dropbox.github.io/r1_redistill_blogpost/)
11. **MegaScience** (2025) - LLM-based decontamination and response length selection
12. **MetaMathQA** - [Dataset](https://huggingface.co/datasets/meta-math/MetaMathQA)

## 7. Risk Factors and Mitigations

| Risk | Mitigation |
|------|-----------|
| CoT too long for 1.5B context window | Length filtering (max 600 tokens), LCPO-style training |
| Benchmark contamination inflating results | 4-stage decontamination pipeline |
| Catastrophic forgetting across domains | Balanced domain mixing, replay of prior data |
| Cold start for STaR | Do SFT distillation first, only STaR after baseline is established |
| Training data too hard | Curriculum learning, start with GSM8K-difficulty problems |
| Licensing issues | Stick to CC-BY, Apache, MIT datasets; avoid GPT-4 generated data for commercial use |
