# nanoGPT Quickstart Guide

## Stream A: Reasoning Model

### 1. Prepare Data

```bash
python data/prepare_reasoning.py
```

This creates `train_prompts.jsonl` and `val.jsonl` for reasoning training.

### 2. Supervised Fine-Tuning (SFT)

```bash
# Single GPU
python train_sft.py

# With custom settings
python train_sft.py --init_from=gpt2 --learning_rate=2e-5 --batch_size=4

# Multi-GPU (4 GPUs)
torchrun --standalone --nproc_per_node=4 train_sft.py
```

Output checkpoint saved to `out-sft/`.

### 3. GRPO Reinforcement Learning

```bash
# Single GPU (starts from SFT checkpoint)
python train_grpo.py

# With custom settings
python train_grpo.py --init_from=out-sft/ckpt.pt --learning_rate=3e-6

# Multi-GPU (4 GPUs)
torchrun --standalone --nproc_per_node=4 train_grpo.py
```

Output checkpoint saved to `out-grpo/`.

### 4. Generate Reasoning Samples

```bash
python sample_reasoning.py --checkpoint=out-grpo/ckpt.pt --prompt="What is 23 * 47?"
```

Displays two-phase output: thinking (dimmed) then answer (highlighted).

### 5. Evaluate

```bash
# Evaluate on GSM8K
python eval_reasoning.py --checkpoint=out-grpo/ckpt.pt --benchmark=gsm8k

# Compare two checkpoints
python eval_reasoning.py --checkpoint=out-sft/ckpt.pt --checkpoint2=out-grpo/ckpt.pt
```

Reports pass@1, pass@k, majority@k, format compliance, and mean CoT length with bootstrap confidence intervals.

### 6. Run Foundation Tests

```bash
python test_foundation.py
```

---

## Stream B: Surgical Editing Toolkit (nanogpt_edit)

### CLI Usage

All commands are run from the nanoGPT root directory.

#### Model Info

```bash
python -m nanogpt_edit info
```

#### Causal Tracing

```bash
python -m nanogpt_edit trace --subject "The Eiffel Tower" --prompt "The Eiffel Tower is in" --target " Paris"
```

#### ROME Edit (Single Fact)

```bash
python -m nanogpt_edit rome-edit \
  --prompt "The Eiffel Tower is in" \
  --subject "The Eiffel Tower" \
  --target " Rome"
```

#### MEMIT Edit (Batch from JSON)

```bash
python -m nanogpt_edit memit-edit --cases nanogpt_edit/test_cases.json --layers 3 4 5 6 7 8
```

#### Task Vectors

```bash
# Compute task vector between two checkpoints
python -m nanogpt_edit task-vector --base ckpt_base.pt --edited ckpt_edited.pt --op negate --output negated.pt

# TIES-Merging
python -m nanogpt_edit task-vector --base ckpt_base.pt --edited ckpt_edited.pt --op ties --output merged.pt
```

#### Steering Vectors

```bash
python -m nanogpt_edit steering \
  --positive "The weather is wonderful and bright" \
  --negative "The weather is terrible and dark" \
  --layer 8 \
  --alpha 5.0 \
  --prompt "Today I feel"
```

#### Evaluation

```bash
# Evaluate edits against test cases
python -m nanogpt_edit eval --cases nanogpt_edit/test_cases.json --method rome

# Evaluate with MEMIT
python -m nanogpt_edit eval --cases nanogpt_edit/test_cases.json --method memit --layers 3 4 5 6 7 8
```

### Python API Usage

```python
from model import GPT
from nanogpt_edit.edit_core import ModelEditor
from nanogpt_edit.data_structures import EditRequest
import tiktoken

# Load model
model = GPT.from_pretrained("gpt2")
model.eval()
tokenizer = tiktoken.get_encoding("gpt2")

# Create editor
editor = ModelEditor(model, tokenizer)

# Apply a ROME edit
request = EditRequest(
    prompt="The Eiffel Tower is in",
    subject="The Eiffel Tower",
    target=" Rome",
)
result = editor.rome_edit(request)
print(f"Efficacy: {result.efficacy_score:.4f}")

# Generate after edit
text = editor.generate("The Eiffel Tower is in", max_new_tokens=20)
print(text)

# Causal tracing
trace = editor.causal_trace(
    prompt="The Eiffel Tower is in",
    subject="The Eiffel Tower",
    target=" Paris",
)

# Steering vectors
from nanogpt_edit.steering import (
    compute_steering_vector, SteeringHook,
    save_steering_vector, load_steering_vector,
)
sv = compute_steering_vector(
    editor,
    positive_texts=["I am happy and joyful!", "This is wonderful!"],
    negative_texts=["I am sad and miserable.", "This is terrible."],
    layer=8,
)
with SteeringHook(model, sv, alpha=5.0):
    text = editor.generate("Today I feel", max_new_tokens=30)

# Save/load steering vectors for reuse
save_steering_vector(sv, "happy_sv.pt")
sv_loaded = load_steering_vector("happy_sv.pt")
```

### Steering Vectors (Behavioral Control at Inference Time)

Steering vectors modify model behavior without changing weights. They work by adding a learned direction to transformer activations during inference.

**How it works:**
1. Provide contrastive text pairs (positive = desired behavior, negative = opposite)
2. Extract activations from each set and compute the difference vector
3. Add this vector (scaled by `alpha`) to a transformer layer during generation

```bash
# CLI: compute and test a steering vector
python -m nanogpt_edit steering \
  --positive "The weather is wonderful and bright" "I love this!" \
  --negative "The weather is terrible and dark" "I hate this." \
  --layer 8 \
  --alpha 5.0 \
  --prompt "Today I feel"
```

```python
# Python API
from nanogpt_edit.steering import (
    compute_steering_vector, SteeringHook, multi_layer_steer,
    save_steering_vector, load_steering_vector,
)

# Compute from contrastive pairs
sv = compute_steering_vector(
    editor,
    positive_texts=["I am happy!", "This is great!", "Everything is wonderful!"],
    negative_texts=["I am sad.", "This is bad.", "Everything is terrible."],
    layer=6,              # which transformer block to extract from
    aggregation="mean_seq" # or "last_token"
)

# Apply during generation
with SteeringHook(model, sv, alpha=5.0):
    output = model.generate(input_ids, max_new_tokens=50)

# Multi-layer steering (apply vectors at different layers simultaneously)
sv_early = compute_steering_vector(editor, pos, neg, layer=4)
sv_late  = compute_steering_vector(editor, pos, neg, layer=10)
with multi_layer_steer(model, [sv_early, sv_late], alpha=3.0):
    output = model.generate(input_ids, max_new_tokens=50)

# Save and reuse
save_steering_vector(sv, "happy_direction.pt")
sv = load_steering_vector("happy_direction.pt")
```

**Key parameters:**
- `layer`: Which transformer block to hook into (0-11 for GPT-2 124M). Middle layers (4-8) often work best.
- `alpha`: Strength of the steering effect. Higher = stronger. Start with 1.0-5.0.
- `aggregation`: `"mean_seq"` averages over all token positions; `"last_token"` uses only the final position.

### Run Tests

```bash
# Foundation tests (reasoning model)
python test_foundation.py

# Editing tests (ROME/MEMIT)
python test_nanogpt_edit.py
python test_rome_qa.py

# Steering tests
python test_steering.py

# Integration tests (end-to-end pipelines)
python nanogpt_edit/test_integration.py
```
