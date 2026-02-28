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
from nanogpt_edit.steering import compute_steering_vector, steer_context
sv = compute_steering_vector(
    model, tokenizer,
    positive=["happy joyful wonderful"],
    negative=["sad terrible awful"],
    layer=8,
)
with steer_context(model, sv, layer=8, alpha=5.0):
    text = editor.generate("Today I feel", max_new_tokens=30)
```

### Run Integration Tests

```bash
python -m pytest nanogpt_edit/test_integration.py -v
```
