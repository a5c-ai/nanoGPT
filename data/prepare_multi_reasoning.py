"""
Prepare multi-domain reasoning training dataset for nanoGPT.

The original reasoning model was trained ONLY on GSM8K math problems,
causing domain collapse. This script creates a diverse training set from:

  - GSM8K (math) - reused from data/gsm8k_cot/train.jsonl
  - OpenBookQA (science)
  - ARC-Easy + ARC-Challenge (science)
  - BoolQ (yes/no reasoning)
  - PIQA (physical intuition)
  - CommonsenseQA (common sense)

Each example is formatted as {prompt, thinking, answer} JSONL.

Domain balance targets:
  - math:               40%
  - science:            25% (OpenBookQA + ARC)
  - logic/common sense: 25% (BoolQ + CommonsenseQA)
  - physical intuition: 10% (PIQA)

Outputs:
  - data/multi_cot/train.jsonl          Combined, shuffled training set
  - data/multi_cot/val.jsonl            Combined validation set
  - data/multi_cot/train_prompts.jsonl  Prompts only (for GRPO)
  - data/multi_cot/domain_stats.json    Counts per domain
  - data/multi_cot/eval/<domain>.jsonl  50-problem eval sets per domain
"""

import io
import json
import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
GSM8K_TRAIN = SCRIPT_DIR / "gsm8k_cot" / "train.jsonl"
GSM8K_VAL = SCRIPT_DIR / "gsm8k_cot" / "val.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "multi_cot"
EVAL_DIR = OUTPUT_DIR / "eval"

# Reproducibility
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list:
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list, path: Path) -> None:
    """Write a list of dicts as a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {path}")


def format_choices(choices_labels: list, choices_texts: list) -> str:
    """Format multiple-choice options as A) text, B) text, ..."""
    parts = []
    for label, text in zip(choices_labels, choices_texts):
        parts.append(f"{label}) {text}")
    return "\n".join(parts)


def label_to_text(answer_key: str, choices_labels: list, choices_texts: list) -> str:
    """Map answer label (e.g., 'A') to its text."""
    for label, text in zip(choices_labels, choices_texts):
        if str(label) == str(answer_key):
            return text
    return answer_key


# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def load_gsm8k_existing() -> tuple:
    """Load existing GSM8K data from gsm8k_cot directory."""
    print("  Loading GSM8K from existing files...")
    train = read_jsonl(GSM8K_TRAIN)
    val = read_jsonl(GSM8K_VAL)

    # Tag domain
    for ex in train:
        ex["domain"] = "math"
    for ex in val:
        ex["domain"] = "math"

    print(f"    Train: {len(train)}, Val: {len(val)}")
    return train, val


def process_openbookqa() -> tuple:
    """Load and format OpenBookQA dataset."""
    print("  Loading OpenBookQA...")
    ds = load_dataset("openbookqa", "main")

    def convert(item):
        question = item["question_stem"].strip()
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]
        answer_text = label_to_text(answer_key, labels, texts)

        prompt = f"{question}\n{format_choices(labels, texts)}"
        thinking = f"This is a science question. The correct answer is {answer_key}) {answer_text}, because it best fits the scientific reasoning required by the question."
        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer_text,
            "domain": "science",
        }

    train = [convert(item) for item in ds["train"]]
    val = [convert(item) for item in ds["validation"]]
    print(f"    Train: {len(train)}, Val: {len(val)}")
    return train, val


def process_arc(config_name: str) -> tuple:
    """Load and format ARC dataset (Easy or Challenge)."""
    print(f"  Loading ARC ({config_name})...")
    ds = load_dataset("allenai/ai2_arc", config_name)

    def convert(item):
        question = item["question"].strip()
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]
        answer_text = label_to_text(answer_key, labels, texts)

        prompt = f"{question}\n{format_choices(labels, texts)}"
        difficulty = "challenging" if "Challenge" in config_name else "straightforward"
        thinking = f"Analyzing this {difficulty} science question: the answer is {answer_key}) {answer_text}. This follows from scientific principles relevant to the question."
        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer_text,
            "domain": "science",
        }

    train = [convert(item) for item in ds["train"]]
    val = [convert(item) for item in ds["validation"]]
    print(f"    Train: {len(train)}, Val: {len(val)}")
    return train, val


def process_boolq() -> tuple:
    """Load and format BoolQ dataset."""
    print("  Loading BoolQ...")
    ds = load_dataset("google/boolq")

    def convert(item):
        passage = item["passage"].strip()
        question = item["question"].strip()
        answer = item["answer"]  # True/False
        answer_str = "yes" if answer else "no"

        # Keep passage short for 124M model: first 300 chars
        short_passage = passage[:300]
        if len(passage) > 300:
            short_passage += "..."

        prompt = f"Passage: {short_passage}\n\nQuestion: {question}?\nAnswer yes or no."
        thinking = f"Based on the passage, the answer to whether {question} is {answer_str}. The passage provides information that supports this conclusion."
        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer_str,
            "domain": "logic",
        }

    train = [convert(item) for item in ds["train"]]
    val = [convert(item) for item in ds["validation"]]
    print(f"    Train: {len(train)}, Val: {len(val)}")
    return train, val


def process_piqa() -> tuple:
    """Load and format PIQA dataset.

    Downloads directly from the original source zip because the HuggingFace
    ybisk/piqa dataset uses a legacy loading script not supported by the
    current datasets library version.
    """
    print("  Loading PIQA (direct download from source)...")
    piqa_url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"

    response = urllib.request.urlopen(piqa_url)
    zip_data = io.BytesIO(response.read())

    train_items = []
    val_items = []
    train_labels = []
    val_labels = []

    with zipfile.ZipFile(zip_data) as zf:
        # Read train data
        with zf.open("physicaliqa-train-dev/train.jsonl") as f:
            for line in f:
                train_items.append(json.loads(line.decode("utf-8")))
        with zf.open("physicaliqa-train-dev/train-labels.lst") as f:
            for line in f:
                train_labels.append(int(line.decode("utf-8").strip()))
        # Read validation data
        with zf.open("physicaliqa-train-dev/dev.jsonl") as f:
            for line in f:
                val_items.append(json.loads(line.decode("utf-8")))
        with zf.open("physicaliqa-train-dev/dev-labels.lst") as f:
            for line in f:
                val_labels.append(int(line.decode("utf-8").strip()))

    def convert(item, label):
        goal = item["goal"].strip()
        sol1 = item["sol1"].strip()
        sol2 = item["sol2"].strip()

        correct_sol = sol1 if label == 0 else sol2
        correct_label = "A" if label == 0 else "B"

        prompt = f"Goal: {goal}\nA) {sol1}\nB) {sol2}\nWhich solution is better?"
        thinking = f"To achieve the goal of {goal.lower().rstrip('.')}, {correct_label}) is the better approach because it is more practical and physically sound."
        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": correct_sol,
            "domain": "physical",
        }

    train = [convert(item, label) for item, label in zip(train_items, train_labels)]
    val = [convert(item, label) for item, label in zip(val_items, val_labels)]
    print(f"    Train: {len(train)}, Val: {len(val)}")
    return train, val


def process_commonsenseqa() -> tuple:
    """Load and format CommonsenseQA dataset."""
    print("  Loading CommonsenseQA...")
    ds = load_dataset("tau/commonsense_qa")

    def convert(item):
        question = item["question"].strip()
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]

        # answerKey can sometimes be missing in test split
        if not answer_key or answer_key not in labels:
            return None

        answer_text = label_to_text(answer_key, labels, texts)

        prompt = f"{question}\n{format_choices(labels, texts)}"
        thinking = f"Thinking about this common sense question: {answer_key}) {answer_text} is the most reasonable answer based on everyday knowledge and reasoning."
        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer_text,
            "domain": "logic",
        }

    train = [x for x in (convert(item) for item in ds["train"]) if x is not None]
    val = [x for x in (convert(item) for item in ds["validation"]) if x is not None]
    print(f"    Train: {len(train)}, Val: {len(val)}")
    return train, val


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

def balance_and_combine(
    math_train, science_train, logic_train, physical_train,
    rng: random.Random,
) -> list:
    """Balance domains according to target proportions and combine.

    Target proportions:
      math: 40%, science: 25%, logic: 25%, physical: 10%

    Uses the math set size as the anchor (since GSM8K is fixed at ~7473).
    """
    math_count = len(math_train)

    # Calculate target sizes based on math being 40%
    total_target = int(math_count / 0.40)
    science_target = int(total_target * 0.25)
    logic_target = int(total_target * 0.25)
    physical_target = int(total_target * 0.10)

    print(f"\n  Balance targets (total ~{total_target}):")
    print(f"    math:     {math_count} (40%)")
    print(f"    science:  {science_target} (25%) from {len(science_train)} available")
    print(f"    logic:    {logic_target} (25%) from {len(logic_train)} available")
    print(f"    physical: {physical_target} (10%) from {len(physical_train)} available")

    def sample_or_oversample(data, target, rng):
        """Sample exactly target items, oversampling if needed."""
        if len(data) >= target:
            return rng.sample(data, target)
        else:
            # Oversample: repeat data and then sample remainder
            result = list(data)
            rng.shuffle(result)
            while len(result) < target:
                extra = list(data)
                rng.shuffle(extra)
                result.extend(extra[:target - len(result)])
            return result[:target]

    sampled_math = list(math_train)  # use all math
    sampled_science = sample_or_oversample(science_train, science_target, rng)
    sampled_logic = sample_or_oversample(logic_train, logic_target, rng)
    sampled_physical = sample_or_oversample(physical_train, physical_target, rng)

    combined = sampled_math + sampled_science + sampled_logic + sampled_physical
    rng.shuffle(combined)

    print(f"\n  Final combined training set: {len(combined)} examples")
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Multi-Domain Reasoning Data Preparation")
    print("=" * 60)

    rng = random.Random(SEED)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load all datasets
    # ------------------------------------------------------------------
    print("\n[1/5] Loading datasets...")

    gsm8k_train, gsm8k_val = load_gsm8k_existing()
    obqa_train, obqa_val = process_openbookqa()
    arc_easy_train, arc_easy_val = process_arc("ARC-Easy")
    arc_challenge_train, arc_challenge_val = process_arc("ARC-Challenge")
    boolq_train, boolq_val = process_boolq()
    piqa_train, piqa_val = process_piqa()
    csqa_train, csqa_val = process_commonsenseqa()

    # ------------------------------------------------------------------
    # Step 2: Group by domain
    # ------------------------------------------------------------------
    print("\n[2/5] Grouping by domain...")

    math_train = gsm8k_train
    math_val = gsm8k_val

    science_train = obqa_train + arc_easy_train + arc_challenge_train
    science_val = obqa_val + arc_easy_val + arc_challenge_val

    logic_train = boolq_train + csqa_train
    logic_val = boolq_val + csqa_val

    physical_train = piqa_train
    physical_val = piqa_val

    print(f"  math train: {len(math_train)}, val: {len(math_val)}")
    print(f"  science train: {len(science_train)}, val: {len(science_val)}")
    print(f"  logic train: {len(logic_train)}, val: {len(logic_val)}")
    print(f"  physical train: {len(physical_train)}, val: {len(physical_val)}")

    # ------------------------------------------------------------------
    # Step 3: Balance and combine training set
    # ------------------------------------------------------------------
    print("\n[3/5] Balancing and combining training set...")

    combined_train = balance_and_combine(
        math_train, science_train, logic_train, physical_train, rng
    )

    # Combine all validation data (no balancing needed for val)
    combined_val = math_val + science_val + logic_val + physical_val
    rng.shuffle(combined_val)
    print(f"  Combined validation set: {len(combined_val)} examples")

    # ------------------------------------------------------------------
    # Step 4: Write output files
    # ------------------------------------------------------------------
    print("\n[4/5] Writing output files...")

    # Main training and validation files
    write_jsonl(combined_train, OUTPUT_DIR / "train.jsonl")
    write_jsonl(combined_val, OUTPUT_DIR / "val.jsonl")

    # Prompts-only file for GRPO
    prompts_only = [{"prompt": ex["prompt"], "domain": ex["domain"]} for ex in combined_train]
    write_jsonl(prompts_only, OUTPUT_DIR / "train_prompts.jsonl")

    # Domain statistics
    domain_counts_train = {}
    for ex in combined_train:
        d = ex["domain"]
        domain_counts_train[d] = domain_counts_train.get(d, 0) + 1

    domain_counts_val = {}
    for ex in combined_val:
        d = ex["domain"]
        domain_counts_val[d] = domain_counts_val.get(d, 0) + 1

    stats = {
        "total_train": len(combined_train),
        "total_val": len(combined_val),
        "train_domain_counts": domain_counts_train,
        "val_domain_counts": domain_counts_val,
        "domain_proportions": {
            k: round(v / len(combined_train) * 100, 1)
            for k, v in sorted(domain_counts_train.items())
        },
    }

    stats_path = OUTPUT_DIR / "domain_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Wrote domain stats to {stats_path}")

    # ------------------------------------------------------------------
    # Step 5: Create eval sets (50 per domain)
    # ------------------------------------------------------------------
    print("\n[5/5] Creating per-domain eval sets (50 each)...")

    eval_counts = {}
    domain_val_pools = {
        "math": math_val,
        "science": science_val,
        "logic": logic_val,
        "physical": physical_val,
    }

    for domain, pool in domain_val_pools.items():
        eval_size = min(50, len(pool))
        eval_set = rng.sample(pool, eval_size)
        eval_path = EVAL_DIR / f"{domain}.jsonl"
        write_jsonl(eval_set, eval_path)
        eval_counts[domain] = eval_size

    print(f"\n  Eval counts: {eval_counts}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total training examples: {len(combined_train)}")
    print(f"  Total validation examples: {len(combined_val)}")
    print(f"  Domain proportions (train):")
    for domain, count in sorted(domain_counts_train.items()):
        pct = count / len(combined_train) * 100
        print(f"    {domain}: {count} ({pct:.1f}%)")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Eval directory: {EVAL_DIR}")

    # Spot-check one example per domain
    print("\n  Spot-check (one per domain):")
    seen = set()
    for ex in combined_train:
        d = ex["domain"]
        if d not in seen:
            seen.add(d)
            print(f"\n  [{d}]")
            print(f"    Prompt:   {ex['prompt'][:100]}...")
            print(f"    Thinking: {ex['thinking'][:100]}...")
            print(f"    Answer:   {ex['answer'][:80]}")
        if len(seen) == 4:
            break

    # Return summary info for output.json
    return {
        "total_train": len(combined_train),
        "total_val": len(combined_val),
        "domain_counts_train": domain_counts_train,
        "domain_counts_val": domain_counts_val,
        "eval_counts": eval_counts,
    }


if __name__ == "__main__":
    result = main()
    print("\nDone.")
