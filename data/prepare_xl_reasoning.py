"""
Prepare XL multi-domain reasoning training dataset for nanoGPT.

Downloads and processes open reasoning datasets from HuggingFace into a
unified Chain-of-Thought (CoT) format suitable for SFT training of a 1.5B
parameter model.

Datasets:
  - GSM8K (train) - math, has native CoT
  - MetaMathQA (meta-math/MetaMathQA) - math augmented, has native CoT
  - ARC-Challenge (allenai/ai2_arc) - science
  - ARC-Easy (allenai/ai2_arc) - science
  - SciQ (allenai/sciq) - science
  - OpenBookQA (allenai/openbookqa) - science
  - BoolQ (google/boolq) - boolean reasoning
  - CommonsenseQA (tau/commonsense_qa) - commonsense
  - PIQA (ybisk/piqa) - physical intuition

Output format per example:
  {"prompt": "...", "thinking": "...", "answer": "...",
   "domain": "math|science|logic|physical",
   "difficulty": "easy|medium|hard", "source": "dataset_name"}

Outputs (data/xl_reasoning/):
  train_full.jsonl       - all training examples, shuffled
  train_easy.jsonl       - easy difficulty only
  train_medium.jsonl     - medium difficulty only
  train_hard.jsonl       - hard difficulty only
  val.jsonl              - 10% held-out validation per domain
  train_prompts.jsonl    - prompt + answer only (for GRPO)
  stats.json             - dataset statistics
  eval/gsm8k_test.jsonl  - 200 GSM8K test problems
  eval/arc_test.jsonl    - 200 ARC-Challenge test problems
  eval/boolq_test.jsonl  - 200 BoolQ validation problems
  eval/commonsense_test.jsonl - 200 CommonsenseQA validation problems
"""

import io
import json
import os
import random
import sys
import traceback
import urllib.request
import zipfile
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not found. Install with: pip install datasets")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "xl_reasoning"
EVAL_DIR = OUTPUT_DIR / "eval"
SEED = 42
MAX_COT_TOKENS = 512  # approximate token limit (chars / 4)
MAX_COT_CHARS = MAX_COT_TOKENS * 4  # rough char proxy for token count

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: list, path: Path) -> None:
    """Write a list of dicts as a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records -> {path.name}")


def approx_tokens(text: str) -> int:
    """Rough token count (chars / 4)."""
    return len(text) // 4


def format_choices(labels: list, texts: list) -> str:
    """Format multiple-choice options as A) text, B) text, ..."""
    return "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))


def label_to_text(answer_key: str, labels: list, texts: list) -> str:
    """Map answer label to its text."""
    for l, t in zip(labels, texts):
        if str(l) == str(answer_key):
            return t
    return answer_key


def quality_filter(records: list) -> list:
    """Remove empty prompts/answers, duplicates, and overly long CoT."""
    seen = set()
    filtered = []
    for r in records:
        prompt = (r.get("prompt") or "").strip()
        answer = (r.get("answer") or "").strip()
        thinking = (r.get("thinking") or "").strip()
        if not prompt or not answer:
            continue
        # Deduplicate on prompt
        if prompt in seen:
            continue
        seen.add(prompt)
        # Limit CoT length
        if approx_tokens(thinking) > MAX_COT_TOKENS:
            thinking = thinking[:MAX_COT_CHARS].rsplit(".", 1)[0] + "."
        r["prompt"] = prompt
        r["answer"] = answer
        r["thinking"] = thinking
        filtered.append(r)
    return filtered


def safe_load_dataset(name, *args, **kwargs):
    """Attempt to load a HuggingFace dataset, returning None on failure."""
    try:
        return load_dataset(name, *args, **kwargs)
    except Exception as e:
        print(f"  WARNING: Failed to load {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Dataset processors - each returns (train_records, val_records)
# ---------------------------------------------------------------------------

def process_gsm8k() -> tuple:
    """GSM8K: math word problems with native CoT in 'answer' field."""
    print("  Loading GSM8K...")
    ds = safe_load_dataset("openai/gsm8k", "main")
    if ds is None:
        return [], []

    def convert(item, difficulty="easy"):
        question = item["question"].strip()
        raw_answer = item["answer"].strip()
        # GSM8K answer format: "step1\nstep2\n...\n#### final_number"
        parts = raw_answer.split("####")
        if len(parts) == 2:
            thinking = parts[0].strip()
            final_answer = parts[1].strip()
        else:
            thinking = raw_answer
            final_answer = raw_answer

        # Difficulty: problems with more steps are harder
        step_count = thinking.count("\n") + 1
        if step_count >= 6:
            diff = "hard"
        elif step_count >= 3:
            diff = "easy"
        else:
            diff = "easy"

        return {
            "prompt": question,
            "thinking": thinking,
            "answer": final_answer,
            "domain": "math",
            "difficulty": diff,
            "source": "gsm8k",
        }

    train = [convert(item) for item in ds["train"]]
    # We'll use test split for eval separately
    print(f"    GSM8K train: {len(train)}")
    return train, []


def process_gsm8k_test() -> list:
    """GSM8K test split for eval."""
    print("  Loading GSM8K test split for eval...")
    ds = safe_load_dataset("openai/gsm8k", "main")
    if ds is None:
        return []

    results = []
    for item in ds["test"]:
        question = item["question"].strip()
        raw_answer = item["answer"].strip()
        parts = raw_answer.split("####")
        thinking = parts[0].strip() if len(parts) == 2 else raw_answer
        final_answer = parts[1].strip() if len(parts) == 2 else raw_answer
        results.append({
            "prompt": question,
            "thinking": thinking,
            "answer": final_answer,
            "domain": "math",
            "difficulty": "easy",
            "source": "gsm8k_test",
        })
    return results


def process_metamathqa() -> tuple:
    """MetaMathQA: augmented math with CoT in 'response' field."""
    print("  Loading MetaMathQA...")
    ds = safe_load_dataset("meta-math/MetaMathQA")
    if ds is None:
        return [], []

    def convert(item):
        query = (item.get("query") or item.get("question") or "").strip()
        response = (item.get("response") or item.get("answer") or "").strip()
        if not query or not response:
            return None

        # Extract final answer: often after "The answer is" or last line
        thinking = response
        answer = response
        if "The answer is" in response:
            parts = response.rsplit("The answer is", 1)
            thinking = parts[0].strip()
            answer = parts[1].strip().rstrip(".")
        elif "\\boxed{" in response:
            # LaTeX boxed answer
            import re
            match = re.search(r"\\boxed\{([^}]+)\}", response)
            if match:
                answer = match.group(1)
                thinking = response[:response.index("\\boxed{")].strip()

        # Filter out very long CoT
        if approx_tokens(thinking) > MAX_COT_TOKENS:
            return None

        # MetaMathQA problems are generally harder
        difficulty = "hard" if len(thinking) > 500 else "medium"

        return {
            "prompt": query,
            "thinking": thinking,
            "answer": answer,
            "domain": "math",
            "difficulty": difficulty,
            "source": "metamathqa",
        }

    records = []
    for item in ds["train"]:
        r = convert(item)
        if r is not None:
            records.append(r)
        if len(records) >= 100000:  # Cap at 100K to keep manageable
            break

    print(f"    MetaMathQA records: {len(records)}")
    return records, []


def process_arc(config_name: str) -> tuple:
    """ARC dataset (Easy or Challenge config)."""
    print(f"  Loading ARC ({config_name})...")
    ds = safe_load_dataset("allenai/ai2_arc", config_name)
    if ds is None:
        return [], []

    is_challenge = "Challenge" in config_name
    difficulty = "medium" if is_challenge else "easy"
    source = "arc_challenge" if is_challenge else "arc_easy"

    def convert(item):
        question = item["question"].strip()
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]
        answer_text = label_to_text(answer_key, labels, texts)

        prompt = f"{question}\n{format_choices(labels, texts)}"
        if is_challenge:
            thinking = (
                f"The question asks about {question[:80].lower().rstrip('?.')}. "
                f"Looking at the options, option {answer_key}) {answer_text} is correct "
                f"because it best aligns with the relevant scientific principles."
            )
        else:
            thinking = (
                f"This is a straightforward science question. "
                f"The answer is {answer_key}) {answer_text}."
            )
        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer_text,
            "domain": "science",
            "difficulty": difficulty,
            "source": source,
        }

    train = [convert(item) for item in ds["train"]]
    val = [convert(item) for item in ds["validation"]]
    print(f"    ARC {config_name} train: {len(train)}, val: {len(val)}")
    return train, val


def process_arc_challenge_test() -> list:
    """ARC-Challenge test split for eval."""
    print("  Loading ARC-Challenge test split for eval...")
    ds = safe_load_dataset("allenai/ai2_arc", "ARC-Challenge")
    if ds is None:
        return []

    results = []
    for item in ds["test"]:
        question = item["question"].strip()
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]
        answer_text = label_to_text(answer_key, labels, texts)
        prompt = f"{question}\n{format_choices(labels, texts)}"
        thinking = (
            f"The question asks about {question[:80].lower().rstrip('?.')}. "
            f"Looking at the options, option {answer_key}) {answer_text} is correct "
            f"because it best aligns with the relevant scientific principles."
        )
        results.append({
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer_text,
            "domain": "science",
            "difficulty": "medium",
            "source": "arc_challenge_test",
        })
    return results


def process_sciq() -> tuple:
    """SciQ: science exam questions with supporting paragraphs."""
    print("  Loading SciQ...")
    ds = safe_load_dataset("allenai/sciq")
    if ds is None:
        return [], []

    def convert(item):
        question = item["question"].strip()
        correct = item["correct_answer"].strip()
        support = (item.get("support") or "").strip()

        # Build distractor options
        distractors = [
            item.get("distractor1", ""),
            item.get("distractor2", ""),
            item.get("distractor3", ""),
        ]
        distractors = [d.strip() for d in distractors if d.strip()]

        # Create multiple choice format
        all_options = [correct] + distractors
        random.shuffle(all_options)
        labels = ["A", "B", "C", "D"][:len(all_options)]
        correct_label = labels[all_options.index(correct)]

        prompt = f"{question}\n{format_choices(labels, all_options)}"

        if support:
            short_support = support[:200]
            thinking = (
                f"Based on the scientific context: {short_support}... "
                f"The answer is {correct_label}) {correct}."
            )
        else:
            thinking = (
                f"This science question asks about {question[:60].lower().rstrip('?.')}. "
                f"The correct answer is {correct_label}) {correct}."
            )

        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": correct,
            "domain": "science",
            "difficulty": "easy",
            "source": "sciq",
        }

    train = [convert(item) for item in ds["train"]]
    val = [convert(item) for item in ds["validation"]]
    print(f"    SciQ train: {len(train)}, val: {len(val)}")
    return train, val


def process_openbookqa() -> tuple:
    """OpenBookQA: elementary science with science facts."""
    print("  Loading OpenBookQA...")
    ds = safe_load_dataset("allenai/openbookqa", "main")
    if ds is None:
        return [], []

    def convert(item):
        question = item["question_stem"].strip()
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]
        answer_text = label_to_text(answer_key, labels, texts)
        fact = (item.get("fact1") or "").strip()

        prompt = f"{question}\n{format_choices(labels, texts)}"
        if fact:
            thinking = (
                f"Considering the scientific fact: {fact}. "
                f"This means the answer is {answer_key}) {answer_text}."
            )
        else:
            thinking = (
                f"This science question requires reasoning about {question[:60].lower().rstrip('?.')}. "
                f"The correct answer is {answer_key}) {answer_text}."
            )
        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer_text,
            "domain": "science",
            "difficulty": "medium",
            "source": "openbookqa",
        }

    train = [convert(item) for item in ds["train"]]
    val = [convert(item) for item in ds["validation"]]
    print(f"    OpenBookQA train: {len(train)}, val: {len(val)}")
    return train, val


def process_boolq() -> tuple:
    """BoolQ: yes/no reading comprehension questions."""
    print("  Loading BoolQ...")
    ds = safe_load_dataset("google/boolq")
    if ds is None:
        return [], []

    def convert(item):
        passage = item["passage"].strip()
        question = item["question"].strip()
        answer = item["answer"]  # True/False
        answer_str = "yes" if answer else "no"

        # Keep passage manageable
        short_passage = passage[:400]
        if len(passage) > 400:
            short_passage += "..."

        prompt = f"Passage: {short_passage}\n\nQuestion: {question}?\nAnswer yes or no."
        thinking = (
            f"Let me consider the passage. It states relevant information about "
            f"{question[:60].rstrip('?.')}. Based on this, the answer is {answer_str}."
        )
        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer_str,
            "domain": "logic",
            "difficulty": "easy",
            "source": "boolq",
        }

    train = [convert(item) for item in ds["train"]]
    val = [convert(item) for item in ds["validation"]]
    print(f"    BoolQ train: {len(train)}, val: {len(val)}")
    return train, val


def process_commonsenseqa() -> tuple:
    """CommonsenseQA: common sense reasoning multiple choice."""
    print("  Loading CommonsenseQA...")
    ds = safe_load_dataset("tau/commonsense_qa")
    if ds is None:
        return [], []

    def convert(item):
        question = item["question"].strip()
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        answer_key = item["answerKey"]
        if not answer_key or answer_key not in labels:
            return None
        answer_text = label_to_text(answer_key, labels, texts)

        prompt = f"{question}\n{format_choices(labels, texts)}"
        thinking = (
            f"The question asks about {question[:60].lower().rstrip('?.')}. "
            f"Using common sense reasoning, {answer_key}) {answer_text} is the most "
            f"reasonable answer because it best fits everyday knowledge."
        )
        return {
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer_text,
            "domain": "logic",
            "difficulty": "medium",
            "source": "commonsenseqa",
        }

    train = [x for x in (convert(item) for item in ds["train"]) if x is not None]
    val = [x for x in (convert(item) for item in ds["validation"]) if x is not None]
    print(f"    CommonsenseQA train: {len(train)}, val: {len(val)}")
    return train, val


def process_piqa() -> tuple:
    """PIQA: physical intuition QA.

    Downloads directly from source since the HF dataset can have loading issues.
    Falls back to HuggingFace if direct download fails.
    """
    print("  Loading PIQA...")

    # Try direct download first (more reliable)
    try:
        piqa_url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
        response = urllib.request.urlopen(piqa_url, timeout=60)
        zip_data = io.BytesIO(response.read())

        train_items, val_items = [], []
        train_labels, val_labels = [], []

        with zipfile.ZipFile(zip_data) as zf:
            with zf.open("physicaliqa-train-dev/train.jsonl") as f:
                for line in f:
                    train_items.append(json.loads(line.decode("utf-8")))
            with zf.open("physicaliqa-train-dev/train-labels.lst") as f:
                for line in f:
                    train_labels.append(int(line.decode("utf-8").strip()))
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
            thinking = (
                f"To achieve the goal of {goal[:80].lower().rstrip('.')}, "
                f"option {correct_label}) is better because it is more practical "
                f"and physically sound."
            )
            return {
                "prompt": prompt,
                "thinking": thinking,
                "answer": correct_sol,
                "domain": "physical",
                "difficulty": "medium",
                "source": "piqa",
            }

        train = [convert(item, label) for item, label in zip(train_items, train_labels)]
        val = [convert(item, label) for item, label in zip(val_items, val_labels)]
        print(f"    PIQA train: {len(train)}, val: {len(val)}")
        return train, val

    except Exception as e:
        print(f"  WARNING: PIQA direct download failed ({e}), trying HuggingFace...")
        try:
            ds = load_dataset("ybisk/piqa")
            def convert_hf(item):
                goal = item["goal"].strip()
                sol1 = item["sol1"].strip()
                sol2 = item["sol2"].strip()
                label = item["label"]
                if label < 0:
                    return None
                correct_sol = sol1 if label == 0 else sol2
                correct_label = "A" if label == 0 else "B"
                prompt = f"Goal: {goal}\nA) {sol1}\nB) {sol2}\nWhich solution is better?"
                thinking = (
                    f"To achieve the goal of {goal[:80].lower().rstrip('.')}, "
                    f"option {correct_label}) is better because it is more practical "
                    f"and physically sound."
                )
                return {
                    "prompt": prompt,
                    "thinking": thinking,
                    "answer": correct_sol,
                    "domain": "physical",
                    "difficulty": "medium",
                    "source": "piqa",
                }
            train = [x for x in (convert_hf(i) for i in ds["train"]) if x]
            val = [x for x in (convert_hf(i) for i in ds["validation"]) if x]
            print(f"    PIQA train: {len(train)}, val: {len(val)}")
            return train, val
        except Exception as e2:
            print(f"  WARNING: PIQA HuggingFace also failed ({e2}), skipping.")
            return [], []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("XL Reasoning Data Pipeline")
    print("=" * 70)

    rng = random.Random(SEED)
    random.seed(SEED)  # for shuffles inside processors

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Download and process all datasets
    # ------------------------------------------------------------------
    print("\n[1/6] Downloading and processing datasets...\n")

    all_train = []
    all_val = []
    dataset_counts = {}

    processors = [
        ("gsm8k", process_gsm8k),
        ("metamathqa", process_metamathqa),
        ("arc_challenge", lambda: process_arc("ARC-Challenge")),
        ("arc_easy", lambda: process_arc("ARC-Easy")),
        ("sciq", process_sciq),
        ("openbookqa", process_openbookqa),
        ("boolq", process_boolq),
        ("commonsenseqa", process_commonsenseqa),
        ("piqa", process_piqa),
    ]

    for name, processor in processors:
        try:
            train, val = processor()
            dataset_counts[name] = {"train": len(train), "val": len(val)}
            all_train.extend(train)
            all_val.extend(val)
            print(f"    -> {name}: {len(train):,} train, {len(val):,} val\n")
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            traceback.print_exc()
            dataset_counts[name] = {"train": 0, "val": 0, "error": str(e)}
            print()

    print(f"\n  Raw totals: {len(all_train):,} train, {len(all_val):,} val")

    # ------------------------------------------------------------------
    # Step 2: Quality filtering
    # ------------------------------------------------------------------
    print("\n[2/6] Quality filtering...")
    before_train = len(all_train)
    before_val = len(all_val)
    all_train = quality_filter(all_train)
    all_val = quality_filter(all_val)
    print(f"  Train: {before_train:,} -> {len(all_train):,} (removed {before_train - len(all_train):,})")
    print(f"  Val:   {before_val:,} -> {len(all_val):,} (removed {before_val - len(all_val):,})")

    # ------------------------------------------------------------------
    # Step 3: Split train into train/val (add 10% of train to val)
    # ------------------------------------------------------------------
    print("\n[3/6] Creating train/val split (10% of each domain for validation)...")

    # Group by domain
    domain_train = {}
    for r in all_train:
        d = r["domain"]
        domain_train.setdefault(d, []).append(r)

    final_train = []
    extra_val = []
    for domain, records in domain_train.items():
        rng.shuffle(records)
        split_idx = max(1, len(records) // 10)
        extra_val.extend(records[:split_idx])
        final_train.extend(records[split_idx:])

    # Combine validation
    final_val = all_val + extra_val
    rng.shuffle(final_train)
    rng.shuffle(final_val)

    print(f"  Final train: {len(final_train):,}")
    print(f"  Final val:   {len(final_val):,}")

    # ------------------------------------------------------------------
    # Step 4: Split by difficulty
    # ------------------------------------------------------------------
    print("\n[4/6] Splitting by difficulty...")

    train_easy = [r for r in final_train if r["difficulty"] == "easy"]
    train_medium = [r for r in final_train if r["difficulty"] == "medium"]
    train_hard = [r for r in final_train if r["difficulty"] == "hard"]

    print(f"  Easy:   {len(train_easy):,}")
    print(f"  Medium: {len(train_medium):,}")
    print(f"  Hard:   {len(train_hard):,}")

    # ------------------------------------------------------------------
    # Step 5: Write output files
    # ------------------------------------------------------------------
    print("\n[5/6] Writing output files...")

    write_jsonl(final_train, OUTPUT_DIR / "train_full.jsonl")
    write_jsonl(train_easy, OUTPUT_DIR / "train_easy.jsonl")
    write_jsonl(train_medium, OUTPUT_DIR / "train_medium.jsonl")
    write_jsonl(train_hard, OUTPUT_DIR / "train_hard.jsonl")
    write_jsonl(final_val, OUTPUT_DIR / "val.jsonl")

    # Prompts only for GRPO
    prompts_only = [
        {"prompt": r["prompt"], "answer": r["answer"], "domain": r["domain"]}
        for r in final_train
    ]
    write_jsonl(prompts_only, OUTPUT_DIR / "train_prompts.jsonl")

    # ------------------------------------------------------------------
    # Step 6: Create eval sets (200 each from test/val splits)
    # ------------------------------------------------------------------
    print("\n[6/6] Creating eval sets (200 each)...")

    eval_counts = {}

    # GSM8K test
    try:
        gsm8k_test = process_gsm8k_test()
        gsm8k_eval = rng.sample(gsm8k_test, min(200, len(gsm8k_test))) if gsm8k_test else []
        write_jsonl(gsm8k_eval, EVAL_DIR / "gsm8k_test.jsonl")
        eval_counts["gsm8k_test"] = len(gsm8k_eval)
    except Exception as e:
        print(f"  WARNING: GSM8K eval failed: {e}")
        eval_counts["gsm8k_test"] = 0

    # ARC-Challenge test
    try:
        arc_test = process_arc_challenge_test()
        arc_eval = rng.sample(arc_test, min(200, len(arc_test))) if arc_test else []
        write_jsonl(arc_eval, EVAL_DIR / "arc_test.jsonl")
        eval_counts["arc_test"] = len(arc_eval)
    except Exception as e:
        print(f"  WARNING: ARC eval failed: {e}")
        eval_counts["arc_test"] = 0

    # BoolQ validation (use the val records we already have)
    try:
        boolq_val_records = [r for r in final_val if r["source"] == "boolq"]
        if len(boolq_val_records) < 200:
            # Also check all_val
            boolq_val_records = [r for r in (all_val + extra_val) if r.get("source") == "boolq"]
        boolq_eval = rng.sample(boolq_val_records, min(200, len(boolq_val_records))) if boolq_val_records else []
        write_jsonl(boolq_eval, EVAL_DIR / "boolq_test.jsonl")
        eval_counts["boolq_test"] = len(boolq_eval)
    except Exception as e:
        print(f"  WARNING: BoolQ eval failed: {e}")
        eval_counts["boolq_test"] = 0

    # CommonsenseQA validation
    try:
        csqa_val_records = [r for r in final_val if r.get("source") == "commonsenseqa"]
        if len(csqa_val_records) < 200:
            csqa_val_records = [r for r in (all_val + extra_val) if r.get("source") == "commonsenseqa"]
        csqa_eval = rng.sample(csqa_val_records, min(200, len(csqa_val_records))) if csqa_val_records else []
        write_jsonl(csqa_eval, EVAL_DIR / "commonsense_test.jsonl")
        eval_counts["commonsense_test"] = len(csqa_eval)
    except Exception as e:
        print(f"  WARNING: CommonsenseQA eval failed: {e}")
        eval_counts["commonsense_test"] = 0

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    domain_counts = {}
    for r in final_train:
        d = r["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1

    difficulty_counts = {}
    for r in final_train:
        d = r["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

    source_counts = {}
    for r in final_train:
        s = r["source"]
        source_counts[s] = source_counts.get(s, 0) + 1

    # List output files
    output_files = []
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            output_files.append(str(f.relative_to(SCRIPT_DIR)))

    stats = {
        "status": "ok",
        "totalTrain": len(final_train),
        "totalVal": len(final_val),
        "domainCounts": domain_counts,
        "difficultyCounts": difficulty_counts,
        "sourceCounts": source_counts,
        "evalCounts": eval_counts,
        "datasetCounts": dataset_counts,
        "files": output_files,
        "summary": (
            f"XL reasoning dataset: {len(final_train):,} train, {len(final_val):,} val examples "
            f"across {len(domain_counts)} domains ({', '.join(f'{k}:{v:,}' for k, v in sorted(domain_counts.items()))}), "
            f"3 difficulty levels ({', '.join(f'{k}:{v:,}' for k, v in sorted(difficulty_counts.items()))}), "
            f"from {len(source_counts)} sources. "
            f"Eval sets: {', '.join(f'{k}:{v}' for k, v in sorted(eval_counts.items()))}."
        ),
    }

    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Wrote stats -> {stats_path.name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total training:   {len(final_train):,}")
    print(f"  Total validation: {len(final_val):,}")
    print(f"\n  Domain breakdown (train):")
    for d, c in sorted(domain_counts.items()):
        pct = c / len(final_train) * 100 if final_train else 0
        print(f"    {d:12s}: {c:>8,} ({pct:5.1f}%)")
    print(f"\n  Difficulty breakdown (train):")
    for d, c in sorted(difficulty_counts.items()):
        pct = c / len(final_train) * 100 if final_train else 0
        print(f"    {d:12s}: {c:>8,} ({pct:5.1f}%)")
    print(f"\n  Source breakdown (train):")
    for s, c in sorted(source_counts.items()):
        print(f"    {s:20s}: {c:>8,}")
    print(f"\n  Eval sets:")
    for k, v in sorted(eval_counts.items()):
        print(f"    {k:20s}: {v}")
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  Files:  {len(output_files)}")

    return stats


if __name__ == "__main__":
    result = main()
    print("\n" + json.dumps(result, indent=2))
    print("\nDone.")
