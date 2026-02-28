"""
Prepare GSM8K dataset for reasoning model training.

Downloads the openai/gsm8k dataset from HuggingFace and converts it into
{prompt, thinking, answer} format suitable for SFT and GRPO training.

Outputs:
  - gsm8k_cot/train.jsonl       Full examples with prompt, thinking, answer
  - gsm8k_cot/val.jsonl         Full examples (test split used as val)
  - gsm8k_cot/train_prompts.jsonl  Prompts only (for GRPO generation)
  - gsm8k_cot/val_prompts.jsonl    Prompts only (for evaluation)
"""

import json
import os
import re
import sys
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "gsm8k_cot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def strip_annotations(text: str) -> str:
    """Remove <<...>> calculator annotations from GSM8K solutions."""
    return re.sub(r"<<.*?>>", "", text)


def parse_example(question: str, answer_field: str) -> dict | None:
    """Parse a single GSM8K example into {prompt, thinking, answer}.

    The answer field in GSM8K has the format:
        Step-by-step reasoning...
        #### final_numeric_answer

    We strip <<...>> annotations and split on #### to get the reasoning
    (thinking) and the final answer separately.
    """
    # Split on the #### delimiter
    if "####" not in answer_field:
        return None

    parts = answer_field.split("####", maxsplit=1)
    thinking_raw = parts[0].strip()
    final_answer = parts[1].strip()

    # Strip calculator annotations from reasoning
    thinking = strip_annotations(thinking_raw)

    # Clean up any double spaces left by annotation removal
    thinking = re.sub(r"  +", " ", thinking)

    # Validate we have non-empty fields
    if not question.strip() or not thinking or not final_answer:
        return None

    return {
        "prompt": question.strip(),
        "thinking": thinking,
        "answer": final_answer,
    }


def process_split(dataset_split, split_name: str) -> list[dict]:
    """Process a HuggingFace dataset split into parsed examples."""
    examples = []
    skipped = 0

    for item in dataset_split:
        parsed = parse_example(item["question"], item["answer"])
        if parsed is not None:
            examples.append(parsed)
        else:
            skipped += 1

    print(f"  {split_name}: {len(examples)} examples parsed, {skipped} skipped")
    return examples


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write a list of dicts as a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {path}")


def write_prompts_jsonl(records: list[dict], path: Path) -> None:
    """Write prompts-only JSONL (for generation / evaluation)."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            prompt_record = {"prompt": record["prompt"]}
            f.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} prompts to {path}")


# ---------------------------------------------------------------------------
# Spot-check / verification
# ---------------------------------------------------------------------------

def spot_check(examples: list[dict], n: int = 5) -> None:
    """Print a few examples for manual inspection."""
    print(f"\n--- Spot-check ({n} examples) ---")
    for i, ex in enumerate(examples[:n]):
        print(f"\n[Example {i+1}]")
        print(f"  Prompt:   {ex['prompt'][:120]}...")
        print(f"  Thinking: {ex['thinking'][:120]}...")
        print(f"  Answer:   {ex['answer']}")
    print()


def verify_jsonl(path: Path, expected_fields: list[str]) -> int:
    """Verify a JSONL file: every line is valid JSON with expected fields."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ERROR: {path} line {line_no}: invalid JSON: {e}")
                return -1
            for field in expected_fields:
                if field not in obj:
                    print(f"  ERROR: {path} line {line_no}: missing field '{field}'")
                    return -1
                if not obj[field].strip():
                    print(f"  ERROR: {path} line {line_no}: empty field '{field}'")
                    return -1
            count += 1
    return count


def verify_no_annotations(path: Path, sample_size: int = 100) -> bool:
    """Check that <<...>> annotations have been stripped."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            obj = json.loads(line)
            if "<<" in obj.get("thinking", ""):
                print(f"  ERROR: annotation found in thinking at line {i+1}")
                return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GSM8K Reasoning Data Preparation")
    print("=" * 60)

    # 1. Download dataset
    print("\n[1/4] Downloading GSM8K from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main")
    print(f"  Train split: {len(ds['train'])} examples")
    print(f"  Test split:  {len(ds['test'])} examples")

    # 2. Parse into {prompt, thinking, answer}
    print("\n[2/4] Parsing examples...")
    train_examples = process_split(ds["train"], "train")
    val_examples = process_split(ds["test"], "val")

    # 3. Write output files
    print("\n[3/4] Writing output files...")
    write_jsonl(train_examples, OUTPUT_DIR / "train.jsonl")
    write_jsonl(val_examples, OUTPUT_DIR / "val.jsonl")
    write_prompts_jsonl(train_examples, OUTPUT_DIR / "train_prompts.jsonl")
    write_prompts_jsonl(val_examples, OUTPUT_DIR / "val_prompts.jsonl")

    # 4. Verification
    print("\n[4/4] Verification...")
    train_count = verify_jsonl(OUTPUT_DIR / "train.jsonl", ["prompt", "thinking", "answer"])
    val_count = verify_jsonl(OUTPUT_DIR / "val.jsonl", ["prompt", "thinking", "answer"])
    train_prompts_count = verify_jsonl(OUTPUT_DIR / "train_prompts.jsonl", ["prompt"])
    val_prompts_count = verify_jsonl(OUTPUT_DIR / "val_prompts.jsonl", ["prompt"])

    print(f"\n  train.jsonl:         {train_count} valid records")
    print(f"  val.jsonl:           {val_count} valid records")
    print(f"  train_prompts.jsonl: {train_prompts_count} valid records")
    print(f"  val_prompts.jsonl:   {val_prompts_count} valid records")

    # Check annotations stripped
    ann_ok = verify_no_annotations(OUTPUT_DIR / "train.jsonl")
    print(f"  Annotations stripped: {'PASS' if ann_ok else 'FAIL'}")

    # Check expected counts
    ok = True
    if train_count < 7400:
        print(f"  WARNING: train count {train_count} < 7400 expected")
        ok = False
    if val_count < 1000:
        print(f"  WARNING: val count {val_count} < 1000 expected")
        ok = False

    # Spot-check
    spot_check(train_examples)

    if ok and train_count > 0 and val_count > 0 and ann_ok:
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION FAILED")
        sys.exit(1)

    print(f"\nDone. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
