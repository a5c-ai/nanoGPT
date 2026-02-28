"""
Evaluation framework for reasoning models.

Benchmarks: GSM8K test set, custom arithmetic sanity checks.
Metrics: pass@1, pass@k (unbiased), majority@k, format compliance, mean CoT length.
Bootstrap confidence intervals, multi-checkpoint comparison with McNemar test.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer_utils import ReasoningTokenizer, get_tokenizer
from reward import extract_answer, normalize_math_answer, accuracy_reward, format_reward


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k(path: str = "data/gsm8k_cot/val.jsonl", max_samples: int | None = None) -> list[dict]:
    """Load GSM8K test problems from JSONL file.

    Each line should have at least 'prompt' and 'answer' fields.
    """
    problems = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            problems.append(obj)
            if max_samples is not None and len(problems) >= max_samples:
                break
    return problems


def generate_sanity_checks(n: int = 50, seed: int = 42) -> list[dict]:
    """Generate simple arithmetic problems as sanity checks."""
    rng = random.Random(seed)
    problems = []
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    for _ in range(n):
        op_sym, op_fn = rng.choice(ops)
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        if op_sym == "-" and a < b:
            a, b = b, a  # keep non-negative
        answer = op_fn(a, b)
        prompt = f"What is {a} {op_sym} {b}?"
        problems.append({
            "prompt": prompt,
            "answer": str(answer),
        })
    return problems


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_completions(
    model: GPT,
    tokenizer: ReasoningTokenizer,
    prompts: list[str],
    k: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k_sampling: int | None = 50,
    device: str = "cuda",
    batch_size: int = 8,
) -> list[list[str]]:
    """Generate k completions per prompt. Returns list of list of decoded strings."""
    stop_tokens = {tokenizer.answer_end_id, tokenizer.eot_id}
    all_completions: list[list[str]] = []

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        completions_for_prompt: list[str] = []

        # Generate k samples, in batches
        remaining = k
        while remaining > 0:
            cur_batch = min(remaining, batch_size)
            idx = prompt_tensor.unsqueeze(0).expand(cur_batch, -1)
            result = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k_sampling,
                stop_tokens=stop_tokens,
            )
            if isinstance(result, dict):
                token_ids = result["token_ids"]
            else:
                token_ids = result

            for i in range(cur_batch):
                gen_ids = token_ids[i, len(prompt_ids):].tolist()
                text = tokenizer.decode(gen_ids)
                completions_for_prompt.append(text)
            remaining -= cur_batch

        all_completions.append(completions_for_prompt)
    return all_completions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pass_at_1(completions: list[str], ground_truth: str) -> float:
    """Fraction of completions that are correct (greedy = first sample)."""
    if not completions:
        return 0.0
    return accuracy_reward(completions[0], ground_truth)


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k given n samples with c correct.

    Uses the formula: 1 - comb(n-c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    # Use log-space to avoid overflow
    log_num = sum(math.log(n - c - i) for i in range(k))
    log_den = sum(math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_num - log_den)


def majority_at_k(completions: list[str], ground_truth: str) -> float:
    """1.0 if the majority-voted answer matches ground truth, else 0.0."""
    answers = []
    for comp in completions:
        ans = extract_answer(comp)
        if ans is not None:
            answers.append(normalize_math_answer(ans))
    if not answers:
        return 0.0
    counter = Counter(answers)
    majority_ans = counter.most_common(1)[0][0]
    return 1.0 if majority_ans == normalize_math_answer(ground_truth) else 0.0


def format_compliance_rate(completions: list[str]) -> float:
    """Fraction of completions with full format compliance (score 1.0)."""
    if not completions:
        return 0.0
    return sum(1.0 for c in completions if format_reward(c) == 1.0) / len(completions)


def mean_cot_length(completions: list[str], tokenizer: ReasoningTokenizer) -> float:
    """Mean token count of chain-of-thought (content between <think> tags)."""
    import re
    lengths = []
    for comp in completions:
        m = re.search(r'<think>(.*?)</think>', comp, re.DOTALL)
        if m:
            cot_text = m.group(1)
            lengths.append(len(tokenizer.encode(cot_text)))
        else:
            lengths.append(0)
    return float(np.mean(lengths)) if lengths else 0.0


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns (mean, lower, upper).
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    point_estimate = float(np.mean(arr))
    if len(arr) <= 1:
        return point_estimate, point_estimate, point_estimate

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return point_estimate, lower, upper


# ---------------------------------------------------------------------------
# McNemar test for comparing two checkpoints
# ---------------------------------------------------------------------------

def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """McNemar's test for paired binary outcomes.

    Returns dict with chi2 statistic, p-value, and contingency counts.
    """
    assert len(correct_a) == len(correct_b)
    # b01: A wrong, B right; b10: A right, B wrong
    b01 = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)
    b10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    b00 = sum(1 for a, b in zip(correct_a, correct_b) if not a and not b)
    b11 = sum(1 for a, b in zip(correct_a, correct_b) if a and b)

    if b01 + b10 == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        # McNemar with continuity correction
        chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        # Approximate p-value from chi2(1) using survival function
        # For simplicity, use a normal approximation
        from math import erfc, sqrt
        p_value = erfc(sqrt(chi2 / 2))  # two-sided

    return {
        "chi2": chi2,
        "p_value": p_value,
        "a_only": b10,
        "b_only": b01,
        "both_correct": b11,
        "both_wrong": b00,
    }


# ---------------------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    problems: list[dict],
    k: int = 16,
    device: str = "cuda",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> dict:
    """Evaluate a single checkpoint on a list of problems.

    Returns a dict of metrics with bootstrap CIs.
    """
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_args" in checkpoint:
        model_args = checkpoint["model_args"]
        gptconf = GPTConfig(**model_args)
    elif "config" in checkpoint:
        gptconf = checkpoint["config"]
    else:
        gptconf = GPTConfig()

    model = GPT(gptconf)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    # Strip DDP prefix if present
    unwrapped = {}
    for key, val in state_dict.items():
        new_key = key.replace("_orig_mod.", "").replace("module.", "")
        unwrapped[new_key] = val
    model.load_state_dict(unwrapped, strict=False)
    model.to(device)
    model.eval()

    tokenizer = get_tokenizer()

    # Generate completions
    prompts = [p["prompt"] for p in problems]
    ground_truths = [p["answer"] for p in problems]

    print(f"  Generating {k} completions for {len(prompts)} problems...")
    with torch.no_grad():
        all_completions = generate_completions(
            model, tokenizer, prompts,
            k=k, max_new_tokens=max_new_tokens,
            temperature=temperature, device=device,
            batch_size=batch_size,
        )

    # Compute per-problem metrics
    pass1_scores = []
    passk_scores = []
    majorityk_scores = []
    fmt_scores = []
    cot_lengths = []

    for i, (comps, gt) in enumerate(zip(all_completions, ground_truths)):
        # pass@1: use first completion
        p1 = accuracy_reward(comps[0], gt)
        pass1_scores.append(p1)

        # Count correct for pass@k
        n_correct = sum(1 for c in comps if accuracy_reward(c, gt) == 1.0)
        pk = pass_at_k_unbiased(len(comps), n_correct, min(k, len(comps)))
        passk_scores.append(pk)

        # majority@k
        mk = majority_at_k(comps, gt)
        majorityk_scores.append(mk)

        # Format compliance (across all completions)
        fmt_scores.append(format_compliance_rate(comps))

        # CoT length
        cot_lengths.append(mean_cot_length(comps, tokenizer))

    # Aggregate with bootstrap CIs
    results = {
        "checkpoint": checkpoint_path,
        "num_problems": len(problems),
        "k": k,
    }

    for name, scores in [
        ("pass@1", pass1_scores),
        (f"pass@{k}", passk_scores),
        (f"majority@{k}", majorityk_scores),
        ("format_compliance", fmt_scores),
        ("mean_cot_length", cot_lengths),
    ]:
        mean_val, ci_lo, ci_hi = bootstrap_ci(scores)
        results[name] = {
            "mean": round(mean_val, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        }

    # Per-problem correctness (for McNemar comparison)
    results["_per_problem_correct"] = [s == 1.0 for s in pass1_scores]

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: list[dict]) -> None:
    """Print a human-readable summary table to stdout."""
    metric_keys = []
    for r in all_results:
        for key in r:
            if key.startswith("_") or key in ("checkpoint", "num_problems", "k"):
                continue
            if key not in metric_keys:
                metric_keys.append(key)

    # Header
    ckpt_width = max(len(os.path.basename(r["checkpoint"])) for r in all_results)
    ckpt_width = max(ckpt_width, 10)
    header = f"{'Checkpoint':<{ckpt_width}}"
    for mk in metric_keys:
        header += f"  {mk:>28}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_results:
        name = os.path.basename(r["checkpoint"])
        row = f"{name:<{ckpt_width}}"
        for mk in metric_keys:
            val = r[mk]
            if isinstance(val, dict):
                row += f"  {val['mean']:>7.4f} [{val['ci_lower']:.4f}, {val['ci_upper']:.4f}]"
            else:
                row += f"  {val:>28}"
        print(row)
    print("=" * len(header))


def print_comparison(results_a: dict, results_b: dict) -> None:
    """Print McNemar comparison between two checkpoints."""
    correct_a = results_a["_per_problem_correct"]
    correct_b = results_b["_per_problem_correct"]
    test = mcnemar_test(correct_a, correct_b)

    name_a = os.path.basename(results_a["checkpoint"])
    name_b = os.path.basename(results_b["checkpoint"])

    print(f"\nMcNemar Test: {name_a} vs {name_b}")
    print(f"  Chi-squared: {test['chi2']:.4f}")
    print(f"  p-value:     {test['p_value']:.4f}")
    print(f"  {name_a} only correct: {test['a_only']}")
    print(f"  {name_b} only correct: {test['b_only']}")
    print(f"  Both correct: {test['both_correct']}")
    print(f"  Both wrong:   {test['both_wrong']}")
    sig = "YES" if test["p_value"] < 0.05 else "NO"
    print(f"  Significant (p<0.05): {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning model checkpoints")
    parser.add_argument(
        "--checkpoint", type=str, nargs="+", required=True,
        help="Path(s) to checkpoint file(s)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1319,
        help="Number of GSM8K problems to evaluate (default: 1319 = full test set)",
    )
    parser.add_argument(
        "--k", type=int, default=16,
        help="Number of samples per problem for pass@k / majority@k",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/gsm8k_cot/val.jsonl",
        help="Path to GSM8K validation JSONL",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
    )
    parser.add_argument(
        "--sanity-only", action="store_true",
        help="Only run sanity check arithmetic problems",
    )
    parser.add_argument(
        "--include-sanity", action="store_true",
        help="Also run sanity check arithmetic problems",
    )
    args = parser.parse_args()

    all_results = []

    for ckpt_path in args.checkpoint:
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_path}")
        print(f"{'='*60}")

        # GSM8K evaluation
        if not args.sanity_only:
            print("\n[GSM8K Evaluation]")
            problems = load_gsm8k(args.data_path, max_samples=args.num_samples)
            print(f"  Loaded {len(problems)} problems")
            gsm_results = evaluate_checkpoint(
                ckpt_path, problems,
                k=args.k, device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                batch_size=args.batch_size,
            )
            gsm_results["benchmark"] = "gsm8k"
            all_results.append(gsm_results)

        # Sanity check evaluation
        if args.sanity_only or args.include_sanity:
            print("\n[Sanity Check Evaluation]")
            sanity_n = min(args.num_samples, 50) if args.sanity_only else 50
            sanity_problems = generate_sanity_checks(n=sanity_n)
            print(f"  Generated {len(sanity_problems)} arithmetic problems")
            sanity_results = evaluate_checkpoint(
                ckpt_path, sanity_problems,
                k=args.k, device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                batch_size=args.batch_size,
            )
            sanity_results["benchmark"] = "sanity_arithmetic"
            all_results.append(sanity_results)

    # Print summary
    print("\n")
    print_summary_table(all_results)

    # Pairwise comparisons if multiple checkpoints on same benchmark
    gsm_results_list = [r for r in all_results if r.get("benchmark") == "gsm8k"]
    for i in range(len(gsm_results_list)):
        for j in range(i + 1, len(gsm_results_list)):
            print_comparison(gsm_results_list[i], gsm_results_list[j])

    # JSON output
    output = {
        "results": [],
    }
    for r in all_results:
        out_r = {k: v for k, v in r.items() if not k.startswith("_")}
        output["results"].append(out_r)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    # Also print JSON to stdout for easy capture
    print("\n--- JSON Output ---")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
