"""
Reward functions for GRPO reinforcement learning training.

Provides rule-based reward signals for mathematical reasoning:
- accuracy_reward: exact match after normalization
- format_reward: structural compliance with <think>/<answer> tags
- length_penalty: penalize too long/short responses
- compute_rewards: batch computation combining all signals
"""

import re
from fractions import Fraction


def extract_answer(text: str) -> str | None:
    """Extract answer from completion using cascade of patterns.

    Priority: <answer> tags > \\boxed{} > "the answer is" > ####
    """
    # 1. <answer>...</answer> tags
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. \boxed{...}
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    if m:
        return m.group(1).strip()

    # 3. "the answer is ..."
    m = re.search(r'[Tt]he\s+answer\s+is[:\s]+([^\.\n]+)', text)
    if m:
        return m.group(1).strip()

    # 4. #### <answer>
    m = re.search(r'####\s*(.+)', text)
    if m:
        return m.group(1).strip()

    return None


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison.

    Handles: $, %, commas, fractions, decimals, whitespace.
    """
    s = answer.strip()
    # Remove $, %, leading/trailing whitespace
    s = s.replace('$', '').replace('%', '').replace(',', '').strip()

    # Try to evaluate as fraction (e.g., "3/4" -> 0.75)
    try:
        if '/' in s and not any(c.isalpha() for c in s):
            val = float(Fraction(s))
            # Normalize to remove trailing zeros
            if val == int(val):
                return str(int(val))
            return f"{val:.10f}".rstrip('0').rstrip('.')
    except (ValueError, ZeroDivisionError):
        pass

    # Try to parse as float to normalize representation
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return f"{val:.10f}".rstrip('0').rstrip('.')
    except ValueError:
        pass

    # Fallback: lowercase stripped string
    return s.lower().strip()


def accuracy_reward(completion: str, ground_truth: str) -> float:
    """Return 1.0 if extracted answer matches ground truth, else 0.0."""
    predicted = extract_answer(completion)
    if predicted is None:
        return 0.0
    return 1.0 if normalize_math_answer(predicted) == normalize_math_answer(ground_truth) else 0.0


def format_reward(completion: str) -> float:
    """Check <think>...</think><answer>...</answer> structure.

    Returns: 1.0 (full compliance), 0.5 (partial), 0.0 (none).
    """
    has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))

    if has_think and has_answer:
        return 1.0
    elif has_think or has_answer:
        return 0.5
    else:
        return 0.0


def length_penalty(completion: str, min_len: int = 10, max_len: int = 1024,
                   ideal_min: int = 50, ideal_max: int = 512) -> float:
    """Penalize responses that are too short or too long.

    Returns:
        1.0 for ideal length range
        Linear decay to 0.0 outside ideal range
        0.0 below min_len or above max_len
    """
    n = len(completion)
    if n < min_len or n > max_len:
        return 0.0
    if ideal_min <= n <= ideal_max:
        return 1.0
    if n < ideal_min:
        return (n - min_len) / (ideal_min - min_len)
    # n > ideal_max
    return (max_len - n) / (max_len - ideal_max)


def compute_rewards(
    completions: list[str],
    ground_truths: list[str],
    accuracy_weight: float = 1.0,
    format_weight: float = 0.5,
    length_weight: float = 0.1,
) -> list[float]:
    """Compute composite rewards for a batch of completions.

    Args:
        completions: list of model completions
        ground_truths: list of ground truth answers (same length)
        accuracy_weight: weight for accuracy reward
        format_weight: weight for format reward
        length_weight: weight for length penalty

    Returns:
        List of composite reward scores.
    """
    assert len(completions) == len(ground_truths), \
        f"Mismatched lengths: {len(completions)} completions vs {len(ground_truths)} ground truths"

    rewards = []
    for comp, gt in zip(completions, ground_truths):
        acc = accuracy_reward(comp, gt)
        fmt = format_reward(comp)
        lp = length_penalty(comp)
        reward = accuracy_weight * acc + format_weight * fmt + length_weight * lp
        rewards.append(reward)
    return rewards
