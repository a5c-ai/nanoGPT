"""
Reward functions for GRPO reinforcement learning training.

Provides rule-based reward signals for mathematical reasoning:
- accuracy_reward: exact match after normalization (math-specific)
- general_accuracy_reward: multi-domain matching (text, yes/no, multiple choice)
- format_reward: structural compliance with <think>/<answer> tags
- length_penalty: penalize too long/short responses
- compute_rewards: batch computation combining all signals (math)
- compute_rewards_multi: batch computation with domain-aware reward selection
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


def _normalize_general(text: str) -> str:
    """Normalize a general text answer: lowercase, strip whitespace/punctuation."""
    s = text.strip().lower()
    # Strip trailing punctuation (periods, commas, etc.)
    s = re.sub(r'[\.\,\;\:]+$', '', s).strip()
    return s


# Yes/no variants mapping
_YES_VARIANTS = {'yes', 'y', 'true', 'correct', 'right', 'affirmative', 'yeah', 'yep'}
_NO_VARIANTS = {'no', 'n', 'false', 'incorrect', 'wrong', 'negative', 'nah', 'nope'}


def _is_yes(text: str) -> bool:
    """Check if text represents a 'yes' answer."""
    return _normalize_general(text) in _YES_VARIANTS


def _is_no(text: str) -> bool:
    """Check if text represents a 'no' answer."""
    return _normalize_general(text) in _NO_VARIANTS


def _extract_mc_letter(text: str) -> str | None:
    """Extract a multiple-choice letter (A-D) from text.

    Handles formats like 'A', 'A)', '(A)', 'A.', 'Option A', etc.
    """
    s = _normalize_general(text)
    # Direct single letter
    if s in ('a', 'b', 'c', 'd'):
        return s
    # Patterns like (A), A), A., option A
    m = re.match(r'(?:\(?([a-d])\)?[\.\)]?|option\s+([a-d]))$', s)
    if m:
        return (m.group(1) or m.group(2))
    # Letter at the start followed by other text: "B. Paris"
    m = re.match(r'([a-d])[\.\)\s]', s)
    if m:
        return m.group(1)
    return None


def general_accuracy_reward(completion: str, ground_truth: str) -> float:
    """Multi-domain accuracy reward supporting text, yes/no, and multiple choice.

    Extracts answer from <answer>...</answer> tags (and other patterns via extract_answer),
    then attempts matching in this order:
      1. Exact match (normalized)
      2. Yes/no semantic match
      3. Multiple choice letter match
      4. Substring match (ground_truth in answer or answer in ground_truth)

    Returns 1.0 for match, 0.0 otherwise.
    """
    predicted = extract_answer(completion)
    if predicted is None:
        return 0.0

    pred_norm = _normalize_general(predicted)
    gt_norm = _normalize_general(ground_truth)

    # 1. Exact match (after normalization)
    if pred_norm == gt_norm:
        return 1.0

    # 2. Try math normalization for numeric answers
    try:
        if normalize_math_answer(predicted) == normalize_math_answer(ground_truth):
            return 1.0
    except Exception:
        pass

    # 3. Yes/no semantic match
    if (_is_yes(predicted) and _is_yes(ground_truth)) or \
       (_is_no(predicted) and _is_no(ground_truth)):
        return 1.0

    # 4. Multiple choice letter match
    pred_letter = _extract_mc_letter(predicted)
    gt_letter = _extract_mc_letter(ground_truth)
    if pred_letter is not None and gt_letter is not None and pred_letter == gt_letter:
        return 1.0

    # 5. Substring match (for short ground truths in longer answers or vice versa)
    if len(gt_norm) >= 2 and len(pred_norm) >= 2:
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return 1.0

    return 0.0


def compute_rewards_multi(
    completions: list[str],
    ground_truths: list[str],
    accuracy_weight: float = 1.0,
    format_weight: float = 0.5,
    length_weight: float = 0.1,
    domains: list[str] | None = None,
) -> list[float]:
    """Compute composite rewards using domain-aware accuracy reward selection.

    For each item, picks the appropriate reward function:
      - 'math': uses accuracy_reward (numeric normalization)
      - any other domain or None: uses general_accuracy_reward (multi-domain)

    If domains is None, uses general_accuracy_reward for all items.

    Args:
        completions: list of model completions
        ground_truths: list of ground truth answers (same length)
        accuracy_weight: weight for accuracy reward
        format_weight: weight for format reward
        length_weight: weight for length penalty
        domains: optional list of domain strings per item (e.g., 'math', 'science', 'yesno')

    Returns:
        List of composite reward scores.
    """
    assert len(completions) == len(ground_truths), \
        f"Mismatched lengths: {len(completions)} completions vs {len(ground_truths)} ground truths"
    if domains is not None:
        assert len(domains) == len(completions), \
            f"Mismatched lengths: {len(domains)} domains vs {len(completions)} completions"

    rewards = []
    for i, (comp, gt) in enumerate(zip(completions, ground_truths)):
        domain = domains[i] if domains is not None else None
        if domain == 'math':
            acc = accuracy_reward(comp, gt)
        else:
            acc = general_accuracy_reward(comp, gt)
        fmt = format_reward(comp)
        lp = length_penalty(comp)
        reward = accuracy_weight * acc + format_weight * fmt + length_weight * lp
        rewards.append(reward)
    return rewards


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
