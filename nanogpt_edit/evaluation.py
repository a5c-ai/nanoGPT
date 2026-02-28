"""Evaluation framework for model editing (T4.1-T4.4)."""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .edit_core import ModelEditor


def _get_next_token_logits(editor: ModelEditor, prompt: str) -> torch.Tensor:
    """Run prompt through model and return logits for the next token.

    Returns:
        Logits tensor of shape (vocab_size,).
    """
    device = next(editor.model.parameters()).device
    input_ids = torch.tensor(
        [editor.tokenizer.encode(prompt)], dtype=torch.long, device=device
    )
    editor.model.eval()
    with torch.no_grad():
        logits, _ = editor.model(input_ids)
    # logits shape: (1, 1, vocab_size) in inference mode (no targets)
    return logits[0, -1]


def _get_full_logits(editor: ModelEditor, prompt: str) -> torch.Tensor:
    """Run prompt and return logits at all positions.

    Returns:
        Logits tensor of shape (seq_len, vocab_size).
    """
    device = next(editor.model.parameters()).device
    input_ids = torch.tensor(
        [editor.tokenizer.encode(prompt)], dtype=torch.long, device=device
    )
    editor.model.eval()
    # Pass dummy targets to get full logits (not just last position)
    dummy_targets = torch.zeros_like(input_ids)
    with torch.no_grad():
        logits, _ = editor.model(input_ids, targets=dummy_targets)
    return logits[0]  # (seq_len, vocab_size)


def _target_token_id(editor: ModelEditor, target_new: str) -> int:
    """Get the first token ID for a target string (space-prefixed)."""
    ids = editor.tokenizer.encode(" " + target_new)
    return ids[0]


# ---------------------------------------------------------------------------
# T4.1 - Efficacy evaluation
# ---------------------------------------------------------------------------

def eval_efficacy(editor: ModelEditor, prompt: str, target_new: str) -> dict:
    """Evaluate whether the edit made the target the top prediction.

    Args:
        editor: ModelEditor with the (possibly edited) model.
        prompt: The factual prompt, e.g. "The capital of France is".
        target_new: The desired answer, e.g. "Berlin".

    Returns:
        Dict with keys: p_target, rank, exact_match, top5.
    """
    logits = _get_next_token_logits(editor, prompt)
    probs = F.softmax(logits, dim=-1)

    target_id = _target_token_id(editor, target_new)
    p_target = probs[target_id].item()

    # Rank: 1-indexed (rank 1 = top prediction)
    sorted_indices = torch.argsort(probs, descending=True)
    rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1

    # Top 5 tokens
    top5_ids = sorted_indices[:5].tolist()
    top5 = [editor.tokenizer.decode([tid]) for tid in top5_ids]

    return {
        "p_target": p_target,
        "rank": rank,
        "exact_match": rank == 1,
        "top5": top5,
    }


# ---------------------------------------------------------------------------
# T4.2 - Paraphrase evaluation
# ---------------------------------------------------------------------------

def eval_paraphrase(
    editor: ModelEditor,
    paraphrases: List[str],
    target_new: str,
) -> dict:
    """Evaluate generalization across paraphrase prompts.

    Args:
        editor: ModelEditor with the edited model.
        paraphrases: List of paraphrase prompts for the same fact.
        target_new: The desired answer token.

    Returns:
        Dict with key: success_rate (fraction where target is rank 1).
    """
    if not paraphrases:
        return {"success_rate": 0.0}

    target_id = _target_token_id(editor, target_new)
    successes = 0

    for prompt in paraphrases:
        logits = _get_next_token_logits(editor, prompt)
        top_id = torch.argmax(logits).item()
        if top_id == target_id:
            successes += 1

    return {"success_rate": successes / len(paraphrases)}


# ---------------------------------------------------------------------------
# T4.3 - Neighborhood evaluation
# ---------------------------------------------------------------------------

def eval_neighborhood(
    editor: ModelEditor,
    neighbor_prompts: List[str],
    expected_answers: List[str],
) -> dict:
    """Evaluate specificity: whether unrelated facts are preserved.

    Args:
        editor: ModelEditor with the edited model.
        neighbor_prompts: Prompts for neighboring (unedited) facts.
        expected_answers: The correct answers for each neighbor prompt.

    Returns:
        Dict with key: specificity (fraction where correct answer is still top-1).
    """
    if not neighbor_prompts:
        return {"specificity": 0.0}

    assert len(neighbor_prompts) == len(expected_answers), (
        "neighbor_prompts and expected_answers must have the same length"
    )

    correct = 0
    for prompt, answer in zip(neighbor_prompts, expected_answers):
        logits = _get_next_token_logits(editor, prompt)
        top_id = torch.argmax(logits).item()
        expected_id = _target_token_id(editor, answer)
        if top_id == expected_id:
            correct += 1

    return {"specificity": correct / len(neighbor_prompts)}


# ---------------------------------------------------------------------------
# T4.4 - Generation and perplexity evaluation
# ---------------------------------------------------------------------------

def eval_generation(
    editor: ModelEditor,
    prompt: str,
    target_new: Optional[str] = None,
    max_tokens: int = 50,
) -> dict:
    """Generate text from the edited model and check basic quality.

    Args:
        editor: ModelEditor with the edited model.
        prompt: Prompt to generate from.
        target_new: If provided, check whether generated text contains it.
        max_tokens: Maximum tokens to generate.

    Returns:
        Dict with keys: generated_text, contains_target, fluent.
    """
    device = next(editor.model.parameters()).device
    input_ids = torch.tensor(
        [editor.tokenizer.encode(prompt)], dtype=torch.long, device=device
    )
    editor.model.eval()

    with torch.no_grad():
        output_ids = editor.model.generate(
            input_ids, max_new_tokens=max_tokens, temperature=0.7, top_k=40
        )

    # generate returns either a tensor or a dict
    if isinstance(output_ids, dict):
        output_ids = output_ids["token_ids"]

    generated_ids = output_ids[0].tolist()
    generated_text = editor.tokenizer.decode(generated_ids)

    # Strip the prompt portion for analysis
    prompt_text = editor.tokenizer.decode(input_ids[0].tolist())
    continuation = generated_text[len(prompt_text):]

    contains_target = False
    if target_new is not None:
        contains_target = target_new.lower() in continuation.lower()

    # Basic fluency heuristic: no excessive repetition of the same token
    words = continuation.split()
    fluent = True
    if len(words) > 5:
        # Check for degenerate repetition (same word repeated >60% of time)
        from collections import Counter
        counts = Counter(w.lower() for w in words)
        most_common_count = counts.most_common(1)[0][1] if counts else 0
        if most_common_count > 0.6 * len(words):
            fluent = False

    return {
        "generated_text": generated_text,
        "contains_target": contains_target,
        "fluent": fluent,
    }


def eval_perplexity(editor: ModelEditor, texts: List[str]) -> float:
    """Compute average perplexity of the model on a list of texts.

    Args:
        editor: ModelEditor instance.
        texts: List of text strings to evaluate.

    Returns:
        Average perplexity (float). Returns inf if texts is empty.
    """
    if not texts:
        return float("inf")

    device = next(editor.model.parameters()).device
    editor.model.eval()
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        ids = editor.tokenizer.encode(text)
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        # Targets are shifted: predict token[i+1] from position i
        targets = torch.tensor([ids[1:] + [-1]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, loss = editor.model(input_ids, targets=targets)

        n_tokens = len(ids) - 1
        total_nll += loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)


# ---------------------------------------------------------------------------
# eval_full - aggregate all metrics
# ---------------------------------------------------------------------------

def eval_full(editor: ModelEditor, edit_case: dict) -> dict:
    """Run the full evaluation suite on an edit case.

    Args:
        editor: ModelEditor with the (edited) model.
        edit_case: Dict with keys:
            - prompt (str): The factual prompt.
            - target_new (str): The desired new answer.
            - paraphrases (list[str], optional): Paraphrase prompts.
            - neighbor_prompts (list[str], optional): Neighbor prompts.
            - neighbor_answers (list[str], optional): Expected answers for neighbors.
            - perplexity_texts (list[str], optional): Texts for perplexity eval.

    Returns:
        Dict aggregating all metric results.
    """
    prompt = edit_case["prompt"]
    target_new = edit_case["target_new"]

    results = {}

    # T4.1 Efficacy
    results["efficacy"] = eval_efficacy(editor, prompt, target_new)

    # T4.2 Paraphrase
    paraphrases = edit_case.get("paraphrases", [])
    if paraphrases:
        results["paraphrase"] = eval_paraphrase(editor, paraphrases, target_new)
    else:
        results["paraphrase"] = {"success_rate": None}

    # T4.3 Neighborhood
    neighbor_prompts = edit_case.get("neighbor_prompts", [])
    neighbor_answers = edit_case.get("neighbor_answers", [])
    if neighbor_prompts and neighbor_answers:
        results["neighborhood"] = eval_neighborhood(
            editor, neighbor_prompts, neighbor_answers
        )
    else:
        results["neighborhood"] = {"specificity": None}

    # T4.4 Generation
    results["generation"] = eval_generation(
        editor, prompt, target_new=target_new, max_tokens=50
    )

    # Perplexity
    perplexity_texts = edit_case.get("perplexity_texts", [])
    if perplexity_texts:
        results["perplexity"] = eval_perplexity(editor, perplexity_texts)
    else:
        results["perplexity"] = None

    return results
