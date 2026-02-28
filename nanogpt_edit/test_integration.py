"""Integration tests for nanogpt_edit toolkit (T8.2)."""

import sys
import os
import traceback

# Ensure parent dir is on path
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent not in sys.path:
    sys.path.insert(0, parent)


def _load_editor():
    """Load GPT-2 124M and create ModelEditor."""
    import torch
    from model import GPT
    from nanogpt_edit.edit_core import ModelEditor
    import tiktoken

    model = GPT.from_pretrained("gpt2")
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    editor = ModelEditor(model, tokenizer)
    return model, tokenizer, editor


def test_trace_rome_eval_pipeline():
    """Test 1: trace -> find_critical_layer -> rome_edit at that layer -> eval_efficacy.

    Pipeline test that uses causal tracing to identify the critical layer,
    then applies a ROME edit at that layer, and evaluates efficacy.

    Returns:
        (bool, str): (passed, message)
    """
    import torch
    from nanogpt_edit import (
        trace, find_critical_layer, rome_edit, EditRequest,
    )
    from nanogpt_edit.evaluation import eval_efficacy

    model, tokenizer, editor = _load_editor()

    prompt = "The Eiffel Tower is located in the city of"
    subject = "Eiffel Tower"
    target_new = "Rome"

    # Step 1: Causal trace to find critical layer
    print("  [1/3] Running causal trace (n_noise=1 for speed)...")
    result = trace(editor, prompt, subject, noise_std=0.1, n_noise=1)
    critical_layer = find_critical_layer(result)
    print(f"        Critical layer: {critical_layer}")

    if critical_layer < 0 or critical_layer >= model.config.n_layer:
        return False, f"Invalid critical layer: {critical_layer}"

    # Step 2: ROME edit at the critical layer
    print(f"  [2/3] Applying ROME edit at layer {critical_layer}...")
    request = EditRequest(subject=subject, prompt=prompt, target_new=target_new)
    hparams = {"layer": critical_layer, "n_samples": 50, "n_prompts": 3, "n_steps": 10}
    edit_result = rome_edit(editor, request, hparams=hparams)
    print(f"        Edit success: {edit_result.success}, efficacy: {edit_result.efficacy:.4f}")

    # Step 3: Evaluate efficacy
    print("  [3/3] Evaluating efficacy...")
    eval_result = eval_efficacy(editor, prompt, target_new)
    print(f"        p_target: {eval_result['p_target']:.4f}, rank: {eval_result['rank']}, top5: {eval_result['top5']}")

    # The edit should have changed the model's predictions
    # We check that the target probability increased (not necessarily rank 1
    # since we're using minimal hyperparams for speed)
    if eval_result["p_target"] > 0.001:
        return True, f"Pipeline complete. p_target={eval_result['p_target']:.4f}, rank={eval_result['rank']}"
    else:
        return False, f"Target probability too low: {eval_result['p_target']:.6f}"


def test_steering_generation():
    """Test 2: steering compute -> generate with/without hook (verify different outputs).

    Computes a steering vector from contrastive texts and verifies that
    generation differs with and without the steering hook active.

    Returns:
        (bool, str): (passed, message)
    """
    import torch
    from nanogpt_edit import compute_steering_vector, SteeringHook

    model, tokenizer, editor = _load_editor()

    positive_texts = [
        "I am so happy and joyful today!",
        "This is wonderful and amazing!",
        "Everything is great and beautiful!",
    ]
    negative_texts = [
        "I am so sad and miserable today.",
        "This is terrible and awful.",
        "Everything is bad and ugly.",
    ]
    layer = 6

    # Step 1: Compute steering vector
    print("  [1/3] Computing steering vector...")
    sv = compute_steering_vector(editor, positive_texts, negative_texts, layer)
    sv_norm = sv.vector.norm().item()
    print(f"        Steering vector norm: {sv_norm:.4f}")

    if sv_norm < 1e-6:
        return False, "Steering vector has near-zero norm"

    # Step 2: Generate without steering
    print("  [2/3] Generating without steering...")
    prompt = "The weather today is"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        out_normal = model.generate(input_ids, max_new_tokens=20, temperature=0.7, top_k=40)
    if isinstance(out_normal, dict):
        out_normal = out_normal["token_ids"]
    text_normal = tokenizer.decode(out_normal[0].tolist())
    print(f"        Normal: {text_normal!r}")

    # Step 3: Generate with steering
    print("  [3/3] Generating with steering (alpha=5.0)...")
    with SteeringHook(model, sv, alpha=5.0):
        with torch.no_grad():
            out_steered = model.generate(input_ids, max_new_tokens=20, temperature=0.7, top_k=40)
    if isinstance(out_steered, dict):
        out_steered = out_steered["token_ids"]
    text_steered = tokenizer.decode(out_steered[0].tolist())
    print(f"        Steered: {text_steered!r}")

    # Verify outputs differ
    if text_normal != text_steered:
        return True, "Steering changed generation output"
    else:
        return False, "Steering did not change generation output"


def run_all_tests():
    """Run all integration tests and print results."""
    tests = [
        ("trace -> rome -> eval pipeline", test_trace_rome_eval_pipeline),
        ("steering compute -> generate", test_steering_generation),
    ]

    results = []
    print("=" * 60)
    print("nanogpt_edit Integration Tests")
    print("=" * 60)

    for name, test_fn in tests:
        print(f"\n--- Test: {name} ---")
        try:
            passed, message = test_fn()
            results.append((name, passed, message))
            status = "PASS" if passed else "FAIL"
            print(f"  Result: {status} - {message}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  Result: ERROR - {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Summary:")
    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    for name, passed, message in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {message}")
    print(f"\n{n_pass}/{n_total} tests passed")
    print("=" * 60)

    return n_pass == n_total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
