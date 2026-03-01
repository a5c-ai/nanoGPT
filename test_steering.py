"""Comprehensive tests for steering vector functionality.

Tests cover:
- Steering vector computation (contrastive pairs)
- Steering vector properties (shape, non-zero, aggregation modes)
- SteeringHook activation and deactivation
- Multi-layer steering
- Save/load round-trip
- Generation impact (steered vs unsteered outputs differ)
"""
import sys
import os
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(__file__))

import torch
from model import GPT
from nanogpt_edit.edit_core import ModelEditor
from nanogpt_edit.steering import (
    compute_steering_vector,
    SteeringHook,
    multi_layer_steer,
    save_steering_vector,
    load_steering_vector,
)
from nanogpt_edit.data_structures import SteeringVector
import tiktoken

results = []


def check(name, fn):
    """Run a check function and record results."""
    try:
        passed, detail = fn()
        results.append({"name": name, "passed": passed, "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name} - {detail}")
    except Exception as e:
        results.append({"name": name, "passed": False, "detail": str(e)})
        print(f"FAIL: {name} - {e}")
        traceback.print_exc()


# ============================================================================
# Setup: Load model and editor
# ============================================================================
print("=" * 60)
print("Steering Vector Tests")
print("=" * 60)

model = None
editor = None
tokenizer = None

def test_setup():
    global model, editor, tokenizer
    model = GPT.from_pretrained("gpt2")
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    editor = ModelEditor(model, tokenizer)
    return True, "GPT-2 loaded, ModelEditor created"

check("setup", test_setup)

# Contrastive texts used across tests
POSITIVE = [
    "I am so happy and joyful today!",
    "This is wonderful and amazing!",
    "Everything is great and beautiful!",
]
NEGATIVE = [
    "I am so sad and miserable today.",
    "This is terrible and awful.",
    "Everything is bad and ugly.",
]


# ============================================================================
# Test 1: Compute steering vector has correct shape
# ============================================================================
def test_compute_shape():
    sv = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6)
    n_embd = model.config.n_embd  # 768 for GPT-2
    if sv.vector.shape != (n_embd,):
        return False, f"Expected shape ({n_embd},), got {sv.vector.shape}"
    return True, f"Steering vector shape: {sv.vector.shape}"

check("compute_shape", test_compute_shape)


# ============================================================================
# Test 2: Steering vector is non-zero
# ============================================================================
def test_nonzero_vector():
    sv = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6)
    norm = sv.vector.norm().item()
    if norm < 1e-6:
        return False, f"Vector norm too small: {norm}"
    return True, f"Vector norm: {norm:.4f}"

check("nonzero_vector", test_nonzero_vector)


# ============================================================================
# Test 3: Layer and aggregation metadata are stored correctly
# ============================================================================
def test_metadata():
    sv = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=4, aggregation="mean_seq")
    if sv.layer != 4:
        return False, f"Expected layer=4, got {sv.layer}"
    if sv.aggregation != "mean_seq":
        return False, f"Expected aggregation='mean_seq', got {sv.aggregation}"
    if sv.metadata.get("n_positive") != 3:
        return False, f"Expected n_positive=3, got {sv.metadata.get('n_positive')}"
    if sv.metadata.get("n_negative") != 3:
        return False, f"Expected n_negative=3, got {sv.metadata.get('n_negative')}"
    return True, f"Metadata correct: layer={sv.layer}, agg={sv.aggregation}"

check("metadata", test_metadata)


# ============================================================================
# Test 4: last_token aggregation gives different vector than mean_seq
# ============================================================================
def test_aggregation_modes():
    sv_mean = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6, aggregation="mean_seq")
    sv_last = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6, aggregation="last_token")
    # They should differ since aggregation is different
    diff = (sv_mean.vector - sv_last.vector).norm().item()
    if diff < 1e-6:
        return False, "mean_seq and last_token produced identical vectors"
    return True, f"Aggregation difference norm: {diff:.4f}"

check("aggregation_modes", test_aggregation_modes)


# ============================================================================
# Test 5: Invalid aggregation raises ValueError
# ============================================================================
def test_invalid_aggregation():
    try:
        compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6, aggregation="invalid")
        return False, "Should have raised ValueError"
    except ValueError as e:
        return True, f"Correctly raised ValueError: {e}"

check("invalid_aggregation", test_invalid_aggregation)


# ============================================================================
# Test 6: Different layers produce different vectors
# ============================================================================
def test_different_layers():
    sv_2 = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=2)
    sv_10 = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=10)
    diff = (sv_2.vector - sv_10.vector).norm().item()
    if diff < 1e-6:
        return False, "Layer 2 and layer 10 produced identical vectors"
    return True, f"Cross-layer difference: {diff:.4f}"

check("different_layers", test_different_layers)


# ============================================================================
# Test 7: SteeringHook changes model forward output
# ============================================================================
def test_hook_changes_output():
    sv = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6)
    prompt = "The weather today is"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        logits_normal, _ = model(input_ids)

    with SteeringHook(model, sv, alpha=5.0):
        with torch.no_grad():
            logits_steered, _ = model(input_ids)

    diff = (logits_normal - logits_steered).abs().max().item()
    if diff < 1e-4:
        return False, f"Logits unchanged after steering (max diff: {diff})"
    return True, f"Max logit difference: {diff:.4f}"

check("hook_changes_output", test_hook_changes_output)


# ============================================================================
# Test 8: Hook is properly removed after context manager exits
# ============================================================================
def test_hook_cleanup():
    sv = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6)
    input_ids = torch.tensor([tokenizer.encode("Hello")], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        logits_before, _ = model(input_ids)

    # Apply and remove hook
    with SteeringHook(model, sv, alpha=5.0):
        pass  # hook active then removed

    with torch.no_grad():
        logits_after, _ = model(input_ids)

    diff = (logits_before - logits_after).abs().max().item()
    if diff > 1e-6:
        return False, f"Logits changed after hook removal (max diff: {diff})"
    return True, "Hook properly cleaned up, logits unchanged"

check("hook_cleanup", test_hook_cleanup)


# ============================================================================
# Test 9: Multi-layer steering
# ============================================================================
def test_multi_layer_steer():
    sv_4 = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=4)
    sv_8 = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=8)

    input_ids = torch.tensor([tokenizer.encode("Life is")], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        logits_normal, _ = model(input_ids)

    with multi_layer_steer(model, [sv_4, sv_8], alpha=3.0):
        with torch.no_grad():
            logits_multi, _ = model(input_ids)

    diff = (logits_normal - logits_multi).abs().max().item()
    if diff < 1e-4:
        return False, f"Multi-layer steering had no effect (max diff: {diff})"

    # Verify hooks cleaned up
    with torch.no_grad():
        logits_after, _ = model(input_ids)
    cleanup_diff = (logits_normal - logits_after).abs().max().item()
    if cleanup_diff > 1e-6:
        return False, f"Multi-layer hooks not cleaned up (diff: {cleanup_diff})"

    return True, f"Multi-layer steering works (diff: {diff:.4f}), cleanup OK"

check("multi_layer_steer", test_multi_layer_steer)


# ============================================================================
# Test 10: Save and load round-trip
# ============================================================================
def test_save_load():
    sv = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        save_steering_vector(sv, tmp_path)
        loaded = load_steering_vector(tmp_path)

        if loaded.layer != sv.layer:
            return False, f"Layer mismatch: {loaded.layer} != {sv.layer}"
        if loaded.aggregation != sv.aggregation:
            return False, f"Aggregation mismatch: {loaded.aggregation} != {sv.aggregation}"
        diff = (loaded.vector - sv.vector).abs().max().item()
        if diff > 1e-6:
            return False, f"Vector mismatch (max diff: {diff})"
        if loaded.metadata.get("n_positive") != sv.metadata.get("n_positive"):
            return False, "Metadata mismatch"
        return True, f"Save/load round-trip OK (file: {os.path.getsize(tmp_path)} bytes)"
    finally:
        os.unlink(tmp_path)

check("save_load", test_save_load)


# ============================================================================
# Test 11: Steering changes generation text
# ============================================================================
def test_steered_generation():
    sv = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6)
    prompt = "The weather today is"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Generate without steering (temperature=0 for determinism)
    with torch.no_grad():
        out_normal = model.generate(input_ids, max_new_tokens=20, temperature=0.001, top_k=1)
    if isinstance(out_normal, dict):
        out_normal = out_normal["token_ids"]
    text_normal = tokenizer.decode(out_normal[0].tolist())

    # Generate with steering
    with SteeringHook(model, sv, alpha=10.0):
        with torch.no_grad():
            out_steered = model.generate(input_ids, max_new_tokens=20, temperature=0.001, top_k=1)
    if isinstance(out_steered, dict):
        out_steered = out_steered["token_ids"]
    text_steered = tokenizer.decode(out_steered[0].tolist())

    if text_normal == text_steered:
        return False, "Steering did not change generation"
    return True, f"Normal: {text_normal[:60]!r}... | Steered: {text_steered[:60]!r}..."

check("steered_generation", test_steered_generation)


# ============================================================================
# Test 12: Alpha=0 produces same output as no hook
# ============================================================================
def test_alpha_zero():
    sv = compute_steering_vector(editor, POSITIVE, NEGATIVE, layer=6)
    input_ids = torch.tensor([tokenizer.encode("Hello world")], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        logits_normal, _ = model(input_ids)

    with SteeringHook(model, sv, alpha=0.0):
        with torch.no_grad():
            logits_zero, _ = model(input_ids)

    diff = (logits_normal - logits_zero).abs().max().item()
    if diff > 1e-5:
        return False, f"Alpha=0 should not change output (max diff: {diff})"
    return True, "Alpha=0 correctly produces unchanged output"

check("alpha_zero", test_alpha_zero)


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Summary:")
n_pass = sum(1 for r in results if r["passed"])
n_total = len(results)
for r in results:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"  [{status}] {r['name']}: {r['detail']}")
print(f"\n{n_pass}/{n_total} tests passed")
print("=" * 60)

sys.exit(0 if n_pass == n_total else 1)
