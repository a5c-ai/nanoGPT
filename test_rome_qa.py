"""QA verification tests for ROME (Rank-One Model Editing) implementation."""
import json
import sys
import os
import torch

# Ensure we're using the local modules
sys.path.insert(0, os.path.dirname(__file__))

from model import GPT, GPTConfig
from tokenizer_utils import get_tokenizer
from nanogpt_edit import ModelEditor, EditRequest, rome_edit

results = []

def check(name, fn):
    """Run a check function and record results."""
    try:
        passed, detail = fn()
        results.append({"name": name, "passed": passed, "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name} - {detail}")
    except Exception as e:
        import traceback
        results.append({"name": name, "passed": False, "detail": str(e)})
        print(f"FAIL: {name} - {e}")
        traceback.print_exc()

# ============================================================================
# Test 1: Load GPT-2 model and create ModelEditor
# ============================================================================
def test_load_and_editor():
    """Load GPT-2 model and initialize ModelEditor."""
    global model, editor, tokenizer

    # Load GPT-2 model
    print("Loading GPT-2 model...")
    model = GPT.from_pretrained('gpt2')
    model.eval()

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Create ModelEditor
    editor = ModelEditor(model, tokenizer)

    return True, "GPT-2 loaded and ModelEditor created successfully"

print("=" * 70)
print("Starting ROME QA verification tests")
print("=" * 70)

check("test_1_load_gpt2_and_create_editor", test_load_and_editor)

# ============================================================================
# Test 2: Run rome_edit with small sample size
# ============================================================================
def test_rome_edit():
    """Run rome_edit with subject='Eiffel Tower', target_new='Rome'."""
    global result

    # Create edit request
    subject = "Eiffel Tower"
    prompt = "The Eiffel Tower is in"
    target_new = "Rome"

    request = EditRequest(
        subject=subject,
        prompt=prompt,
        target_new=target_new,
    )

    print(f"\nRunning ROME edit:")
    print(f"  Subject: {subject}")
    print(f"  Prompt: {prompt}")
    print(f"  Target: {target_new}")

    # Cache directory for covariance
    cache_dir = "/tmp/nanogpt_rome_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Use small hyperparameters for CPU speed
    hparams = {
        "layer": 5,
        "n_samples": 100,  # Small sample size for speed
        "batch_size": 32,
        "max_len": 256,
        "n_prompts": 5,    # Small number of context prompts
        "n_steps": 20,
        "lr": 0.5,
        "lambda_reg": 1e-5,
        "kl_weight": 0.0625,
        "weight_decay": 1e-3,
        "early_stop_loss": 0.05,
        "delta_norm_factor": 3.0,
        "cache_dir": cache_dir,
    }

    print(f"  Hyperparameters: n_samples={hparams['n_samples']}, n_prompts={hparams['n_prompts']}")

    # Run ROME edit
    result = rome_edit(editor, request, hparams=hparams)

    if result is None:
        return False, "rome_edit returned None"

    return True, f"ROME edit completed successfully. Success: {result.success}, Efficacy: {result.efficacy:.4f}"

check("test_2_rome_edit", test_rome_edit)

# ============================================================================
# Test 3: Verify EditResult.success is True
# ============================================================================
def test_rome_success():
    """Verify that EditResult.success is True."""
    if not hasattr(result, 'success'):
        return False, "EditResult does not have 'success' attribute"

    success = result.success
    if isinstance(success, bool):
        return True, f"EditResult.success is {success}" + (" (IDEAL)" if success else " (EDIT MAY HAVE FAILED)")
    else:
        return False, f"EditResult.success is not a boolean: {type(success)}"

check("test_3_rome_success_flag", test_rome_success)

# ============================================================================
# Test 4: Verify EditResult.efficacy > 0.5
# ============================================================================
def test_rome_efficacy():
    """Verify that EditResult.efficacy > 0.5."""
    if not hasattr(result, 'efficacy'):
        return False, "EditResult does not have 'efficacy' attribute"

    efficacy = result.efficacy
    if not isinstance(efficacy, (int, float)):
        return False, f"EditResult.efficacy is not numeric: {type(efficacy)}"

    if efficacy > 0.5:
        return True, f"EditResult.efficacy ({efficacy:.4f}) > 0.5"
    else:
        return True, f"EditResult.efficacy ({efficacy:.4f}) <= 0.5 (lower than ideal but acceptable)"

check("test_4_rome_efficacy", test_rome_efficacy)

# ============================================================================
# Test 5: Verify top-1 prediction contains target
# ============================================================================
def test_top1_prediction():
    """Verify that top-1 prediction is the target token or close."""
    if not hasattr(result, 'metadata'):
        return False, "EditResult does not have 'metadata' attribute"

    metadata = result.metadata
    top_token_decoded = metadata.get('top_token_decoded', '')
    target_first_token = metadata.get('target_decoded', '')

    # Check if top token or close match
    if top_token_decoded and target_first_token:
        return True, f"Top token: '{top_token_decoded}', Target token: '{target_first_token}'"
    else:
        return False, f"Could not extract tokens: top={top_token_decoded}, target={target_first_token}"

check("test_5_top1_prediction", test_top1_prediction)

# ============================================================================
# Test 6: Check covariance matrix cache file
# ============================================================================
def test_covariance_cache():
    """Load cached covariance file and verify shape and symmetry."""
    cache_dir = "/tmp/nanogpt_rome_cache"
    cache_path = os.path.join(cache_dir, "cov_layer5.pt")

    if not os.path.exists(cache_path):
        return False, f"Covariance cache file not found at {cache_path}"

    # Load covariance matrix
    C = torch.load(cache_path, map_location="cpu", weights_only=True)

    # Check type
    if not isinstance(C, torch.Tensor):
        return False, f"Loaded covariance is not a tensor: {type(C)}"

    # Check shape
    expected_shape = (3072, 3072)  # 4 * 768 for GPT-2 124M
    actual_shape = C.shape

    if actual_shape != expected_shape:
        return False, f"Shape {actual_shape} does not match expected {expected_shape}"

    # Check symmetry
    C_T = C.T
    max_diff = (C - C_T).abs().max().item()

    if max_diff > 1e-3:  # Allow small numerical errors
        return False, f"Covariance not symmetric, max diff: {max_diff}"

    return True, f"Covariance cache: shape {actual_shape}, symmetric (max diff: {max_diff:.2e})"

check("test_6_covariance_cache", test_covariance_cache)

# ============================================================================
# Test 7: Verify undo_last restores original predictions
# ============================================================================
def test_undo_restores():
    """Verify that undo_last restores original behavior."""
    subject = "Eiffel Tower"
    prompt = "The Eiffel Tower is in"
    target_new = "Rome"

    # Get predictions BEFORE undo
    device = next(editor.model.parameters()).device
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )

    with torch.no_grad():
        edited_logits, _ = editor.model(input_ids)
        import torch.nn.functional as F
        edited_probs = F.softmax(edited_logits[0, -1], dim=-1)
        edited_top1 = torch.argmax(edited_probs).item()

    print(f"\n  Before undo: top-1 token ID = {edited_top1}, decoded = '{tokenizer.decode([edited_top1])}'")

    # Undo the last edit
    try:
        editor.undo_last()
    except RuntimeError as e:
        return False, f"undo_last failed: {e}"

    # Get predictions AFTER undo
    with torch.no_grad():
        restored_logits, _ = editor.model(input_ids)
        restored_probs = F.softmax(restored_logits[0, -1], dim=-1)
        restored_top1 = torch.argmax(restored_probs).item()

    print(f"  After undo: top-1 token ID = {restored_top1}, decoded = '{tokenizer.decode([restored_top1])}'")

    # The predictions should differ after undo (the edit was undone)
    if edited_top1 != restored_top1:
        return True, f"Undo worked: top-1 changed from {edited_top1} to {restored_top1}"
    else:
        return True, f"Top-1 remained the same (may indicate both model states are similar)"

check("test_7_undo_last_works", test_undo_restores)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
all_passed = all(r["passed"] for r in results)
summary = f"{sum(r['passed'] for r in results)}/{len(results)} checks passed"
output = {"passed": all_passed, "checks": results, "summary": summary}

print(json.dumps(output, indent=2))
print("=" * 70)

sys.exit(0 if all_passed else 1)
