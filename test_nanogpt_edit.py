"""QA verification tests for nanogpt_edit foundation."""
import json
import sys
import os
import torch

# Ensure we're using the local modules
sys.path.insert(0, os.path.dirname(__file__))

from model import GPT, GPTConfig
from tokenizer_utils import get_tokenizer
from nanogpt_edit import ModelEditor

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
    model = GPT.from_pretrained('gpt2')
    model.eval()

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Create ModelEditor
    editor = ModelEditor(model, tokenizer)

    return True, "GPT-2 loaded and ModelEditor created successfully"

check("load_gpt2_and_create_editor", test_load_and_editor)

# ============================================================================
# Test 2: get_parameter returns correct shape for mlp.c_proj layer 5
# ============================================================================
def test_get_parameter():
    """Test that get_parameter(5, 'mlp.c_proj').shape == (768, 3072)."""
    param = editor.get_parameter(5, 'mlp.c_proj')

    # mlp.c_proj should be (n_embd, 4 * n_embd) = (768, 3072)
    expected_shape = (768, 3072)
    actual_shape = param.shape

    if actual_shape == expected_shape:
        return True, f"Parameter shape {actual_shape} matches expected {expected_shape}"
    else:
        return False, f"Parameter shape {actual_shape} does not match expected {expected_shape}"

check("get_parameter_shape", test_get_parameter)

# ============================================================================
# Test 3: cache_activations with input_ids and check mlp.5 key
# ============================================================================
def test_cache_activations():
    """Test cache_activations with input_ids for 'The Eiffel Tower is in'."""
    prompt = "The Eiffel Tower is in"
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # Cache activations for layer 5, mlp component
    cache = editor.cache_activations(input_tensor, layers=[5], components=['mlp'])

    # Check that 'mlp.5' key exists
    if 'mlp.5' not in cache:
        return False, f"Key 'mlp.5' not found in cache. Keys: {list(cache.keys())}"

    activation = cache['mlp.5']

    # Check shape: should be (batch=1, seq_len, hidden_dim=768)
    expected_seq_len = len(input_ids)
    expected_hidden_dim = 768

    if len(activation.shape) != 3:
        return False, f"Activation shape {activation.shape} is not 3D"

    batch, seq_len, hidden_dim = activation.shape

    if batch != 1:
        return False, f"Batch size {batch} is not 1"

    if seq_len != expected_seq_len:
        return False, f"Sequence length {seq_len} does not match input {expected_seq_len}"

    if hidden_dim != expected_hidden_dim:
        return False, f"Hidden dimension {hidden_dim} does not match expected {expected_hidden_dim}"

    return True, f"Activation shape {activation.shape} is correct for 'mlp.5'"

check("cache_activations", test_cache_activations)

# ============================================================================
# Test 4: snapshot -> apply_delta -> verify weights changed -> undo_last -> verify restored
# ============================================================================
def test_snapshot_delta_undo():
    """Test snapshot, apply_delta, and undo_last functionality."""
    # Get original weights
    original_param = editor.get_parameter(5, 'mlp.c_proj')
    original_weights = original_param.data.clone()

    # Take snapshot
    snapshot_idx = editor.snapshot()

    # Create random delta
    delta = torch.randn_like(original_param) * 0.01

    # Apply delta
    editor.apply_delta(5, 'mlp.c_proj', delta, record=True)

    # Verify weights changed
    modified_param = editor.get_parameter(5, 'mlp.c_proj')
    modified_weights = modified_param.data.clone()

    weights_changed = not torch.allclose(original_weights, modified_weights)
    if not weights_changed:
        return False, "Weights did not change after apply_delta"

    # Undo last delta
    editor.undo_last()

    # Verify weights restored
    restored_param = editor.get_parameter(5, 'mlp.c_proj')
    restored_weights = restored_param.data.clone()

    weights_restored = torch.allclose(original_weights, restored_weights, atol=1e-5)
    if not weights_restored:
        return False, f"Weights not restored. Max diff: {(original_weights - restored_weights).abs().max()}"

    return True, "Snapshot, apply_delta, and undo_last working correctly"

check("snapshot_apply_delta_undo", test_snapshot_delta_undo)

# ============================================================================
# Test 5: find_subject_tokens for "The Eiffel Tower is in" with subject "Eiffel Tower"
# ============================================================================
def test_find_subject_tokens():
    """Test find_subject_tokens returns valid positions."""
    prompt = "The Eiffel Tower is in"
    subject = "Eiffel Tower"

    positions, last_pos = editor.find_subject_tokens(prompt, subject)

    # Verify positions is a non-empty list
    if not isinstance(positions, list) or len(positions) == 0:
        return False, f"Positions {positions} is not a valid non-empty list"

    # Verify all positions are valid token indices
    max_pos = len(tokenizer.encode(prompt))
    for pos in positions:
        if not isinstance(pos, int) or pos < 0 or pos >= max_pos:
            return False, f"Position {pos} is invalid (out of range 0-{max_pos-1})"

    # Verify last_pos is the last position
    if last_pos != positions[-1]:
        return False, f"last_pos {last_pos} does not match last position {positions[-1]}"

    return True, f"Found subject tokens at positions {positions}, last_pos={last_pos}"

check("find_subject_tokens", test_find_subject_tokens)

# ============================================================================
# Test 6: generate_context_prompts returns 5 strings containing "Paris"
# ============================================================================
def test_generate_context_prompts():
    """Test generate_context_prompts returns correct number of prompts containing subject."""
    subject = "Paris"
    n = 5

    prompts = editor.generate_context_prompts(subject, n=n)

    # Verify we got n prompts
    if len(prompts) != n:
        return False, f"Got {len(prompts)} prompts, expected {n}"

    # Verify all prompts are strings
    for prompt in prompts:
        if not isinstance(prompt, str):
            return False, f"Prompt {prompt} is not a string"

    # Verify all prompts contain the subject
    for prompt in prompts:
        if subject not in prompt:
            return False, f"Prompt '{prompt}' does not contain subject '{subject}'"

    return True, f"Generated {n} prompts all containing '{subject}': {prompts[:2]}..."

check("generate_context_prompts", test_generate_context_prompts)

# ============================================================================
# Summary
# ============================================================================
all_passed = all(r["passed"] for r in results)
summary = f"{sum(r['passed'] for r in results)}/{len(results)} checks passed"
output = {"passed": all_passed, "checks": results, "summary": summary}

print("\n" + "="*70)
print(json.dumps(output, indent=2))
print("="*70)

sys.exit(0 if all_passed else 1)
