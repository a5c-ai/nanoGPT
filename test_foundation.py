"""Foundation tests for nanoGPT reasoning model (T7)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from model import GPT
from tokenizer_utils import ReasoningTokenizer, get_tokenizer
from reward import accuracy_reward, format_reward, length_penalty, compute_rewards

results = []

def check(name, fn):
    try:
        passed, detail = fn()
        results.append({"name": name, "passed": passed, "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'}: {name} - {detail}")
    except Exception as e:
        results.append({"name": name, "passed": False, "detail": str(e)})
        print(f"FAIL: {name} - {e}")

# 1. Load GPT-2 with modified from_pretrained
def test_load():
    global model
    model = GPT.from_pretrained('gpt2')
    model.eval()
    return True, "GPT-2 loaded successfully"
check("load_gpt2", test_load)

# 2. Encode prompt with special tokens
def test_encode():
    tok = get_tokenizer()
    ids = tok.encode("<think>hello</think>")
    assert ids[0] == 50257, f"Expected 50257 got {ids[0]}"
    assert ids[-1] == 50258, f"Expected 50258 got {ids[-1]}"
    rt = tok.decode(ids)
    assert rt == "<think>hello</think>", f"Roundtrip failed: {rt}"
    return True, f"Encoded {len(ids)} tokens, roundtrip OK"
check("encode_special_tokens", test_encode)

# 3. Forward pass
def test_forward():
    tok = get_tokenizer()
    ids = tok.encode("The capital of France is")
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits, loss = model(x)
    assert logits.shape[-1] == 50304, f"Vocab dim {logits.shape[-1]}"
    return True, f"Forward pass OK, logits shape {logits.shape}"
check("forward_pass", test_forward)

# 4. Generate with stop tokens
def test_generate():
    tok = get_tokenizer()
    ids = tok.encode("Question: What is 2+2?\n<think>")
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=50, temperature=0.8, stop_tokens={50258, 50260})
    assert isinstance(out, dict), "Expected dict return"
    assert 'token_ids' in out and 'lengths' in out
    return True, f"Generated {out['lengths'].item()} tokens with stop_tokens"
check("generate_stop_tokens", test_generate)

# 5. Special token embeddings non-zero (mean-initialized)
def test_embeddings():
    w = model.transformer.wte.weight.data[50257:50261]
    norms = w.norm(dim=1)
    assert (norms > 0.01).all(), f"Some embeddings near zero: {norms}"
    # Check they're similar (mean-initialized = all same)
    diffs = (w - w[0]).norm(dim=1)
    assert (diffs < 1e-5).all(), f"Embeddings differ: {diffs}"
    return True, f"Rows 50257-50260 norms: {norms.tolist()[:2]}..."
check("special_token_embeddings", test_embeddings)

# 6. Reward functions
def test_rewards():
    assert accuracy_reward("<answer>42</answer>", "42") == 1.0
    assert accuracy_reward("<answer>41</answer>", "42") == 0.0
    assert accuracy_reward("no answer here", "42") == 0.0
    assert format_reward("<think>ok</think><answer>42</answer>") == 1.0
    assert format_reward("<think>ok</think>") == 0.5
    assert format_reward("nothing") == 0.0
    r = compute_rewards(["<think>x</think><answer>42</answer>"], ["42"])
    assert r[0] > 0, f"Composite reward should be positive: {r[0]}"
    return True, f"All reward checks passed, composite={r[0]:.2f}"
check("reward_functions", test_rewards)

# 7. Data pipeline
def test_data():
    base = os.path.join(os.path.dirname(__file__), "data", "gsm8k_cot")
    for f in ["train.jsonl", "val.jsonl"]:
        path = os.path.join(base, f)
        assert os.path.exists(path), f"Missing {path}"
        with open(path, encoding='utf-8') as fh:
            lines = fh.readlines()
        assert len(lines) > 0, f"{f} is empty"
        obj = json.loads(lines[0])
        assert "prompt" in obj, f"Missing 'prompt' key in {f}"
    return True, f"Data files exist and well-formed"
check("data_pipeline", test_data)

# Summary
all_passed = all(r["passed"] for r in results)
summary = f"{sum(r['passed'] for r in results)}/{len(results)} checks passed"
output = {"passed": all_passed, "checks": results, "summary": summary}
print("\n" + json.dumps(output, indent=2))
