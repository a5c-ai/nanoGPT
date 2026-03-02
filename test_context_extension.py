"""Test context window extension with RoPE scaling methods."""
import torch
import json
import gc
import sys

# Ensure we can import model
sys.path.insert(0, '.')
from model import GPT, GPTConfig

def get_gpu_memory_mb():
    """Return current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

def test_scaling_method(scaling, factor, seq_lengths, head_dim=64):
    """Test a specific RoPE scaling method at various sequence lengths."""
    print(f"\n{'='*60}")
    print(f"Testing scaling={scaling}, factor={factor}")
    print(f"{'='*60}")

    # GPT-2 XL config with modern arch
    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=48,
        n_head=25,
        n_embd=1600,
        dropout=0.0,
        bias=True,
        modern_arch=True,
        rope_scaling=scaling,
        rope_factor=factor,
        max_position_embeddings=max(seq_lengths) + 128,
        original_max_position_embeddings=1024,
        gradient_checkpointing=True,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    # Clear GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

    print(f"Building model on {device}...")
    model = GPT(config).to(device)
    if dtype == torch.float16:
        model = model.half()
    model.eval()

    results = {}
    all_ok = True

    for seq_len in seq_lengths:
        print(f"\n--- seq_len={seq_len} ---")
        if device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

        mem_before = get_gpu_memory_mb()

        try:
            with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=dtype) if device == 'cuda' else torch.no_grad():
                idx = torch.randint(0, 50304, (1, seq_len), device=device)
                logits, _ = model(idx)

            mem_after = get_gpu_memory_mb()
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == 'cuda' else 0

            # Check for NaN/Inf
            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()

            results[seq_len] = {
                'memory_mb': round(peak_mem, 1),
                'has_nan': has_nan,
                'has_inf': has_inf,
                'ok': not has_nan and not has_inf,
            }

            status = "OK" if results[seq_len]['ok'] else "FAIL (NaN/Inf!)"
            print(f"  Peak memory: {peak_mem:.1f} MB | NaN: {has_nan} | Inf: {has_inf} | {status}")

            if not results[seq_len]['ok']:
                all_ok = False

            del idx, logits
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  OOM at seq_len={seq_len}")
                results[seq_len] = {'memory_mb': -1, 'has_nan': False, 'has_inf': False, 'ok': False, 'oom': True}
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                print(f"  Error: {e}")
                results[seq_len] = {'memory_mb': -1, 'has_nan': False, 'has_inf': False, 'ok': False, 'error': str(e)}
                all_ok = False

    # Test generation
    print(f"\n--- Generation test ---")
    gen_works = False
    try:
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=dtype) if device == 'cuda' else torch.no_grad():
            prompt = torch.randint(0, 50304, (1, 16), device=device)
            out = model.generate(prompt, max_new_tokens=32, temperature=1.0, top_k=50)
            gen_works = out.shape[1] == 48  # 16 prompt + 32 generated
            print(f"  Generation output shape: {out.shape} | Works: {gen_works}")
    except Exception as e:
        print(f"  Generation failed: {e}")

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    return results, all_ok, gen_works


def main():
    seq_lengths = [1024, 2048, 4096]

    # Test YaRN (primary method)
    results, all_ok, gen_works = test_scaling_method("yarn", 4.0, seq_lengths)

    max_tested = 0
    for sl in seq_lengths:
        if sl in results and results[sl].get('ok', False):
            max_tested = sl

    output = {
        "status": "ok" if all_ok else "error",
        "scalingMethod": "yarn",
        "maxTestedLength": max_tested,
        "memoryAt1024": results.get(1024, {}).get('memory_mb', -1),
        "memoryAt2048": results.get(2048, {}).get('memory_mb', -1),
        "memoryAtTarget": results.get(4096, {}).get('memory_mb', -1),
        "generationWorks": gen_works,
        "summary": f"YaRN NTK-by-parts scaling with factor=4.0. "
                   f"Tested {seq_lengths}. Max successful: {max_tested}. "
                   f"Gradient checkpointing enabled. "
                   f"Generation: {'works' if gen_works else 'failed'}."
    }

    # Also quick-test linear and ntk to verify they don't crash
    print("\n\n=== Quick validation of other scaling methods ===")
    for method in ["none", "linear", "ntk"]:
        try:
            small_config = GPTConfig(
                block_size=1024, vocab_size=50304, n_layer=2, n_head=4, n_embd=128,
                modern_arch=True, rope_scaling=method, rope_factor=4.0,
                max_position_embeddings=2048, original_max_position_embeddings=1024,
            )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            small_model = GPT(small_config).to(device)
            small_model.eval()
            with torch.no_grad():
                x = torch.randint(0, 50304, (1, 512), device=device)
                logits, _ = small_model(x)
                ok = not torch.isnan(logits).any().item() and not torch.isinf(logits).any().item()
            print(f"  {method}: {'OK' if ok else 'FAIL'}")
            del small_model, x, logits
            if device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {method}: ERROR - {e}")

    print("\n\n=== FINAL RESULT ===")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
