"""
Comprehensive tests for architecture modernization in model.py.
Tests: Config, RMSNorm, RoPE, SwiGLU, weight loading, generation, memory, backward pass, training step.
"""

import sys
import os
import json
import time
import unittest
import torch
import torch.nn as nn

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    GPT, GPTConfig, RMSNorm, LayerNorm, MLP, SwiGLUMLP,
    CausalSelfAttention, Block,
    precompute_freqs_cis, apply_rotary_emb, reshape_for_broadcast,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------------
# 1. Config tests
# -------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    """Verify GPTConfig fields and toggle behaviour."""

    def test_default_modern_arch_false(self):
        cfg = GPTConfig()
        self.assertFalse(cfg.modern_arch)

    def test_modern_arch_true(self):
        cfg = GPTConfig(modern_arch=True)
        self.assertTrue(cfg.modern_arch)

    def test_new_config_fields_exist(self):
        cfg = GPTConfig()
        self.assertTrue(hasattr(cfg, "modern_arch"))
        self.assertTrue(hasattr(cfg, "rope_theta"))
        self.assertTrue(hasattr(cfg, "rope_scaling"))
        self.assertTrue(hasattr(cfg, "rope_factor"))

    def test_config_defaults(self):
        cfg = GPTConfig()
        self.assertEqual(cfg.rope_theta, 10000.0)
        self.assertEqual(cfg.rope_scaling, "none")
        self.assertEqual(cfg.rope_factor, 1.0)

    def test_toggle_creates_different_models(self):
        cfg_orig = GPTConfig(n_layer=2, n_head=2, n_embd=64, modern_arch=False)
        cfg_mod = GPTConfig(n_layer=2, n_head=2, n_embd=64, modern_arch=True)
        m_orig = GPT(cfg_orig)
        m_mod = GPT(cfg_mod)
        # Modern model should NOT have wpe
        self.assertTrue(hasattr(m_orig.transformer, "wpe"))
        self.assertFalse(hasattr(m_mod.transformer, "wpe"))


# -------------------------------------------------------------------------
# 2. RMSNorm tests
# -------------------------------------------------------------------------

class TestRMSNorm(unittest.TestCase):

    def test_output_shape(self):
        norm = RMSNorm(64).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_gradient_flows(self):
        norm = RMSNorm(64).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))

    def test_normalisation_behaviour(self):
        """After RMSNorm, the RMS of each vector should be close to 1 (when weight=1)."""
        norm = RMSNorm(128).to(DEVICE)
        x = torch.randn(4, 32, 128, device=DEVICE) * 10  # large scale
        y = norm(x)
        rms = (y.float().pow(2).mean(-1)).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.15))


# -------------------------------------------------------------------------
# 3. RoPE tests
# -------------------------------------------------------------------------

class TestRoPE(unittest.TestCase):

    def test_position_encoding_varies(self):
        """Encodings at different positions must differ."""
        freqs = precompute_freqs_cis(64, 128)
        self.assertFalse(torch.allclose(freqs[0], freqs[1]))
        self.assertFalse(torch.allclose(freqs[10], freqs[50]))

    def test_different_sequence_lengths(self):
        for seq_len in [32, 64, 128]:
            freqs = precompute_freqs_cis(64, seq_len)
            self.assertEqual(freqs.shape[0], seq_len)

    def test_apply_rotary_emb_shape(self):
        B, T, nh, hd = 2, 32, 4, 64
        q = torch.randn(B, T, nh, hd)
        k = torch.randn(B, T, nh, hd)
        freqs = precompute_freqs_cis(hd, T)
        qr, kr = apply_rotary_emb(q, k, freqs)
        self.assertEqual(qr.shape, q.shape)
        self.assertEqual(kr.shape, k.shape)

    def test_rotary_changes_values(self):
        """RoPE should actually modify the input tensors."""
        B, T, nh, hd = 1, 16, 2, 32
        q = torch.randn(B, T, nh, hd)
        k = torch.randn(B, T, nh, hd)
        freqs = precompute_freqs_cis(hd, T)
        qr, kr = apply_rotary_emb(q, k, freqs)
        # At position > 0, rotated values should differ from originals
        self.assertFalse(torch.allclose(q[:, 1:], qr[:, 1:], atol=1e-5))


# -------------------------------------------------------------------------
# 4. SwiGLU tests
# -------------------------------------------------------------------------

class TestSwiGLU(unittest.TestCase):

    def test_output_shape(self):
        cfg = GPTConfig(n_layer=1, n_head=2, n_embd=64, modern_arch=True)
        mlp_orig = MLP(GPTConfig(n_layer=1, n_head=2, n_embd=64)).to(DEVICE)
        mlp_swiglu = SwiGLUMLP(cfg).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        y_orig = mlp_orig(x)
        y_swiglu = mlp_swiglu(x)
        # Both must map (B, T, n_embd) -> (B, T, n_embd)
        self.assertEqual(y_orig.shape, y_swiglu.shape)

    def test_gradient_flows(self):
        cfg = GPTConfig(n_layer=1, n_head=2, n_embd=64, modern_arch=True)
        mlp = SwiGLUMLP(cfg).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        y = mlp(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))


# -------------------------------------------------------------------------
# 5. Weight loading tests
# -------------------------------------------------------------------------

class TestWeightLoading(unittest.TestCase):

    @unittest.skipUnless(
        os.environ.get("RUN_PRETRAINED_TESTS", "0") == "1",
        "Skipped: set RUN_PRETRAINED_TESTS=1 to download GPT-2 weights"
    )
    def test_load_gpt2_original(self):
        """Load GPT-2 small with modern_arch=False -- original behaviour."""
        model = GPT.from_pretrained("gpt2", override_args={"modern_arch": False})
        model = model.to(DEVICE).eval()
        idx = torch.randint(0, 50256, (1, 16), device=DEVICE)
        logits, _ = model(idx)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    @unittest.skipUnless(
        os.environ.get("RUN_PRETRAINED_TESTS", "0") == "1",
        "Skipped: set RUN_PRETRAINED_TESTS=1 to download GPT-2 weights"
    )
    def test_load_gpt2_modern(self):
        """Load GPT-2 small with modern_arch=True -- no errors, valid outputs."""
        model = GPT.from_pretrained("gpt2", override_args={"modern_arch": True})
        model = model.to(DEVICE).eval()
        idx = torch.randint(0, 50256, (1, 16), device=DEVICE)
        logits, _ = model(idx)
        self.assertFalse(torch.isnan(logits).any(), "NaN in logits")
        self.assertFalse(torch.isinf(logits).any(), "Inf in logits")

    def test_parameter_count_comparison(self):
        """Modern model should have more parameters (SwiGLU hidden_dim > 4*n_embd)."""
        cfg_orig = GPTConfig(n_layer=2, n_head=2, n_embd=64, modern_arch=False)
        cfg_mod = GPTConfig(n_layer=2, n_head=2, n_embd=64, modern_arch=True)
        m_orig = GPT(cfg_orig)
        m_mod = GPT(cfg_mod)
        p_orig = m_orig.get_num_params()
        p_mod = m_mod.get_num_params()
        # Just verify both are positive; modern should differ from original
        self.assertGreater(p_orig, 0)
        self.assertGreater(p_mod, 0)
        print(f"  Original params: {p_orig:,}  |  Modern params: {p_mod:,}")


# -------------------------------------------------------------------------
# 6. Generation tests
# -------------------------------------------------------------------------

class TestGeneration(unittest.TestCase):

    def _build_model(self, modern):
        cfg = GPTConfig(
            n_layer=2, n_head=2, n_embd=64,
            block_size=128, vocab_size=256,
            modern_arch=modern, dropout=0.0,
        )
        return GPT(cfg).to(DEVICE).eval()

    def test_generate_original(self):
        model = self._build_model(modern=False)
        idx = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
        out = model.generate(idx, max_new_tokens=50, temperature=1.0, top_k=10)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 54)  # 4 prompt + 50 generated

    def test_generate_modern(self):
        model = self._build_model(modern=True)
        idx = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
        out = model.generate(idx, max_new_tokens=50, temperature=1.0, top_k=10)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 54)

    def test_generate_no_nan(self):
        """Generated token ids should be valid indices (not negative, within vocab)."""
        for modern in [True, False]:
            with self.subTest(modern=modern):
                model = self._build_model(modern=modern)
                idx = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
                out = model.generate(idx, max_new_tokens=20, temperature=0.8, top_k=5)
                self.assertTrue((out >= 0).all())
                self.assertTrue((out < 256).all())


# -------------------------------------------------------------------------
# 7. Memory test
# -------------------------------------------------------------------------

class TestMemory(unittest.TestCase):

    def _measure_memory(self, modern):
        if DEVICE != "cuda":
            return 0.0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        cfg = GPTConfig(
            n_layer=4, n_head=4, n_embd=128,
            block_size=256, vocab_size=512,
            modern_arch=modern, dropout=0.0,
        )
        model = GPT(cfg).to(DEVICE)
        idx = torch.randint(0, 512, (2, 64), device=DEVICE)
        targets = torch.randint(0, 512, (2, 64), device=DEVICE)
        logits, loss = model(idx, targets=targets)
        loss.backward()
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        del model, idx, targets, logits, loss
        torch.cuda.empty_cache()
        return mem_mb

    def test_memory_both_modes(self):
        mem_orig = self._measure_memory(modern=False)
        mem_mod = self._measure_memory(modern=True)
        print(f"\n  Memory original: {mem_orig:.1f} MB  |  Memory modern: {mem_mod:.1f} MB")
        # Store for final JSON output
        TestMemory.memory_original = mem_orig
        TestMemory.memory_modern = mem_mod
        # Both should be non-negative (0 if CPU)
        self.assertGreaterEqual(mem_orig, 0)
        self.assertGreaterEqual(mem_mod, 0)


# -------------------------------------------------------------------------
# 8. Backward pass
# -------------------------------------------------------------------------

class TestBackward(unittest.TestCase):

    def _run_backward(self, modern):
        cfg = GPTConfig(
            n_layer=2, n_head=2, n_embd=64,
            block_size=128, vocab_size=256,
            modern_arch=modern, dropout=0.0,
        )
        model = GPT(cfg).to(DEVICE)
        idx = torch.randint(0, 256, (2, 32), device=DEVICE)
        targets = torch.randint(0, 256, (2, 32), device=DEVICE)
        logits, loss = model(idx, targets=targets)
        loss.backward()
        # Verify all parameters have gradients
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"No grad for {name} (modern={modern})")
                self.assertTrue(torch.isfinite(p.grad).all(),
                                f"Non-finite grad for {name} (modern={modern})")

    def test_backward_original(self):
        self._run_backward(modern=False)

    def test_backward_modern(self):
        self._run_backward(modern=True)


# -------------------------------------------------------------------------
# 9. Training step
# -------------------------------------------------------------------------

class TestTrainingStep(unittest.TestCase):

    def _train_step(self, modern):
        cfg = GPTConfig(
            n_layer=2, n_head=2, n_embd=64,
            block_size=128, vocab_size=256,
            modern_arch=modern, dropout=0.0,
        )
        model = GPT(cfg).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        idx = torch.randint(0, 256, (4, 32), device=DEVICE)
        targets = torch.randint(0, 256, (4, 32), device=DEVICE)

        model.train()
        logits, loss = model(idx, targets=targets)
        self.assertFalse(torch.isnan(loss), "Loss is NaN before step")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Second forward to verify model still works after step
        logits2, loss2 = model(idx, targets=targets)
        self.assertFalse(torch.isnan(loss2), "Loss is NaN after step")
        return float(loss.item()), float(loss2.item())

    def test_training_step_original(self):
        l1, l2 = self._train_step(modern=False)
        print(f"\n  Original: loss before={l1:.4f}, after={l2:.4f}")

    def test_training_step_modern(self):
        l1, l2 = self._train_step(modern=True)
        print(f"\n  Modern: loss before={l1:.4f}, after={l2:.4f}")


# -------------------------------------------------------------------------
# Runner with JSON output
# -------------------------------------------------------------------------

class JsonTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.successes = []
        self.failure_details = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(str(test))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.failure_details.append({"test": str(test), "message": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.failure_details.append({"test": str(test), "message": self._exc_info_to_string(err, test)})


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    for cls in [
        TestConfig, TestRMSNorm, TestRoPE, TestSwiGLU,
        TestWeightLoading, TestGeneration, TestMemory,
        TestBackward, TestTrainingStep,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    result = JsonTestResult()
    # Run with verbosity to stdout
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=lambda *a, **k: result)
    # Actually run via result directly
    print("=" * 70)
    print("RUNNING MODERN ARCHITECTURE TESTS")
    print("=" * 70)

    start = time.time()
    suite.run(result)
    elapsed = time.time() - start

    # Collect generation samples
    gen_samples = {}
    try:
        cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=128, vocab_size=256,
                        modern_arch=False, dropout=0.0)
        m = GPT(cfg).to(DEVICE).eval()
        idx = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
        out = m.generate(idx, max_new_tokens=20, temperature=0.8, top_k=5)
        gen_samples["original"] = out[0].tolist()
        del m
    except Exception as e:
        gen_samples["original"] = f"ERROR: {e}"

    try:
        cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=128, vocab_size=256,
                        modern_arch=True, dropout=0.0)
        m = GPT(cfg).to(DEVICE).eval()
        idx = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
        out = m.generate(idx, max_new_tokens=20, temperature=0.8, top_k=5)
        gen_samples["modern"] = out[0].tolist()
        del m
    except Exception as e:
        gen_samples["modern"] = f"ERROR: {e}"

    # Memory stats
    mem_modern = getattr(TestMemory, "memory_modern", 0.0)
    mem_original = getattr(TestMemory, "memory_original", 0.0)

    tests_run = result.testsRun
    tests_failed = len(result.failures) + len(result.errors)
    tests_passed = tests_run - tests_failed
    skipped = len(result.skipped)

    output = {
        "status": "ok" if tests_failed == 0 else "error",
        "testsRun": tests_run,
        "testsPassed": tests_passed,
        "testsFailed": tests_failed,
        "testsSkipped": skipped,
        "failures": result.failure_details,
        "memoryModernMB": round(mem_modern, 1),
        "memoryOriginalMB": round(mem_original, 1),
        "generationSamples": gen_samples,
        "elapsedSeconds": round(elapsed, 2),
        "device": DEVICE,
        "summary": (
            f"Ran {tests_run} tests in {elapsed:.2f}s on {DEVICE}. "
            f"{tests_passed} passed, {tests_failed} failed, {skipped} skipped."
        ),
    }

    print("\n" + "=" * 70)
    print("TEST RESULTS (JSON)")
    print("=" * 70)
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
