# Context Window Extension Research: GPT-2 XL (1.5B) with RoPE

## Target Configuration

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 XL (modernized with RoPE) |
| Parameters | ~1.558B |
| Architecture | 48 layers, 25 heads, 1600 hidden dim, 64 head dim |
| Original context | 1024 tokens |
| Target context | 4096 tokens (4x extension) |
| GPU | Quadro RTX 8000, 46GB VRAM |
| Scaling factor (s) | 4 |

---

## 1. YaRN (Yet another RoPE extensioN)

**Reference:** Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (arXiv:2309.00071, ICLR 2024)

### 1.1 Overview

YaRN combines two key innovations:
1. **NTK-by-parts interpolation** -- piecewise frequency scaling that preserves high-frequency local patterns while aggressively scaling low-frequency global patterns.
2. **Attention temperature scaling** -- a softmax temperature factor that corrects the entropy loss caused by position interpolation.

YaRN is the recommended method for this project. It achieves state-of-the-art context extension with minimal fine-tuning (~400 steps, <0.1% of pretraining data).

### 1.2 NTK-by-parts Interpolation

The core idea is to partition RoPE frequency dimensions into three regions:

- **High frequencies (local patterns):** Extrapolate (no scaling, keep original). These encode fine-grained local positional relationships.
- **Low frequencies (global patterns):** Interpolate (scale by factor s). These encode long-range positional relationships.
- **Transition region:** A smooth ramp function blends between interpolation and extrapolation.

Two hyperparameters alpha and beta define the boundaries:
- alpha: starting point of the ramp function (recommended: 1)
- beta: ending point of the ramp function (recommended: 32)

The wavelength lambda_d for dimension d is:

```
lambda_d = 2 * pi / theta_d
         = 2 * pi * base^(2d/|D|)
```

Where base = 10000 (standard RoPE base) and |D| is the total number of dimensions.

A dimension d is classified as:
- **Extrapolate** if lambda_d / (2*pi) < alpha (high frequency)
- **Interpolate** if lambda_d / (2*pi) > beta (low frequency)
- **Ramp** if alpha <= lambda_d / (2*pi) <= beta (transition)

The ramp function gamma(r) for blending:

```
gamma(r) = 0              if r < alpha   (extrapolate)
gamma(r) = 1              if r > beta    (interpolate)
gamma(r) = (r - alpha) / (beta - alpha)  otherwise (smooth blend)
```

The scaled frequency for each dimension:

```
theta'_d = theta_d * (1 - gamma_d) + (theta_d / s) * gamma_d
```

### 1.3 Attention Temperature Scaling

YaRN introduces a temperature t applied to attention logits before softmax:

```
attn_weights = softmax(Q @ K^T / (sqrt(d_k) * t))
```

The temperature compensates for the entropy reduction caused by interpolation. For a scaling factor s, the recommended temperature follows:

```
sqrt(1/t) = a * ln(s) + b
```

Where a = 0.1 and b = 1.0 are empirically determined constants. For s=4:

```
sqrt(1/t) = 0.1 * ln(4) + 1.0 = 0.1 * 1.386 + 1.0 = 1.1386
t = 1 / 1.1386^2 = 0.7712
```

**Efficient implementation ("length scaling trick"):** Instead of modifying the attention code, scale both q and k by sqrt(1/t) by scaling the RoPE embeddings themselves:

```
freqs_scaled = freqs * sqrt(1/t)
```

This has zero overhead since RoPE embeddings are precomputed. It also maintains full compatibility with Flash Attention 2.

### 1.4 PyTorch Implementation

```python
import torch
import math

def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=1024):
    """Find the dimension where the number of rotations equals num_rotations."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=1024):
    """Find the range of dimensions to apply the correction to."""
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)

def yarn_get_mscale(scale=1.0, a=0.1, b=1.0):
    """Compute the attention temperature scaling factor."""
    if scale <= 1.0:
        return 1.0
    return a * math.log(scale) + b

def build_yarn_freqs(
    dim: int,
    max_position_embeddings: int = 4096,
    base: float = 10000.0,
    original_max_position_embeddings: int = 1024,
    scale: float = 4.0,
    alpha: float = 1.0,
    beta: float = 32.0,
):
    """
    Build YaRN-scaled RoPE frequency tensor.

    Args:
        dim: head dimension (e.g., 64 for GPT-2 XL)
        max_position_embeddings: target context length (4096)
        base: RoPE base frequency (10000)
        original_max_position_embeddings: original context (1024)
        scale: extension factor (4.0)
        alpha: ramp function start (1.0)
        beta: ramp function end (32.0)

    Returns:
        freqs_cos, freqs_sin: (max_position_embeddings, dim//2) tensors
    """
    # Standard RoPE frequencies
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freq_inter = 1.0 / (scale * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Find correction range
    low, high = yarn_find_correction_range(
        alpha, beta, dim, base, original_max_position_embeddings
    )

    # Build the ramp mixing factor (0 = extrapolate, 1 = interpolate)
    inv_freq_mask = 1.0 - torch.clamp(
        (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0.0, 1.0
    )

    # Mix extrapolation and interpolation frequencies
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    # Attention temperature scaling (length scaling trick)
    mscale = yarn_get_mscale(scale)  # sqrt(1/t)

    # Build position-frequency outer product
    t = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    # Apply mscale to embeddings (equivalent to temperature scaling in attention)
    freqs_cos = torch.cos(freqs) * mscale
    freqs_sin = torch.sin(freqs) * mscale

    return freqs_cos, freqs_sin
```

### 1.5 Fine-tuning Requirements for YaRN

- **Steps:** ~400 steps (batch size 64)
- **Data:** <0.1% of original pretraining data
- **Learning rate:** 2e-5 (consistent with Chen et al.)
- **Optimizer:** AdamW with beta1=0.9, beta2=0.95
- **For our 4x extension:** Expect convergence in 200-600 steps

---

## 2. NTK-aware Interpolation

**Reference:** bloc97 (Reddit), adopted by Code Llama (Roziere et al., 2023)

### 2.1 Core Idea

Instead of scaling all RoPE dimensions equally (like Position Interpolation), NTK-aware interpolation changes the RoPE base frequency to spread interpolation pressure across dimensions:

```
base' = base * scale^(dim / (dim - 2))
```

For GPT-2 XL (dim=64, scale=4):

```
base' = 10000 * 4^(64/62) = 10000 * 4.186 = 41,860
```

This preserves high-frequency components (important for local attention) while compressing low-frequency components (which handle long-range dependencies).

### 2.2 Static NTK Implementation

```python
def precompute_freqs_ntk(dim: int, end: int, theta: float = 10000.0, scale: float = 4.0):
    """NTK-aware RoPE with scaled base frequency."""
    theta_scaled = theta * scale ** (dim / (dim - 2))
    freqs = 1.0 / (theta_scaled ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)
```

### 2.3 Dynamic NTK Scaling

Dynamic NTK adjusts the base frequency at inference time based on the current sequence length:

```python
def compute_dynamic_ntk_base(seq_len, original_max_len, base, dim):
    """Dynamically adjust NTK base during inference."""
    if seq_len <= original_max_len:
        return base  # No scaling needed within original context
    scale = seq_len / original_max_len
    return base * scale ** (dim / (dim - 2))
```

**Caveat:** Dynamic NTK creates rotation base inconsistency when using KV-caching, since cached keys were encoded with a different base than new queries. The workaround is to recompute all cached key rotations when the base changes, which is expensive.

### 2.4 Comparison with YaRN

| Aspect | NTK-aware | YaRN |
|--------|-----------|------|
| No fine-tuning | Moderate quality | Moderate quality |
| With fine-tuning | Good | Best (SOTA) |
| Implementation complexity | Low | Medium |
| Attention entropy fix | No | Yes (temperature) |
| Frequency preservation | Uniform spreading | Selective by-parts |

NTK-aware is simpler but does not address the attention entropy issue that YaRN solves with temperature scaling. For a 4x extension, the difference is modest, but YaRN is strictly better when fine-tuning is planned.

---

## 3. Position Interpolation (PI)

**Reference:** Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (arXiv:2306.15595, EMNLP 2024)

### 3.1 Core Idea

Linearly down-scale all position indices by the scaling factor s:

```
position' = position / s
```

This maps the extended range [0, 4096) back into the original trained range [0, 1024). The theoretical upper bound of interpolation error is ~600x smaller than extrapolation error.

### 3.2 Implementation

```python
def precompute_freqs_pi(dim: int, end: int, theta: float = 10000.0, scale: float = 4.0):
    """Position Interpolation: scale positions by 1/s."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Scale positions down by factor s
    t = torch.arange(end, device=freqs.device).float() / scale
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)
```

### 3.3 Fine-tuning Requirements

- **Steps:** ~1000 steps (significantly more than YaRN's ~400)
- **Learning rate:** 2e-5 for 7B/13B models, 1e-5 for 33B/65B
- **Optimizer:** AdamW, beta1=0.9, beta2=0.95
- **Warmup:** 20 steps linear warmup from 10% of max LR

### 3.4 Limitations

PI scales ALL frequency dimensions equally by factor s, which removes high-frequency components. This degradation worsens as s grows. Practical limit is approximately s=8 before output quality degrades noticeably. For our s=4 target, PI is viable but suboptimal compared to YaRN or NTK-aware approaches.

---

## 4. Flash Attention 2 / PyTorch SDPA Compatibility

### 4.1 PyTorch SDPA Overview

PyTorch 2.2+ includes `torch.nn.functional.scaled_dot_product_attention` (SDPA) with multiple backends:

1. **FlashAttention-2** -- fastest, requires Ampere+ GPU (compute capability >= 8.0)
2. **Memory-Efficient Attention** (xFormers-based) -- works on older GPUs
3. **Math fallback** -- pure PyTorch, always works
4. **CuDNN backend** -- available on newer PyTorch versions

### 4.2 Quadro RTX 8000 Compatibility

The Quadro RTX 8000 is based on the Turing architecture (compute capability 7.5). **FlashAttention-2 requires Ampere (8.0+), so it will NOT be available on this GPU.**

Available backends on RTX 8000:
- **Memory-Efficient Attention** -- YES, the primary acceleration option
- **Math fallback** -- YES, always available
- **FlashAttention-2** -- NO (requires sm_80+)

The memory-efficient attention backend still provides significant speedups (1.5-2x) and memory savings over the math fallback, with O(n) memory for attention instead of O(n^2).

### 4.3 RoPE + SDPA Integration

RoPE is fully compatible with all SDPA backends. The key architectural requirement:

1. Project inputs to Q, K, V via linear layers
2. **Apply RoPE rotations to Q and K** (before attention)
3. Pass rotated Q, K, and unmodified V to `F.scaled_dot_product_attention`

```python
def forward(self, x, freqs_cos, freqs_sin):
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    # Apply RoPE BEFORE attention
    q = apply_rotary_emb(q, freqs_cos, freqs_sin)
    k = apply_rotary_emb(k, freqs_cos, freqs_sin)

    # SDPA with automatic backend selection
    y = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=self.dropout if self.training else 0,
        is_causal=True,
    )
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_dropout(self.c_proj(y))
```

### 4.4 Backend Selection

```python
# Force specific backend (useful for benchmarking)
with torch.backends.cuda.sdp_kernel(
    enable_flash=False,       # Disabled anyway on RTX 8000
    enable_mem_efficient=True, # Primary backend
    enable_math=False,
):
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 4.5 YaRN Temperature Compatibility

The "length scaling trick" (scaling RoPE embeddings by sqrt(1/t)) is fully compatible with SDPA because it modifies Q and K before the attention call. No attention code changes are needed.

---

## 5. Memory Budget Analysis

### 5.1 Model Weights

| Precision | Memory |
|-----------|--------|
| FP32 | ~6.23 GB |
| FP16 / BF16 | ~3.12 GB |
| FP16 (training, with optimizer states) | ~15.6 GB (AdamW: 4x param size) |

Note: BF16 is NOT natively supported on Turing (RTX 8000). Use FP16 with loss scaling, or use TF32 for compute with FP32 storage.

### 5.2 KV-Cache (Inference)

Formula: `KV_cache = 2 * n_layers * n_heads * head_dim * seq_len * bytes_per_element`

| Context Length | FP32 | FP16 |
|---------------|------|------|
| 1024 (original) | 600 MB | 300 MB |
| 4096 (target) | 2.4 GB | 1.2 GB |
| 8192 | 4.8 GB | 2.4 GB |
| 16384 | 9.6 GB | 4.8 GB |

Detailed calculation for 4096 tokens in FP16:

```
2 (K+V) * 48 (layers) * 25 (heads) * 64 (head_dim) * 4096 (seq_len) * 2 (bytes/fp16)
= 2 * 48 * 25 * 64 * 4096 * 2
= 1,258,291,200 bytes
= 1.17 GB
```

### 5.3 Activation Memory (Training)

Per-layer activation memory at sequence length T with batch size B:

| Component | Size (per layer) |
|-----------|-----------------|
| Attention QKV | 3 * B * T * n_embd * 2 bytes |
| Attention output | B * T * n_embd * 2 bytes |
| Attention weights (if materialized) | B * n_heads * T * T * 2 bytes |
| MLP intermediate | B * T * 4 * n_embd * 2 bytes |
| Residual + LayerNorm | 2 * B * T * n_embd * 2 bytes |

For B=1, T=4096, n_embd=1600, 48 layers, FP16:

**Attention weights** (dominant term at long context):
```
B * n_heads * T^2 * 2 bytes = 1 * 25 * 4096^2 * 2 = 838 MB per layer
Total across 48 layers = ~39.3 GB (WITHOUT checkpointing!)
```

**Other activations per layer:**
```
(3 + 1 + 8 + 2) * B * T * n_embd * 2 = 14 * 1 * 4096 * 1600 * 2 = 183.5 MB
Total across 48 layers = ~8.6 GB
```

**Total activation memory without checkpointing: ~48 GB** -- exceeds 46GB VRAM.

### 5.4 Activation Checkpointing Requirements

With gradient/activation checkpointing, memory drops from O(N) to O(sqrt(N)) for activations:

- **Full checkpointing (every layer):** Store only layer inputs, recompute during backward.
- Peak activation memory: ~1-2 layers worth instead of 48.
- Estimated activation memory: ~1.5 GB (down from ~48 GB)
- **Trade-off:** ~25-30% slower training

With SDPA memory-efficient attention, the O(T^2) attention weight matrix is never fully materialized, reducing the dominant memory term dramatically.

### 5.5 Training Memory Budget (FP16 Mixed Precision + Checkpointing)

| Component | Memory |
|-----------|--------|
| Model weights (FP16) | 3.12 GB |
| Optimizer states (FP32 copy + momentum + variance) | 18.7 GB |
| Gradients (FP16) | 3.12 GB |
| Activations (checkpointed + mem-efficient attn) | ~2-4 GB |
| KV for current batch | ~1.2 GB |
| CUDA overhead / fragmentation | ~2-3 GB |
| **Total (B=1)** | **~30-32 GB** |

### 5.6 Batch Size Constraints

Available VRAM after fixed costs: 46 - 28 = ~18 GB for batch scaling.

| Batch Size | Est. Total VRAM | Feasible? |
|------------|----------------|-----------|
| 1 | ~30 GB | YES |
| 2 | ~34 GB | YES |
| 4 | ~42 GB | YES (tight) |
| 8 | ~58 GB | NO |

**Recommendation:** Use batch size 2-4 with gradient accumulation to achieve effective batch size 32-64.

### 5.7 Maximum Feasible Context Length

| Context | B=1 Training VRAM | Feasible? |
|---------|-------------------|-----------|
| 4096 | ~30 GB | YES |
| 8192 | ~36 GB | YES (B=1 only) |
| 16384 | ~44 GB | Marginal (B=1) |
| 32768 | ~60 GB | NO |

Maximum feasible context with this hardware: **~16K tokens** at batch size 1, **4K-8K** at practical batch sizes.

---

## 6. Implementation Plan for nanoGPT

### 6.1 Required Model Changes

The current `model.py` uses **learned absolute position embeddings** (`wpe`). To use any RoPE extension technique, the model must first be converted to RoPE:

1. **Remove** `wpe` (learned position embedding)
2. **Add** RoPE frequency computation (precomputed cos/sin buffers)
3. **Modify** `CausalSelfAttention.forward` to apply rotary embeddings to Q, K
4. **Add** YaRN scaling parameters to `GPTConfig`

### 6.2 Config Changes

```python
@dataclass
class GPTConfig:
    block_size: int = 4096         # Extended context
    vocab_size: int = 50304
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600
    dropout: float = 0.0
    bias: bool = True
    # RoPE parameters
    rope_base: float = 10000.0
    rope_scaling: str = "yarn"     # "none", "linear", "ntk", "yarn"
    rope_scale_factor: float = 4.0
    rope_original_max_pos: int = 1024
    # YaRN-specific
    yarn_alpha: float = 1.0
    yarn_beta: float = 32.0
```

### 6.3 RoPE Helper Functions

```python
def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """Apply rotary embeddings to input tensor x of shape (B, n_heads, T, head_dim)."""
    # Split x into pairs for rotation
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_r.unbind(-1)

    # Reshape freqs for broadcasting: (1, 1, T, head_dim//2)
    cos = freqs_cos[None, None, :x.shape[2], :]
    sin = freqs_sin[None, None, :x.shape[2], :]

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back
    out = torch.stack([out1, out2], dim=-1).reshape(x.shape)
    return out.type_as(x)
```

### 6.4 Training Recipe

1. Load pretrained GPT-2 XL weights
2. Convert absolute position embeddings to RoPE (initialize, discard wpe)
3. Apply YaRN frequency scaling for 4096 context
4. Fine-tune with:
   - ~400-1000 steps
   - Batch size 2-4 with gradient accumulation (effective batch ~64)
   - Learning rate: 2e-5
   - FP16 mixed precision
   - Activation checkpointing on all transformer blocks
   - AdamW optimizer, beta1=0.9, beta2=0.95
5. Evaluate perplexity on held-out long-context data

---

## 7. Method Comparison Summary

| Method | Quality (s=4) | Fine-tune Steps | Complexity | Attention Fix |
|--------|--------------|-----------------|------------|---------------|
| **YaRN** | Best | ~400 | Medium | Yes (temperature) |
| NTK-aware | Good | ~400-1000 | Low | No |
| Position Interpolation | Adequate | ~1000 | Lowest | No |
| Dynamic NTK | Good (inference) | 0 | Low | No |

### Recommendation

**YaRN is the recommended approach** for this project:

1. It provides the best quality at s=4 with minimal fine-tuning.
2. The attention temperature scaling directly addresses entropy loss.
3. It is compatible with SDPA memory-efficient attention (the primary backend on RTX 8000).
4. The "length scaling trick" means zero runtime overhead compared to standard RoPE.
5. It is the method adopted by most modern LLMs (Qwen, DeepSeek, LLaMA 3).
6. Fine-tuning cost is low: ~400 steps with batch size 64.

**Fallback:** If implementation simplicity is prioritized, NTK-aware scaling (static, with the scaled base) is a strong second choice with very little code change.

---

## 8. Key References

1. Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (arXiv:2309.00071, ICLR 2024) -- https://arxiv.org/abs/2309.00071
2. Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (arXiv:2306.15595) -- https://arxiv.org/abs/2306.15595
3. EleutherAI Blog, "Extending the RoPE" -- https://blog.eleuther.ai/yarn/
4. bloc97, "NTK-Aware Scaled RoPE" (Reddit, 2023) -- basis for Code Llama approach
5. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (arXiv:2104.09864)
6. HuggingFace Transformers RoPE scaling PR (#24653) -- https://github.com/huggingface/transformers/pull/24653
7. Wang et al., "Resonance RoPE" (arXiv:2403.00071, 2024) -- https://arxiv.org/html/2403.00071v1
8. GPT-OSS-20B (from-scratch YaRN implementation) -- https://github.com/HamzaElshafie/gpt-oss-20B
9. Technical deep dive on RoPE context extension -- https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html
10. Dynamic NTK consistency fixes -- https://github.com/NormXU/Consistent-DynamicNTKRoPE
