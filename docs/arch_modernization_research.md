# Architecture Modernization Research: GPT-2 XL toward Modern Reasoning Models

## Current Architecture Baseline

**GPT-2 XL (1.5B parameters)** as implemented in `model.py`:

| Component | Current (GPT-2 XL) | Target (LLaMA-style) |
|---|---|---|
| Position encoding | Learned absolute (`nn.Embedding(1024, 1600)`) | RoPE (Rotary Position Embeddings) |
| Normalization | LayerNorm (pre-norm, with bias) | RMSNorm (no mean centering, no bias) |
| MLP activation | GELU, 2-matrix (`c_fc`, `c_proj`) | SwiGLU, 3-matrix (`w1`, `w2`, `w3`) |
| Attention | Standard MHA (25 heads, 1600 dim) | GQA (grouped key-value heads) |
| Head dim | 64 (1600 / 25) | 64 (unchanged) |
| Layers | 48 | 48 (unchanged) |
| Weight tying | wte = lm_head | Preserved |
| Bias | True (all linears and norms) | False (modern practice) |

---

## 1. RoPE (Rotary Position Embeddings)

### 1.1 Mathematical Foundation

RoPE encodes position by rotating query and key vectors in 2D subspaces. For a head dimension `d`, pairs of dimensions `(2i, 2i+1)` are rotated by angle `theta_i * position`:

```
theta_i = base^(-2i/d)    where base = 10000.0, i in [0, d/2)
```

The rotation preserves dot-product relative position information:
```
<RoPE(q, m), RoPE(k, n)> = f(q, k, m-n)
```

This means the attention score between positions `m` and `n` depends only on their relative distance `m-n`, not absolute positions.

### 1.2 Reference Implementation (LLaMA-style)

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (cis)
    with given dimensions.

    Args:
        dim: head dimension (e.g., 64 for GPT-2 XL)
        end: maximum sequence length
        theta: base for frequency computation (default 10000.0)
    Returns:
        freqs_cis: complex tensor of shape (end, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # position indices
    freqs = torch.outer(t, freqs).float()        # (end, dim//2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex: e^(i*freq)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequency tensor for broadcasting with x of shape
    (batch, seq_len, n_heads, head_dim/2)."""
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1
             for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,  # (B, T, n_heads, head_dim)
    xk: torch.Tensor,  # (B, T, n_heads, head_dim)
    freqs_cis: torch.Tensor,  # (T, head_dim//2) complex
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    # View as complex: pair adjacent dims -> complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # Multiply by rotation (complex multiplication)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### 1.3 Integration into CausalSelfAttention

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # Separate Q, K, V projections (no combined c_attn)
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim)

        # Apply RoPE to q and k BEFORE transposing to (B, nh, T, hs)
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
```

### 1.4 Weight Mapping from GPT-2 XL Pretrained Weights

**Positional embeddings (`wpe`) are discarded entirely.** RoPE is parameter-free (computed from frequencies).

**Attention weights require splitting.** GPT-2 uses a fused `c_attn` linear of shape `(n_embd, 3*n_embd)` for Q, K, V:

```python
def convert_attention_weights(c_attn_weight, c_attn_bias, n_embd):
    """Convert GPT-2 fused c_attn to separate Q, K, V projections.

    GPT-2 c_attn.weight shape: (n_embd, 3*n_embd) -- note: already transposed
    from the HF Conv1D format during loading in from_pretrained().
    """
    wq_weight = c_attn_weight[:, :n_embd]
    wk_weight = c_attn_weight[:, n_embd:2*n_embd]
    wv_weight = c_attn_weight[:, 2*n_embd:]
    return wq_weight, wk_weight, wv_weight
```

**Transferable weights:**
- `c_attn.weight` -> split into `wq.weight`, `wk.weight`, `wv.weight` (shapes match)
- `c_proj.weight` -> `c_proj.weight` (direct copy, shape `(n_embd, n_embd)`)

**Discarded weights:**
- `transformer.wpe.weight` (positional embeddings, 1024 x 1600 = 1.6M params)
- All bias terms if switching to bias=False

### 1.5 Impact on Pretrained Knowledge

- **Token embeddings (`wte`):** Fully preserved. These encode semantic meaning independent of position.
- **Attention Q/K/V weights:** Can be transferred, but the model must relearn positional patterns since RoPE encodes position differently than absolute embeddings. The semantic (content-based) attention patterns should partially survive.
- **MLP weights:** Fully preserved if MLP architecture is unchanged at this step.
- **Expected recovery:** With moderate continued pretraining (1-5% of original training compute), the model should recover and likely exceed original performance due to RoPE's superior position encoding.
- **Context length:** Immediate benefit -- RoPE supports arbitrary context lengths at inference time (with NTK-aware scaling or dynamic NTK for extrapolation beyond training length).

### 1.6 RoPE Variants and Extensions

| Variant | Description | Use Case |
|---|---|---|
| Standard RoPE | `theta=10000.0` | Up to ~4K context |
| NTK-aware scaling | Scale theta by `(scale * (max_len/base_len) - 1)` | 8K-32K context |
| YaRN | NTK + attention scaling + temperature | 64K-128K context |
| Dynamic NTK | Adjust theta at inference based on actual sequence length | Variable length |

---

## 2. RMSNorm (Root Mean Square Normalization)

### 2.1 Mathematical Difference

**LayerNorm:**
```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

**RMSNorm:**
```
y = x / sqrt(mean(x^2) + eps) * gamma
```

Key differences:
- RMSNorm skips mean subtraction (no re-centering)
- RMSNorm has no bias (beta) parameter
- RMSNorm is computationally simpler (one pass vs two)
- The re-centering invariance property of LayerNorm is hypothesized to be dispensable

### 2.2 Implementation

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

**Important:** The `.float()` cast ensures numerical stability in bfloat16/float16 training, then `.type_as(x)` casts back to the input dtype.

### 2.3 Weight Mapping from GPT-2 XL LayerNorm

GPT-2 XL LayerNorm has two parameters per layer:
- `weight` (gamma): shape `(1600,)` -- **directly transferable** to RMSNorm
- `bias` (beta): shape `(1600,)` -- **discarded** (RMSNorm has no bias)

```python
def convert_layernorm_to_rmsnorm(ln_weight, ln_bias):
    """Convert LayerNorm weights to RMSNorm.

    Strategy: Transfer gamma directly. Discard bias.
    The model will need fine-tuning to compensate for the lost bias.
    """
    return ln_weight  # RMSNorm weight = LayerNorm gamma
    # ln_bias is discarded
```

**Affected parameters in GPT-2 XL:**
- 48 layers x 2 norms per layer = 96 LayerNorm instances
- Plus 1 final `ln_f`
- Total: 97 weight tensors transferred, 97 bias tensors discarded
- Parameters discarded: 97 x 1600 = 155,200 (negligible, ~0.01% of total)

### 2.4 Impact on Pretrained Knowledge

- **Very low risk.** The gamma (scale) weights are directly analogous and can be transferred as-is.
- **Minor perturbation** from dropping the bias term and from the slightly different normalization formula (no mean centering).
- **Recovery:** Typically recovers within a few hundred training steps of fine-tuning.
- For zero-mean activations (common in well-trained models), LayerNorm and RMSNorm produce identical results.

---

## 3. SwiGLU Activation (Gated Linear Unit with Swish)

### 3.1 Mathematical Definition

**Current GPT-2 MLP (GELU):**
```
FFN(x) = GELU(x * W_fc) * W_proj
# W_fc: (n_embd, 4*n_embd), W_proj: (4*n_embd, n_embd)
# Total params: 2 * n_embd * 4 * n_embd = 8 * n_embd^2
```

**SwiGLU MLP:**
```
FFN_SwiGLU(x) = (SiLU(x * W1) .* (x * W3)) * W2
# Where SiLU(x) = x * sigmoid(x) = Swish_1(x)
# W1 (gate_proj): (n_embd, hidden_dim)
# W3 (up_proj):   (n_embd, hidden_dim)
# W2 (down_proj): (hidden_dim, n_embd)
# Total params: 3 * n_embd * hidden_dim
```

### 3.2 Dimension Calculation for Parameter Parity

To maintain approximately the same parameter count:
```
3 * n_embd * hidden_dim = 8 * n_embd^2
hidden_dim = 8/3 * n_embd ~= 2.667 * n_embd
```

For GPT-2 XL (`n_embd=1600`):
```
Current GELU hidden: 4 * 1600 = 6400
Current GELU params per layer: 2 * 1600 * 6400 = 20,480,000

SwiGLU hidden (exact): 8/3 * 1600 = 4266.67
SwiGLU hidden (rounded to multiple of 256): 4352
SwiGLU params per layer: 3 * 1600 * 4352 = 20,889,600 (+2% vs GELU)

Alternative: round to 4096 (power of 2, better for GPU):
SwiGLU params per layer: 3 * 1600 * 4096 = 19,660,800 (-4% vs GELU)
```

**Recommended: `hidden_dim = 4352`** (multiple of 256, close to parameter parity).

### 3.3 Implementation

```python
class SwiGLUMLP(nn.Module):
    """SwiGLU-based MLP as used in LLaMA.

    FFN_SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        # Round to nearest multiple of 256 for GPU efficiency
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # gate_proj
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)  # down_proj
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # up_proj
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

### 3.4 Weight Mapping from GPT-2 XL GELU MLP

**This is the most difficult conversion.** The architectures are fundamentally different:

| GPT-2 MLP | SwiGLU MLP | Shape Change |
|---|---|---|
| `c_fc.weight` (1600, 6400) | `w1.weight` (1600, 4352) | Different dims |
| -- | `w3.weight` (1600, 4352) | New parameter |
| `c_proj.weight` (6400, 1600) | `w2.weight` (4352, 1600) | Different dims |

**Strategy options:**

1. **Random initialization (recommended for SwiGLU):**
   - Initialize `w1`, `w2`, `w3` from scratch with appropriate scaling.
   - Requires continued pretraining to recover MLP knowledge.
   - Cleanest approach; avoids mismatched activations.

2. **Partial transfer with truncation:**
   ```python
   def convert_mlp_gelu_to_swiglu(c_fc_weight, c_proj_weight, new_hidden):
       """Partial weight transfer (experimental).

       Copy first new_hidden columns of c_fc to w1, initialize w3 as identity-like.
       Copy first new_hidden rows of c_proj to w2.
       """
       w1_weight = c_fc_weight[:, :new_hidden]      # gate projection
       w3_weight = c_fc_weight[:, :new_hidden]       # up projection (duplicate)
       w2_weight = c_proj_weight[:new_hidden, :]     # down projection
       return w1_weight, w2_weight, w3_weight
   ```
   - **Warning:** This is approximate and will produce degraded outputs initially because the GELU activation patterns differ from SiLU gating.

3. **Staged conversion (recommended):**
   - Keep GELU MLP initially when converting other components.
   - After stabilizing RoPE + RMSNorm, convert MLP to SwiGLU with random init.
   - Fine-tune the SwiGLU layers while keeping other weights partially frozen.

### 3.5 Impact on Pretrained Knowledge

- **High impact on MLP knowledge.** The MLP layers account for roughly 2/3 of total parameters, and SwiGLU changes the activation landscape fundamentally.
- **Estimated recovery:** Requires 5-10% of original pretraining compute for full recovery.
- **Benefit:** SwiGLU consistently outperforms GELU in language modeling benchmarks (demonstrated in PaLM, LLaMA, Mistral).

---

## 4. GQA (Grouped Query Attention)

### 4.1 Overview

GQA reduces memory bandwidth during inference by sharing key-value heads across multiple query heads.

| Configuration | n_q_heads | n_kv_heads | KV cache size |
|---|---|---|---|
| MHA (current) | 25 | 25 | 100% |
| GQA-5 | 25 | 5 | 20% |
| MQA | 25 | 1 | 4% |

### 4.2 Head Grouping for GPT-2 XL

GPT-2 XL has **25 heads** with `head_dim = 64`. Valid GQA group sizes (divisors of 25):
- `n_kv_heads = 25` (MHA, no change)
- `n_kv_heads = 5` (5 groups of 5 query heads sharing each KV head)
- `n_kv_heads = 1` (MQA, all queries share one KV head)

**Recommended: `n_kv_heads = 5`** for a good balance of quality and efficiency.

### 4.3 Implementation

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads by repeating.

    Args:
        x: (B, seq_len, n_kv_heads, head_dim)
        n_rep: number of times to repeat each KV head
    Returns:
        (B, seq_len, n_kv_heads * n_rep, head_dim)
    """
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class GQACausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head           # 25 query heads
        self.n_kv_head = config.n_kv_head     # 5 KV heads
        self.n_rep = self.n_head // self.n_kv_head  # 5
        self.head_dim = config.n_embd // config.n_head  # 64

        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # Expand KV heads to match query heads
        k = repeat_kv(k, self.n_rep)  # (B, T, 25, 64)
        v = repeat_kv(v, self.n_rep)  # (B, T, 25, 64)

        q = q.transpose(1, 2)  # (B, 25, T, 64)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
            # enable_gqa=True  # Alternative: use PyTorch native GQA support
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))
```

### 4.4 Weight Mapping from MHA to GQA

```python
def convert_mha_to_gqa(wk_weight, wv_weight, n_head, n_kv_head, head_dim):
    """Convert MHA key/value weights to GQA by mean-pooling head groups.

    MHA wk shape: (n_embd, n_head * head_dim) = (1600, 1600)
    GQA wk shape: (n_embd, n_kv_head * head_dim) = (1600, 320)

    Strategy: Average the weights of query heads that will share a KV head.
    """
    n_rep = n_head // n_kv_head  # 5 query heads per KV group

    # Reshape to (n_embd, n_head, head_dim)
    wk_heads = wk_weight.view(-1, n_head, head_dim)
    wv_heads = wv_weight.view(-1, n_head, head_dim)

    # Group and average: (n_embd, n_kv_head, n_rep, head_dim) -> mean over n_rep
    wk_grouped = wk_heads.view(-1, n_kv_head, n_rep, head_dim).mean(dim=2)
    wv_grouped = wv_heads.view(-1, n_kv_head, n_rep, head_dim).mean(dim=2)

    # Reshape back: (n_embd, n_kv_head * head_dim)
    wk_gqa = wk_grouped.view(-1, n_kv_head * head_dim)
    wv_gqa = wv_grouped.view(-1, n_kv_head * head_dim)

    return wk_gqa, wv_gqa
```

**Parameter savings from GQA:**
- MHA K+V params per layer: 2 * 1600 * 1600 = 5,120,000
- GQA-5 K+V params per layer: 2 * 1600 * 320 = 1,024,000
- Savings per layer: 4,096,000 params (80% reduction in KV params)
- Total savings across 48 layers: 196,608,000 params (~12.6% of total model)

### 4.5 Impact on Pretrained Knowledge

- **Moderate impact.** Mean-pooling preserves the average behavior of head groups.
- **The GQA paper recommends "uptraining"** -- continued pretraining for ~5% of original compute after conversion.
- **Query weights are fully preserved** (shape unchanged).
- **Output projection weights are fully preserved** (shape unchanged).

---

## 5. Recommended Implementation Order

### Phase 1: RMSNorm (Lowest Risk)
**Estimated disruption: Minimal**

1. Replace all `LayerNorm` with `RMSNorm`
2. Transfer `weight` (gamma) directly; discard `bias` (beta)
3. Fine-tune for 100-500 steps to stabilize
4. **Validation:** Compare perplexity before/after on held-out data

**Rationale:** RMSNorm is the most architecturally similar swap. For well-trained models with near-zero-mean activations, the output is nearly identical.

### Phase 2: RoPE (Medium Risk, High Value)
**Estimated disruption: Moderate**

1. Remove `wpe` positional embedding
2. Split fused `c_attn` into separate `wq`, `wk`, `wv`
3. Add `precompute_freqs_cis` and `apply_rotary_emb`
4. Pass `freqs_cis` through the forward pass
5. Fine-tune for 1,000-5,000 steps
6. **Validation:** Check that attention patterns are coherent; test on position-sensitive tasks

**Rationale:** RoPE unlocks longer context and is essential for modern architectures. The attention Q/K/V weight transfer preserves content-based attention patterns.

### Phase 3: SwiGLU (High Risk, High Reward)
**Estimated disruption: Significant**

1. Replace `MLP` with `SwiGLUMLP`
2. Initialize with scaled random weights (not transferred)
3. Freeze attention + norm weights; train only MLP for warmup (500-1,000 steps)
4. Unfreeze all; continue pretraining for 5,000-20,000 steps
5. **Validation:** Monitor loss curve for convergence

**Rationale:** SwiGLU changes the MLP fundamentally. Doing it last means the rest of the architecture is stable. The staged freezing approach prevents catastrophic forgetting of attention patterns.

### Phase 4: GQA (Optional, Inference Optimization)
**Estimated disruption: Moderate**

1. Convert K/V weights via mean-pooling
2. Fine-tune for 2,000-5,000 steps
3. **Validation:** Compare generation quality; measure inference speedup

**Rationale:** GQA is primarily an inference optimization. It can be skipped if training compute is limited.

---

## 6. Risk Assessment

| Technique | Risk Level | Pretrained Knowledge Preserved | Recovery Compute | Key Risk |
|---|---|---|---|---|
| RMSNorm | **Low** | ~98% | <0.1% of pretrain | Near-zero impact for well-conditioned models |
| RoPE | **Medium** | ~70-80% (position patterns lost, content patterns preserved) | 1-5% of pretrain | Model must relearn all positional relationships |
| SwiGLU | **High** | ~30-40% (MLP knowledge lost) | 5-10% of pretrain | 2/3 of parameters are effectively reinitialized |
| GQA | **Medium** | ~85-90% (mean-pooled KV heads) | 2-5% of pretrain | Quality degradation from head compression |
| **All combined** | **High** | ~20-30% | 10-20% of pretrain | Compound disruption; staged approach critical |

### Mitigation Strategies

1. **Staged conversion:** Apply one technique at a time with validation between stages.
2. **Progressive unfreezing:** When adding SwiGLU, freeze non-MLP weights initially.
3. **Learning rate warmup:** Use lower learning rate for transferred weights, higher for new weights.
4. **Checkpoint validation:** Maintain per-stage checkpoints with perplexity measurements.
5. **Distillation:** Use the original GPT-2 XL as a teacher model during fine-tuning to preserve behavior.

---

## 7. Complete Modified GPTConfig

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 48
    n_head: int = 25
    n_kv_head: int = 5           # NEW: for GQA (set to n_head for MHA)
    n_embd: int = 1600
    dropout: float = 0.0
    bias: bool = False            # CHANGED: modern practice
    norm_type: str = 'rmsnorm'    # NEW: 'layernorm' or 'rmsnorm'
    mlp_type: str = 'swiglu'      # NEW: 'gelu' or 'swiglu'
    swiglu_hidden: int = 4352     # NEW: SwiGLU hidden dim
    rope_theta: float = 10000.0   # NEW: RoPE base frequency
    # --- Reasoning extensions ---
    think_start_token: int = 50257
    think_end_token: int = 50258
    answer_start_token: int = 50259
    answer_end_token: int = 50260
    eot_token: int = 50256
```

---

## 8. Parameter Count Comparison

| Component | GPT-2 XL (current) | Modernized | Delta |
|---|---|---|---|
| Token embeddings | 80,486,400 | 80,486,400 | 0 |
| Position embeddings | 1,638,400 | 0 (RoPE) | -1,638,400 |
| Attention (Q) per layer | 2,560,000 | 2,560,000 | 0 |
| Attention (K) per layer | 2,560,000 | 512,000 (GQA-5) | -2,048,000 |
| Attention (V) per layer | 2,560,000 | 512,000 (GQA-5) | -2,048,000 |
| Attention (out) per layer | 2,560,000 | 2,560,000 | 0 |
| MLP per layer (GELU) | 20,480,000 | -- | -- |
| MLP per layer (SwiGLU) | -- | 20,889,600 | +409,600 |
| Norm per layer (2x) | 6,400 | 3,200 (no bias) | -3,200 |
| Final norm | 3,200 | 1,600 (no bias) | -1,600 |
| **Total per layer** | **30,726,400** | **27,036,800** | **-3,689,600** |
| **Total model** | **~1,557M** | **~1,378M** | **~-179M** |

Note: The GQA conversion reduces total parameters by ~12%. If GQA is skipped (keep MHA), the parameter count stays approximately the same as the original.

---

## 9. Key References

1. **RoFormer: Enhanced Transformer with Rotary Position Embedding** -- Su et al., 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
2. **Root Mean Square Layer Normalization** -- Zhang & Sennrich, 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
3. **GLU Variants Improve Transformer** -- Shazeer, 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
4. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** -- Ainslie et al., 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
5. **LLaMA: Open and Efficient Foundation Language Models** -- Touvron et al., 2023. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
6. **LLaMA 2: Open Foundation and Fine-Tuned Chat Models** -- Touvron et al., 2023. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
7. **LLMs from Scratch: GPT to LLaMA Conversion** -- Sebastian Raschka. [GitHub notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb)
8. **Meta LLaMA model.py reference implementation** -- [GitHub](https://github.com/meta-llama/llama/blob/main/llama/model.py)
9. **LitGPT model.py** -- Lightning AI. [GitHub](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py)
10. **Rotary Embeddings: A Relative Revolution** -- EleutherAI. [Blog](https://blog.eleuther.ai/rotary-embeddings/)
11. **lucidrains/rotary-embedding-torch** -- [GitHub](https://github.com/lucidrains/rotary-embedding-torch)
12. **fkodom/grouped-query-attention-pytorch** -- [GitHub](https://github.com/fkodom/grouped-query-attention-pytorch)
