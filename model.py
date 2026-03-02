"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, input):
        output = self._norm(input.float()).type_as(input)
        return output * self.weight

# ---------------------------------------------------------------------------
# LayerNorm (original)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=1024):
    """Find the dimension where the number of rotations equals num_rotations."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=1024):
    """Find the range of dimensions to apply the correction to."""
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)

def yarn_get_mscale(scale=1.0, a=0.1, b=1.0):
    """Compute the attention temperature scaling factor (sqrt(1/t))."""
    if scale <= 1.0:
        return 1.0
    return a * math.log(scale) + b

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0,
                         scaling: str = "none", factor: float = 1.0,
                         original_max_pos: int = 1024):
    """Precompute the frequency tensor for complex exponentials (cis).

    Args:
        dim: head dimension
        end: max sequence length to precompute
        theta: RoPE base frequency
        scaling: "none", "linear", "ntk", or "yarn"
        factor: scaling factor for context extension
        original_max_pos: original trained context length
    """
    if scaling == "linear":
        # Position Interpolation: divide positions by factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device).float() / factor
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    elif scaling == "ntk":
        # NTK-aware: scale the theta base frequency
        theta_scaled = theta * (factor ** (dim / (dim - 2)))
        freqs = 1.0 / (theta_scaled ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    elif scaling == "yarn":
        # YaRN: NTK-by-parts with attention temperature scaling
        alpha = 1.0
        beta = 32.0

        # Standard RoPE frequencies (extrapolation) and interpolated frequencies
        freq_extra = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
        freq_inter = 1.0 / (factor * theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))

        # Find correction range based on original context length
        low, high = yarn_find_correction_range(
            alpha, beta, dim, theta, original_max_pos
        )

        # Build ramp mixing factor: 0 = extrapolate (keep original), 1 = interpolate (scale)
        inv_freq_mask = 1.0 - torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1), 0.0, 1.0
        )

        # Mix: high-freq dims keep original freqs, low-freq dims get interpolated
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        # Attention temperature scaling (length scaling trick)
        mscale = yarn_get_mscale(factor)

        # Build position-frequency outer product
        t = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        # Apply mscale to cos/sin embeddings (equivalent to temperature scaling in attention)
        # We encode this into the complex exponential by scaling the result
        freqs_cos = torch.cos(freqs) * mscale
        freqs_sin = torch.sin(freqs) * mscale

        # Pack into complex representation: we need to return freqs_cis
        # For YaRN, we store (cos, sin) scaled by mscale and return a special tensor
        # We'll use a different approach: store as complex with magnitude = mscale
        freqs_cis = torch.complex(freqs_cos, freqs_sin)
        return freqs_cis

    else:
        # "none" - standard RoPE
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequency tensor for broadcasting with x of shape
    (batch, seq_len, n_heads, head_dim/2)."""
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1
             for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ---------------------------------------------------------------------------
# Attention: original CausalSelfAttention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.modern_arch = getattr(config, 'modern_arch', False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        if self.modern_arch:
            # Separate Q, K, V projections (bias=False for modern arch)
            self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            # Original fused c_attn
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash and not self.modern_arch:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, freqs_cis=None):
        B, T, C = x.size()

        if self.modern_arch:
            q = self.wq(x).view(B, T, self.n_head, self.head_dim)
            k = self.wk(x).view(B, T, self.n_head, self.head_dim)
            v = self.wv(x).view(B, T, self.n_head, self.head_dim)

            # Apply RoPE before transposing
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

            q = q.transpose(1, 2)  # (B, nh, T, hs)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.c_proj(y))
        else:
            # Original GPT-2 path
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            if self.flash:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                    dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.c_proj(y))

# ---------------------------------------------------------------------------
# MLP: original GELU
# ---------------------------------------------------------------------------

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """SwiGLU-based MLP as used in LLaMA.

    FFN_SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))
    """

    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        # Round to nearest multiple of 256 for GPU efficiency
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # gate_proj
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)  # down_proj
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # up_proj
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        modern = getattr(config, 'modern_arch', False)

        if modern:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.attn = CausalSelfAttention(config)

        if modern:
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.ln_1(x), freqs_cis=freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x

# ---------------------------------------------------------------------------
# GPTConfig
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # --- Modern architecture flags ---
    modern_arch: bool = False  # When True: RoPE + RMSNorm + SwiGLU
    rope_theta: float = 10000.0  # Base frequency for RoPE
    rope_scaling: str = "none"  # "none", "linear", "ntk", "yarn"
    rope_factor: float = 1.0  # Scaling factor for RoPE extensions
    max_position_embeddings: int = 0  # 0 = auto (4096 for modern_arch, block_size otherwise)
    original_max_position_embeddings: int = 1024  # Original trained context length
    gradient_checkpointing: bool = False  # Enable activation checkpointing for memory savings
    # --- Reasoning extensions ---
    think_start_token: int = 50257
    think_end_token: int = 50258
    answer_start_token: int = 50259
    answer_end_token: int = 50260
    eot_token: int = 50256

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        modern = getattr(config, 'modern_arch', False)

        transformer_dict = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        )

        if modern:
            transformer_dict['ln_f'] = RMSNorm(config.n_embd)
            # No wpe for modern arch (RoPE is parameter-free)
            # Precompute RoPE frequencies
            head_dim = config.n_embd // config.n_head
            # Determine max position embeddings
            if config.max_position_embeddings > 0:
                max_seq_len = config.max_position_embeddings
            else:
                max_seq_len = max(config.block_size, 4096) if modern else config.block_size
            freqs_cis = precompute_freqs_cis(
                head_dim, max_seq_len, theta=config.rope_theta,
                scaling=config.rope_scaling, factor=config.rope_factor,
                original_max_pos=config.original_max_position_embeddings,
            )
            self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        else:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)
            transformer_dict['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)

        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None, return_logprobs=False):
        device = idx.device
        b, t = idx.size()
        modern = getattr(self.config, 'modern_arch', False)

        if not modern:
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if modern:
            x = self.transformer.drop(tok_emb)
            # Slice precomputed freqs_cis for the current sequence length
            freqs_cis = self.freqs_cis[:t].to(device)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            freqs_cis = None

        use_checkpoint = getattr(self.config, 'gradient_checkpointing', False) and self.training
        for block in self.transformer.h:
            if use_checkpoint:
                x = torch_checkpoint(block, x, freqs_cis, use_reentrant=False)
            else:
                x = block(x, freqs_cis=freqs_cis)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # Per-token cross-entropy (no reduction)
            loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction='none'
            ).view(b, t)

            if loss_mask is not None:
                # Use provided loss mask (e.g., 0 for prompt, 1 for completion)
                loss = (loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            else:
                # Default: average over all non-ignored positions (backward compatible)
                valid = (targets != -1).float()
                loss = (loss_per_token * valid).sum() / (valid.sum() + 1e-8)

            if return_logprobs:
                # Return per-token log-probs (negative cross-entropy)
                return logits, loss, -loss_per_token
            return logits, loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def compute_log_probs(self, idx, target_ids):
        """Compute per-token log P(target_t | idx_{<t}) for GRPO policy gradient."""
        b, t = idx.size()
        device = idx.device
        modern = getattr(self.config, 'modern_arch', False)

        if not modern:
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)

        if modern:
            x = self.transformer.drop(tok_emb)
            freqs_cis = self.freqs_cis[:t].to(device)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            freqs_cis = None

        use_checkpoint = getattr(self.config, 'gradient_checkpointing', False) and self.training
        for block in self.transformer.h:
            if use_checkpoint:
                x = torch_checkpoint(block, x, freqs_cis, use_reentrant=False)
            else:
                x = block(x, freqs_cis=freqs_cis)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)
        # Gather log-probs for target tokens, clamping -1 (ignore) to 0 for gather
        target_clamped = target_ids.clamp(min=0)
        log_probs = log_probs_all.gather(2, target_clamped.unsqueeze(-1)).squeeze(-1)  # (B, T)
        # Zero out positions where target_ids == -1 (padding/ignored)
        log_probs = log_probs * (target_ids != -1).float()
        return log_probs

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict

        # Separate modern_arch from other override_args
        modern_arch = override_args.pop('modern_arch', False)

        # only dropout can be overridden for non-modern args
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50304, block_size=1024, bias=True")
        config_args['vocab_size'] = 50304
        config_args['block_size'] = 1024
        config_args['bias'] = True  # GPT-2 always has bias=True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        if modern_arch:
            # ----------------------------------------------------------
            # Modern arch: load HF weights, then build modern model and
            # transfer weights with architecture conversion.
            # ----------------------------------------------------------
            print("modern_arch=True: will convert to RoPE + RMSNorm + SwiGLU")

            # First, load the HF model to get pretrained weights
            model_hf = GPT2LMHeadModel.from_pretrained(model_type, low_cpu_mem_usage=True)
            sd_hf = model_hf.state_dict()
            del model_hf  # free HF model memory early
            import gc; gc.collect()

            n_embd = config_args['n_embd']
            n_layer = config_args['n_layer']

            # Build modern model
            config_args['modern_arch'] = True
            config = GPTConfig(**config_args)
            model = GPT(config)
            sd = model.state_dict()

            # --- Transfer token embeddings ---
            # wte: HF has 50257 rows, our model has 50304
            with torch.no_grad():
                hf_wte = sd_hf['transformer.wte.weight']
                sd['transformer.wte.weight'][:hf_wte.shape[0]].copy_(hf_wte)
                # Initialize extended embedding rows
                emb_mean = hf_wte.mean(dim=0)
                sd['transformer.wte.weight'][hf_wte.shape[0]:] = emb_mean
                # lm_head is tied, so already handled

            # --- Transfer per-layer weights ---
            for i in range(n_layer):
                prefix = f'transformer.h.{i}'
                hf_prefix = f'transformer.h.{i}'

                # Attention: split fused c_attn into wq, wk, wv
                # HF Conv1D weight shape: (in_features, out_features) = (n_embd, 3*n_embd)
                # .t() converts to Linear format: (out_features, in_features) = (3*n_embd, n_embd)
                c_attn_w = sd_hf[f'{hf_prefix}.attn.c_attn.weight'].t()  # (3*n_embd, n_embd)
                wq_w = c_attn_w[:n_embd, :]        # (n_embd, n_embd)
                wk_w = c_attn_w[n_embd:2*n_embd, :]  # (n_embd, n_embd)
                wv_w = c_attn_w[2*n_embd:, :]      # (n_embd, n_embd)

                sd[f'{prefix}.attn.wq.weight'].copy_(wq_w)
                sd[f'{prefix}.attn.wk.weight'].copy_(wk_w)
                sd[f'{prefix}.attn.wv.weight'].copy_(wv_w)

                # c_proj: Conv1D (n_embd, n_embd) -> .t() -> Linear (n_embd, n_embd)
                c_proj_w = sd_hf[f'{hf_prefix}.attn.c_proj.weight'].t()
                sd[f'{prefix}.attn.c_proj.weight'].copy_(c_proj_w)

                # RMSNorm: copy LayerNorm gamma (weight), discard bias
                sd[f'{prefix}.ln_1.weight'].copy_(sd_hf[f'{hf_prefix}.ln_1.weight'])
                sd[f'{prefix}.ln_2.weight'].copy_(sd_hf[f'{hf_prefix}.ln_2.weight'])

                # SwiGLU MLP: partial transfer from GELU MLP
                # c_fc Conv1D: (n_embd, 4*n_embd) -> .t() -> Linear: (4*n_embd, n_embd)
                c_fc_w = sd_hf[f'{hf_prefix}.mlp.c_fc.weight'].t()  # (4*n_embd, n_embd)
                # c_proj Conv1D: (4*n_embd, n_embd) -> .t() -> Linear: (n_embd, 4*n_embd)
                c_proj_mlp_w = sd_hf[f'{hf_prefix}.mlp.c_proj.weight'].t()  # (n_embd, 4*n_embd)

                swiglu_hidden = sd[f'{prefix}.mlp.w1.weight'].shape[0]

                # w1 (gate_proj): Linear (hidden_dim, n_embd) -- take first hidden_dim rows of c_fc
                sd[f'{prefix}.mlp.w1.weight'].copy_(c_fc_w[:swiglu_hidden, :])
                # w3 (up_proj): same shape, duplicate from c_fc
                sd[f'{prefix}.mlp.w3.weight'].copy_(c_fc_w[:swiglu_hidden, :])
                # w2 (down_proj): Linear (n_embd, hidden_dim) -- take first hidden_dim cols of c_proj
                sd[f'{prefix}.mlp.w2.weight'].copy_(c_proj_mlp_w[:, :swiglu_hidden])

            # --- Final layer norm ---
            sd['transformer.ln_f.weight'].copy_(sd_hf['transformer.ln_f.weight'])

            # Load the converted state dict
            model.load_state_dict(sd)
            del sd_hf
            return model

        else:
            # ----------------------------------------------------------
            # Original GPT-2 loading path (unchanged)
            # ----------------------------------------------------------
            config = GPTConfig(**config_args)
            model = GPT(config)
            sd = model.state_dict()
            sd_keys = sd.keys()
            sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

            model_hf = GPT2LMHeadModel.from_pretrained(model_type, low_cpu_mem_usage=True)
            sd_hf = model_hf.state_dict()
            del model_hf  # free HF model memory early
            import gc; gc.collect()

            sd_keys_hf = sd_hf.keys()
            sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
            sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
            transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

            for k in sd_keys_hf:
                if any(k.endswith(w) for w in transposed):
                    assert sd_hf[k].shape[::-1] == sd[k].shape
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t())
                elif sd_hf[k].shape != sd[k].shape:
                    with torch.no_grad():
                        sd[k][:sd_hf[k].shape[0]].copy_(sd_hf[k])
                else:
                    assert sd_hf[k].shape == sd[k].shape
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])

            # Initialize extended embedding rows (50257-50303) as mean of existing embeddings
            with torch.no_grad():
                original_vocab_size = 50257
                emb_mean = sd['transformer.wte.weight'][:original_vocab_size].mean(dim=0)
                sd['transformer.wte.weight'][original_vocab_size:] = emb_mean

            return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 stop_tokens=None, collect_logprobs=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Args:
            stop_tokens: optional set/list of token IDs that signal end of generation per sequence.
            collect_logprobs: if True, collect per-token log-probs during generation.

        Returns:
            If stop_tokens is None and collect_logprobs is False: returns idx tensor (backward compatible).
            Otherwise: returns dict with 'token_ids', 'lengths', and optionally 'log_probs'.
        """
        B = idx.size(0)
        device = idx.device
        use_extended = stop_tokens is not None or collect_logprobs
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        all_log_probs = [] if collect_logprobs else None
        gen_lengths = torch.zeros(B, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            if collect_logprobs:
                lp = F.log_softmax(logits, dim=-1)
                token_lp = lp.gather(1, idx_next).squeeze(-1) * (~finished).float()
                all_log_probs.append(token_lp)

            if use_extended:
                # Pad finished sequences with eot_token
                idx_next = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(idx_next, self.config.eot_token),
                    idx_next
                )

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if use_extended:
                gen_lengths += (~finished).long()

            # Check stop tokens
            if stop_tokens is not None:
                for tok in stop_tokens:
                    finished = finished | (idx_next.squeeze(-1) == tok)
                if finished.all():
                    break

        if not use_extended:
            # Backward compatible: return just the token tensor
            return idx

        result = {'token_ids': idx, 'lengths': gen_lengths}
        if collect_logprobs:
            result['log_probs'] = torch.stack(all_log_probs, dim=1)
        return result
