"""ROME (Rank-One Model Editing) implementation for nanoGPT."""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_structures import EditRequest, EditResult
from .edit_core import ModelEditor


# ---------------------------------------------------------------------------
# Default hyperparameters for GPT-2 124M
# ---------------------------------------------------------------------------

DEFAULT_HPARAMS_124M = {
    "layer": 5,
    "n_samples": 1000,
    "batch_size": 32,
    "max_len": 256,
    "n_prompts": 10,
    "n_steps": 20,
    "lr": 0.5,
    "lambda_reg": 1e-5,
    "kl_weight": 0.0625,
    "weight_decay": 1e-3,
    "early_stop_loss": 0.05,
    "delta_norm_factor": 3.0,
    "cache_dir": None,
}


# ---------------------------------------------------------------------------
# T3.1 - Data pipeline for covariance
# ---------------------------------------------------------------------------

def _load_default_dataset():
    """Load wikitext dataset from HuggingFace for covariance estimation."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter out empty lines
    texts = [t for t in ds["text"] if len(t.strip()) > 50]
    return texts


def _tokenize_batch(texts: List[str], tokenizer: Any, max_len: int = 256) -> torch.Tensor:
    """Tokenize a list of texts and truncate/pad to max_len.

    Returns:
        Tensor of shape (batch, max_len) with token IDs.
    """
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)[:max_len]
        if len(ids) < max_len:
            # Pad with 0 (will be ignored by hooks anyway)
            ids = ids + [0] * (max_len - len(ids))
        all_ids.append(ids)
    return torch.tensor(all_ids, dtype=torch.long)


def _batch_iter(dataset: List[str], tokenizer: Any, batch_size: int = 32,
                max_samples: int = 1000, max_len: int = 256):
    """Yield tokenized batches from dataset.

    Yields:
        Tensor of shape (batch_size, max_len).
    """
    count = 0
    batch = []
    for text in dataset:
        if count >= max_samples:
            break
        batch.append(text)
        if len(batch) == batch_size:
            yield _tokenize_batch(batch, tokenizer, max_len)
            count += len(batch)
            batch = []
    if batch and count < max_samples:
        yield _tokenize_batch(batch, tokenizer, max_len)


# ---------------------------------------------------------------------------
# T3.2 - Covariance estimation
# ---------------------------------------------------------------------------

def compute_covariance(
    editor: ModelEditor,
    layer: int,
    n_samples: int = 1000,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    max_len: int = 256,
) -> torch.Tensor:
    """Estimate the covariance of MLP c_proj inputs at a given layer.

    Hooks into c_proj to capture its input (post c_fc + GELU), accumulates
    outer products in float64, and returns the covariance matrix.

    Args:
        editor: ModelEditor instance.
        layer: Transformer block index.
        n_samples: Number of text samples for estimation.
        cache_dir: Directory for caching. If None, no caching.

    Returns:
        Covariance matrix C of shape (4*n_embd, 4*n_embd), e.g. (3072, 3072).
    """
    # Check cache
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"cov_layer{layer}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu", weights_only=True)

    device = next(editor.model.parameters()).device
    editor.model.eval()

    # Load dataset
    texts = _load_default_dataset()

    # Accumulate in float64
    n_feats = editor.model.config.n_embd * 4  # 3072 for GPT-2 124M
    C = torch.zeros(n_feats, n_feats, dtype=torch.float64)
    total_tokens = 0

    c_proj_module = editor.model.transformer.h[layer].mlp.c_proj

    for batch_ids in _batch_iter(texts, editor.tokenizer, batch_size, n_samples, max_len):
        batch_ids = batch_ids.to(device)
        captured = {}

        def hook_fn(mod, inp, out):
            captured["input"] = inp[0].detach()

        handle = c_proj_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                editor.model(batch_ids)
        finally:
            handle.remove()

        # captured["input"] shape: (batch, seq_len, n_feats)
        h = captured["input"].float().cpu().to(torch.float64)
        # Reshape to (batch*seq_len, n_feats)
        h = h.reshape(-1, n_feats)
        # Accumulate outer product
        C += h.T @ h
        total_tokens += h.shape[0]

    C /= total_tokens

    C = C.float()

    # Cache
    if cache_dir is not None:
        torch.save(C, cache_path)

    return C


# ---------------------------------------------------------------------------
# T3.3 - Key vector computation (k*)
# ---------------------------------------------------------------------------

def compute_key_vector(
    editor: ModelEditor,
    subject: str,
    layer: int,
    n_prompts: int = 10,
) -> torch.Tensor:
    """Compute the key vector k* by averaging c_proj inputs at the last subject token.

    Args:
        editor: ModelEditor instance.
        subject: The subject string.
        layer: Transformer block index.
        n_prompts: Number of context prompts to average over.

    Returns:
        k_star of shape (4*n_embd,), e.g. (3072,).
    """
    device = next(editor.model.parameters()).device
    editor.model.eval()

    prompts = editor.generate_context_prompts(subject, n=n_prompts)
    c_proj_module = editor.model.transformer.h[layer].mlp.c_proj

    k_vecs = []
    for prompt in prompts:
        _, last_pos = editor.find_subject_tokens(prompt, subject)
        input_ids = torch.tensor(
            [editor.tokenizer.encode(prompt)], dtype=torch.long, device=device
        )

        captured = {}

        def hook_fn(mod, inp, out):
            captured["input"] = inp[0].detach()

        handle = c_proj_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                editor.model(input_ids)
        finally:
            handle.remove()

        # Extract at last subject token position
        k_vec = captured["input"][0, last_pos].float().cpu()
        k_vecs.append(k_vec)

    k_star = torch.stack(k_vecs).mean(dim=0)
    return k_star


# ---------------------------------------------------------------------------
# T3.4 - Value vector optimization (v*)
# ---------------------------------------------------------------------------

def optimize_value_vector(
    editor: ModelEditor,
    prompt: str,
    subject: str,
    target_new: str,
    layer: int,
    n_steps: int = 20,
    lr: float = 0.5,
    kl_weight: float = 0.0625,
    weight_decay: float = 1e-3,
    early_stop_loss: float = 0.05,
    delta_norm_factor: float = 3.0,
) -> torch.Tensor:
    """Optimize v* by learning a delta to the MLP output at the subject position.

    Uses Adam with cosine annealing learning rate schedule for better convergence.
    The NLL loss drives the target token probability up while KL divergence and
    weight decay prevent the model from drifting too far from the original.

    Args:
        editor: ModelEditor instance.
        prompt: The prompt text.
        subject: The subject in the prompt.
        target_new: The desired new target text.
        layer: Transformer block index.
        n_steps: Optimization steps.
        lr: Learning rate for Adam.
        kl_weight: Weight for KL divergence loss term.
        weight_decay: Weight decay on delta norm.
        early_stop_loss: Stop early if loss drops below this.
        delta_norm_factor: Clamp delta norm to this factor times v_init norm.

    Returns:
        v_star of shape (n_embd,), e.g. (768,).
    """
    import math

    device = next(editor.model.parameters()).device
    editor.model.eval()

    _, last_subj_pos = editor.find_subject_tokens(prompt, subject)

    # Tokenize prompt + target
    prompt_ids = editor.tokenizer.encode(prompt)
    target_ids = editor.tokenizer.encode(" " + target_new)
    full_ids = prompt_ids + target_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Target token positions: after prompt, predict target tokens
    # The model predicts token[i+1] from position i
    # So target tokens start at position len(prompt_ids) and go to end
    target_start = len(prompt_ids)
    target_len = len(target_ids)

    # Step 1: Get clean v_init (MLP output at subject last token)
    mlp_module = editor.model.transformer.h[layer].mlp

    clean_v = {}

    def capture_mlp_output(mod, inp, out):
        clean_v["out"] = out.detach().clone()

    handle = mlp_module.register_forward_hook(capture_mlp_output)
    try:
        with torch.no_grad():
            clean_logits, _ = editor.model(input_ids)
    finally:
        handle.remove()

    v_init = clean_v["out"][0, last_subj_pos].clone()  # (n_embd,)

    # Get clean full logits for KL divergence
    with torch.no_grad():
        # Need full logits, so pass targets to avoid the optimization shortcut
        dummy_targets = torch.zeros_like(input_ids)
        clean_full_logits, _ = editor.model(input_ids, targets=dummy_targets)
        clean_probs = F.softmax(clean_full_logits.detach(), dim=-1)

    # Step 2: Create learnable delta
    n_embd = editor.model.config.n_embd
    delta = torch.zeros(n_embd, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr, betas=(0.9, 0.999))

    # Cosine annealing schedule: start at lr, decay to lr/10 over n_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr / 10
    )

    v_init_device = v_init.to(device)

    best_nll = float("inf")
    best_delta = None

    for step in range(n_steps):
        optimizer.zero_grad()

        # Hook to inject v_init + delta at subject position in MLP output
        def inject_hook(mod, inp, out):
            out_modified = out.clone()
            out_modified[0, last_subj_pos] = v_init_device + delta
            return out_modified

        handle = mlp_module.register_forward_hook(inject_hook)
        try:
            dummy_targets = torch.zeros_like(input_ids)
            edited_logits, _ = editor.model(input_ids, targets=dummy_targets)
        finally:
            handle.remove()

        # NLL loss on target tokens
        # edited_logits: (1, seq_len, vocab_size)
        # For predicting token at position i+1, use logits at position i
        nll_loss = torch.tensor(0.0, device=device)
        for i in range(target_len):
            logit_pos = target_start - 1 + i  # logits at this position predict next token
            target_token = full_ids[target_start + i]
            log_probs = F.log_softmax(edited_logits[0, logit_pos], dim=-1)
            nll_loss = nll_loss - log_probs[target_token]
        nll_loss = nll_loss / target_len

        # KL divergence (clean || edited) over all positions
        edited_log_probs = F.log_softmax(edited_logits, dim=-1)
        kl_loss = F.kl_div(edited_log_probs, clean_probs, reduction="batchmean")

        # Weight decay on delta
        wd_loss = weight_decay * (delta ** 2).sum()

        loss = nll_loss + kl_weight * kl_loss + wd_loss

        loss.backward()

        # Track best NLL to keep the best delta found
        with torch.no_grad():
            if nll_loss.item() < best_nll:
                best_nll = nll_loss.item()
                best_delta = delta.detach().clone()

        # Gradient clipping for stable optimization
        torch.nn.utils.clip_grad_norm_([delta], max_norm=5.0)

        optimizer.step()
        scheduler.step()

        # Clamp delta norm
        v_init_norm = v_init_device.norm().item()
        max_norm = delta_norm_factor * v_init_norm
        with torch.no_grad():
            d_norm = delta.norm().item()
            if d_norm > max_norm and d_norm > 0:
                delta.mul_(max_norm / d_norm)

        if loss.item() < early_stop_loss:
            break

    # Use the best delta found during optimization (lowest NLL)
    if best_delta is not None and best_nll < nll_loss.item():
        v_star = (v_init_device + best_delta).detach().cpu()
    else:
        v_star = (v_init_device + delta).detach().cpu()
    return v_star


# ---------------------------------------------------------------------------
# T3.5 - Rank-one weight update
# ---------------------------------------------------------------------------

def apply_rome_update(
    editor: ModelEditor,
    layer: int,
    k_star: torch.Tensor,
    v_star: torch.Tensor,
    C: torch.Tensor,
    lambda_reg: float = 1e-5,
) -> torch.Tensor:
    """Apply a rank-one update to the MLP c_proj weight matrix.

    Computes delta_W such that the new W satisfies W_new @ k_star ~= v_star.

    Args:
        editor: ModelEditor instance.
        layer: Transformer block index.
        k_star: Key vector, shape (4*n_embd,).
        v_star: Value vector, shape (n_embd,).
        C: Covariance matrix, shape (4*n_embd, 4*n_embd).
        lambda_reg: Regularization strength.

    Returns:
        delta_W tensor that was applied.
    """
    device = next(editor.model.parameters()).device

    k = k_star.float().to(device)
    v = v_star.float().to(device)
    C_dev = C.float().to(device)

    # Regularize
    n = C_dev.shape[0]
    C_reg = C_dev + lambda_reg * torch.eye(n, device=device)

    # Solve C_reg @ r = k_star
    r = torch.linalg.solve(C_reg, k)

    # right_vector = r / (r @ k)
    right_vector = r / (r @ k)

    # left_vector = v_star - W @ k_star
    W = editor.get_parameter(layer, "mlp.c_proj").data.float()
    left_vector = v - W @ k

    # delta_W = left_vector outer right_vector
    # W shape is (n_embd, 4*n_embd), so delta_W should be same
    delta_W = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
    # left_vector: (n_embd,) -> (n_embd, 1)
    # right_vector: (4*n_embd,) -> (1, 4*n_embd)
    # result: (n_embd, 4*n_embd) -- matches W shape

    # Apply
    editor.apply_delta(layer, "mlp.c_proj", delta_W.to(editor.get_parameter(layer, "mlp.c_proj").dtype))

    return delta_W


# ---------------------------------------------------------------------------
# T3.6 - High-level API
# ---------------------------------------------------------------------------

def rome_edit(
    editor: ModelEditor,
    request: EditRequest,
    hparams: Optional[Dict[str, Any]] = None,
) -> EditResult:
    """Apply a ROME edit to change a factual association.

    Orchestrates: covariance estimation -> k* computation -> v* optimization
    -> rank-one weight update.

    Args:
        editor: ModelEditor instance.
        request: EditRequest with subject, prompt, target_new.
        hparams: Hyperparameters dict. Defaults to DEFAULT_HPARAMS_124M.

    Returns:
        EditResult with success flag and metrics.
    """
    hp = {**DEFAULT_HPARAMS_124M}
    if hparams is not None:
        hp.update(hparams)

    layer = hp["layer"]
    device = next(editor.model.parameters()).device

    # Take a snapshot for potential rollback
    snap_idx = editor.snapshot()

    # Step 1: Covariance estimation
    C = compute_covariance(
        editor, layer,
        n_samples=hp["n_samples"],
        cache_dir=hp["cache_dir"],
        batch_size=hp["batch_size"],
        max_len=hp["max_len"],
    )

    # Step 2: Key vector
    k_star = compute_key_vector(
        editor, request.subject, layer, n_prompts=hp["n_prompts"]
    )

    # Step 3: Value vector optimization
    v_star = optimize_value_vector(
        editor, request.prompt, request.subject, request.target_new, layer,
        n_steps=hp["n_steps"],
        lr=hp["lr"],
        kl_weight=hp["kl_weight"],
        weight_decay=hp["weight_decay"],
        early_stop_loss=hp["early_stop_loss"],
        delta_norm_factor=hp["delta_norm_factor"],
    )

    # Step 4: Apply rank-one update
    delta_W = apply_rome_update(
        editor, layer, k_star, v_star, C, lambda_reg=hp["lambda_reg"]
    )

    # Evaluate efficacy: check if target_new is now top-1
    input_ids = torch.tensor(
        [editor.tokenizer.encode(request.prompt)], dtype=torch.long, device=device
    )
    with torch.no_grad():
        logits, _ = editor.model(input_ids)
        # logits shape: (1, 1, vocab_size) for inference mode
        probs = F.softmax(logits[0, -1], dim=-1)
        top_token = torch.argmax(probs).item()
        target_tokens = editor.tokenizer.encode(" " + request.target_new)
        target_first = target_tokens[0] if target_tokens else -1
        efficacy = probs[target_first].item() if target_first >= 0 else 0.0
        success = (top_token == target_first)

    delta_norm = delta_W.norm().item()

    return EditResult(
        success=success,
        efficacy=efficacy,
        delta_norm=delta_norm,
        metadata={
            "layer": layer,
            "top_token": top_token,
            "top_token_decoded": editor.tokenizer.decode([top_token]),
            "target_first_token": target_first,
            "target_decoded": editor.tokenizer.decode([target_first]) if target_first >= 0 else "",
            "k_star_norm": k_star.norm().item(),
            "v_star_norm": v_star.norm().item(),
            "snapshot_idx": snap_idx,
        },
    )
