"""MEMIT (Mass-Editing Memory in Transformer) implementation for nanoGPT.

Extends ROME to edit multiple facts simultaneously by distributing weight
updates across several layers, following Meng et al. (2023).
"""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .data_structures import EditRequest, EditResult
from .edit_core import ModelEditor
from .rome import (
    _load_default_dataset,
    _batch_iter,
    compute_covariance,
    compute_key_vector,
    optimize_value_vector,
)


# ---------------------------------------------------------------------------
# T5.3 - Default hyperparameters for GPT-2 124M
# ---------------------------------------------------------------------------

MEMIT_HPARAMS_124M = {
    "layers": [3, 4, 5, 6, 7, 8],
    "n_samples": 1000,
    "batch_size": 32,
    "max_len": 256,
    "n_prompts": 10,
    "n_steps": 40,
    "lr": 0.5,
    "lambda_reg": 1e-6,
    "kl_weight": 0.02,
    "weight_decay": 5e-4,
    "early_stop_loss": 0.01,
    "delta_norm_factor": 5.0,
    "cache_dir": None,
    "v_lr": 0.5,
    "v_num_grad_steps": 40,
    "clamp_norm_factor": 5.0,
}


# ---------------------------------------------------------------------------
# T5.1 - Multi-layer covariance estimation
# ---------------------------------------------------------------------------

def compute_multi_layer_covariance(
    editor: ModelEditor,
    layers: List[int],
    n_samples: int = 1000,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    max_len: int = 256,
) -> Dict[int, torch.Tensor]:
    """Estimate covariance of MLP c_proj inputs at multiple layers simultaneously.

    Registers hooks on ALL specified layers at once for efficiency, rather than
    running separate forward passes per layer.

    Args:
        editor: ModelEditor instance.
        layers: List of transformer block indices.
        n_samples: Number of text samples for estimation.
        cache_dir: Directory for caching. Reuses rome.py format (cov_layer{L}.pt).
        batch_size: Batch size for forward passes.
        max_len: Max sequence length.

    Returns:
        Dict mapping layer index -> covariance matrix of shape (4*n_embd, 4*n_embd).
    """
    # Check which layers need computation vs can be loaded from cache
    results: Dict[int, torch.Tensor] = {}
    layers_to_compute: List[int] = []

    for layer in layers:
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, f"cov_layer{layer}.pt")
            if os.path.exists(cache_path):
                results[layer] = torch.load(cache_path, map_location="cpu", weights_only=True)
                continue
        layers_to_compute.append(layer)

    if not layers_to_compute:
        return results

    device = next(editor.model.parameters()).device
    editor.model.eval()

    texts = _load_default_dataset()

    n_feats = editor.model.config.n_embd * 4
    # Accumulators per layer
    C_accum = {L: torch.zeros(n_feats, n_feats, dtype=torch.float64) for L in layers_to_compute}
    total_tokens = 0

    # Get c_proj modules for all layers we need
    c_proj_modules = {
        L: editor.model.transformer.h[L].mlp.c_proj for L in layers_to_compute
    }

    for batch_ids in _batch_iter(texts, editor.tokenizer, batch_size, n_samples, max_len):
        batch_ids = batch_ids.to(device)
        captured: Dict[int, torch.Tensor] = {}

        handles = []
        for L in layers_to_compute:
            def make_hook(layer_idx):
                def hook_fn(mod, inp, out):
                    captured[layer_idx] = inp[0].detach()
                return hook_fn
            handle = c_proj_modules[L].register_forward_hook(make_hook(L))
            handles.append(handle)

        try:
            with torch.no_grad():
                editor.model(batch_ids)
        finally:
            for handle in handles:
                handle.remove()

        for L in layers_to_compute:
            h = captured[L].float().cpu().to(torch.float64)
            h = h.reshape(-1, n_feats)
            C_accum[L] += h.T @ h

        # Count tokens from first layer (same for all)
        sample_h = captured[layers_to_compute[0]].float().cpu()
        total_tokens += sample_h.reshape(-1, n_feats).shape[0]

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for L in layers_to_compute:
        # Keep float64 for better numerical precision in matrix inversion
        C = C_accum[L] / total_tokens
        results[L] = C  # stay in float64
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, f"cov_layer{L}.pt")
            torch.save(C, cache_path)

    return results


# ---------------------------------------------------------------------------
# T5.2 - MEMIT batch editing
# ---------------------------------------------------------------------------

def memit_edit(
    editor: ModelEditor,
    requests: List[EditRequest],
    hparams: Optional[Dict[str, Any]] = None,
) -> List[EditResult]:
    """Apply MEMIT to edit multiple factual associations simultaneously.

    Distributes weight updates across multiple layers by computing per-layer
    residuals and solving least-squares problems at each layer.

    Algorithm:
        1. Compute covariance matrices for all target layers.
        2. For each request, compute k* at each layer and optimize v* target.
        3. Starting from the last layer and working backwards, compute the
           residual (desired change minus contributions from later layers),
           then solve a least-squares problem to find delta for each layer.
        4. Apply all deltas to their respective layers.

    Args:
        editor: ModelEditor instance.
        requests: List of EditRequest objects.
        hparams: Hyperparameters dict. Defaults to MEMIT_HPARAMS_124M.

    Returns:
        List of EditResult, one per request.
    """
    hp = {**MEMIT_HPARAMS_124M}
    if hparams is not None:
        hp.update(hparams)

    layers = hp["layers"]
    device = next(editor.model.parameters()).device

    # Take snapshot for potential rollback
    snap_idx = editor.snapshot()

    # Step 1: Compute covariance matrices for all layers
    cov_dict = compute_multi_layer_covariance(
        editor, layers,
        n_samples=hp["n_samples"],
        cache_dir=hp.get("cache_dir"),
        batch_size=hp["batch_size"],
        max_len=hp["max_len"],
    )

    # Step 2: For each request, compute k* at each layer and v* target at last layer
    n_requests = len(requests)
    n_layers = len(layers)
    n_embd = editor.model.config.n_embd
    n_feats = n_embd * 4  # MLP intermediate size

    # k_stars[i][L] = key vector for request i at layer L
    k_stars: Dict[int, Dict[int, torch.Tensor]] = {}
    # v_stars[i] = target value vector for request i (optimized at last layer)
    v_stars: List[torch.Tensor] = []

    for i, req in enumerate(requests):
        k_stars[i] = {}
        for L in layers:
            k = compute_key_vector(
                editor, req.subject, L, n_prompts=hp["n_prompts"]
            )
            k_stars[i][L] = k

        # Optimize v* at the last MEMIT layer
        v_star = optimize_value_vector(
            editor, req.prompt, req.subject, req.target_new, layers[-1],
            n_steps=hp["n_steps"],
            lr=hp["lr"],
            kl_weight=hp["kl_weight"],
            weight_decay=hp["weight_decay"],
            early_stop_loss=hp["early_stop_loss"],
            delta_norm_factor=hp["delta_norm_factor"],
        )
        v_stars.append(v_star)

    # Step 3: Distribute residual across layers (last to first)
    # Collect all deltas to apply
    all_deltas: Dict[int, torch.Tensor] = {}  # layer -> delta_W

    # For each request, compute the initial residual at the last layer
    residuals = []
    for i, req in enumerate(requests):
        last_layer = layers[-1]
        W = editor.get_parameter(last_layer, "mlp.c_proj").data.float().cpu()
        k = k_stars[i][last_layer].float()
        current_v = W @ k
        r = v_stars[i].float() - current_v
        residuals.append(r)

    # Process layers from last to first
    for layer_idx_pos in range(n_layers - 1, -1, -1):
        L = layers[layer_idx_pos]
        # Number of remaining layers (including this one)
        n_remaining = layer_idx_pos + 1

        # Per-layer share of residual for each request
        layer_residuals = [r / n_remaining for r in residuals]

        # Build the key matrix K and residual matrix R for this layer
        K = torch.stack([k_stars[i][L].float() for i in range(n_requests)], dim=1)
        R = torch.stack(layer_residuals, dim=1)

        # Use float64 for the critical matrix solve to improve numerical stability
        C = cov_dict[L].to(torch.float64).to(device)
        lambda_reg = hp["lambda_reg"]
        C_reg = C + lambda_reg * torch.eye(C.shape[0], device=device, dtype=torch.float64)

        K_dev = K.to(torch.float64).to(device)
        R_dev = R.to(torch.float64).to(device)

        # Batch solve: C_reg @ X = K  =>  X = C_reg^{-1} @ K
        X = torch.linalg.solve(C_reg, K_dev)  # (n_feats, n_requests)

        # Normalize: for each request, divide by k_i^T @ x_i
        denominators = (K_dev * X).sum(dim=0, keepdim=True)  # (1, n_requests)
        denominators = denominators.clamp(min=1e-10)
        X_normalized = X / denominators  # (n_feats, n_requests)

        # delta_W = R @ X_normalized^T
        delta_W = (R_dev @ X_normalized.T).float()

        all_deltas[L] = delta_W.cpu()

        # Update residuals: subtract contribution of this layer's delta
        for i in range(n_requests):
            k_i = k_stars[i][L].float()
            contribution = (delta_W.cpu().float() @ k_i)
            residuals[i] = residuals[i] - contribution

    # Step 4: Apply all deltas
    total_delta_norms = {}
    for L in layers:
        delta_W = all_deltas[L].to(device)
        param = editor.get_parameter(L, "mlp.c_proj")
        delta_W = delta_W.to(param.dtype)
        editor.apply_delta(L, "mlp.c_proj", delta_W)
        total_delta_norms[L] = delta_W.float().norm().item()

    # Step 5: Evaluate each edit
    results = []
    for i, req in enumerate(requests):
        input_ids = torch.tensor(
            [editor.tokenizer.encode(req.prompt)], dtype=torch.long, device=device
        )
        with torch.no_grad():
            logits, _ = editor.model(input_ids)
            probs = F.softmax(logits[0, -1], dim=-1)
            top_token = torch.argmax(probs).item()
            target_tokens = editor.tokenizer.encode(" " + req.target_new)
            target_first = target_tokens[0] if target_tokens else -1
            efficacy = probs[target_first].item() if target_first >= 0 else 0.0
            success = (top_token == target_first)

        total_norm = sum(total_delta_norms.values())
        results.append(EditResult(
            success=success,
            efficacy=efficacy,
            delta_norm=total_norm,
            metadata={
                "layers": layers,
                "top_token": top_token,
                "top_token_decoded": editor.tokenizer.decode([top_token]),
                "target_first_token": target_first,
                "target_decoded": editor.tokenizer.decode([target_first]) if target_first >= 0 else "",
                "delta_norms_per_layer": total_delta_norms,
                "snapshot_idx": snap_idx,
            },
        ))

    return results
