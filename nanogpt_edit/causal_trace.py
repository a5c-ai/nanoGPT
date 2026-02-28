"""Causal tracing for nanoGPT models (T2.1-T2.4).

Implements the causal mediation analysis from Meng et al. (2022):
corrupt subject embeddings with noise, then selectively restore
hidden states to measure each (layer, position)'s indirect effect.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .data_structures import CausalTraceResult


def _get_logits_with_embedding_hook(editor, input_ids, hook_fn):
    """Run forward pass with a hook on the embedding output (transformer.drop).

    Args:
        editor: ModelEditor instance.
        input_ids: Token IDs, shape (batch, seq_len).
        hook_fn: Hook function to register on transformer.drop.

    Returns:
        Logits tensor from the full vocabulary (not just last position).
    """
    model = editor.model
    handle = model.transformer.drop.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            # We need full logits, not just last-position.
            # The model's forward without targets returns logits[:, [-1], :].
            # We replicate the forward manually to get all positions.
            device = input_ids.device
            b, t = input_ids.size()
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = model.transformer.wte(input_ids)
            pos_emb = model.transformer.wpe(pos)
            x = model.transformer.drop(tok_emb + pos_emb)
            for block in model.transformer.h:
                x = block(x)
            x = model.transformer.ln_f(x)
            logits = model.lm_head(x)
        return logits
    finally:
        handle.remove()


def _full_forward(model, input_ids):
    """Full forward pass returning logits at all positions."""
    device = input_ids.device
    b, t = input_ids.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        pos_emb = model.transformer.wpe(pos)
        x = model.transformer.drop(tok_emb + pos_emb)
        for block in model.transformer.h:
            x = block(x)
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
    return logits


def _full_forward_with_block_hooks(model, input_ids, hooks_spec):
    """Forward pass with hooks that can corrupt or restore block outputs.

    Args:
        model: GPT model.
        input_ids: Token IDs.
        hooks_spec: List of (layer_idx, hook_fn) tuples to register on blocks.

    Returns:
        Logits at all positions.
    """
    handles = []
    try:
        for layer_idx, hook_fn in hooks_spec:
            h = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
            handles.append(h)
        with torch.no_grad():
            device = input_ids.device
            b, t = input_ids.size()
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = model.transformer.wte(input_ids)
            pos_emb = model.transformer.wpe(pos)
            x = model.transformer.drop(tok_emb + pos_emb)
            for block in model.transformer.h:
                x = block(x)
            x = model.transformer.ln_f(x)
            logits = model.lm_head(x)
        return logits
    finally:
        for h in handles:
            h.remove()


def _corrupt_forward(model, input_ids, subject_positions, noise, clean_embeds=None):
    """Forward pass with noise added to subject token embeddings.

    Returns (logits, corrupted_embeddings_before_blocks).
    We also cache block outputs for restoration.
    """
    device = input_ids.device
    b, t = input_ids.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        pos_emb = model.transformer.wpe(pos)
        x = tok_emb + pos_emb
        # Add noise to subject positions
        for p in subject_positions:
            x[:, p, :] = x[:, p, :] + noise[:, p, :]
        x = model.transformer.drop(x)

        # Cache block outputs for later restoration
        block_outputs = []
        for block in model.transformer.h:
            x = block(x)
            block_outputs.append(x.clone())
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
    return logits, block_outputs


def _corrupt_and_restore_forward(
    model, input_ids, subject_positions, noise,
    clean_block_outputs, restore_layer, restore_position
):
    """Forward with corruption + single (layer, position) restored from clean run."""
    device = input_ids.device
    b, t = input_ids.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        pos_emb = model.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for p in subject_positions:
            x[:, p, :] = x[:, p, :] + noise[:, p, :]
        x = model.transformer.drop(x)

        for li, block in enumerate(model.transformer.h):
            x = block(x)
            if li == restore_layer:
                # Restore just the single token position from clean
                x[:, restore_position, :] = clean_block_outputs[li][:, restore_position, :]

        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
    return logits


def _corrupt_and_restore_component_forward(
    model, input_ids, subject_positions, noise,
    clean_component_outputs, restore_layer, restore_position, component
):
    """Forward with corruption + single (layer, position, component) restored."""
    device = input_ids.device
    b, t = input_ids.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        pos_emb = model.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for p in subject_positions:
            x[:, p, :] = x[:, p, :] + noise[:, p, :]
        x = model.transformer.drop(x)

        for li, block in enumerate(model.transformer.h):
            if li == restore_layer:
                # Manually compute block forward with selective restoration
                # Block: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))
                attn_out = block.attn(block.ln_1(x))
                if component == 'attn':
                    attn_out[:, restore_position, :] = (
                        clean_component_outputs[('attn', li)][:, restore_position, :]
                    )
                x = x + attn_out

                mlp_out = block.mlp(block.ln_2(x))
                if component == 'mlp':
                    mlp_out[:, restore_position, :] = (
                        clean_component_outputs[('mlp', li)][:, restore_position, :]
                    )
                x = x + mlp_out
            else:
                x = block(x)

        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
    return logits


def _clean_forward_with_cache(model, input_ids):
    """Clean forward caching block outputs and component (mlp/attn) outputs."""
    device = input_ids.device
    b, t = input_ids.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        pos_emb = model.transformer.wpe(pos)
        x = model.transformer.drop(tok_emb + pos_emb)

        block_outputs = []
        component_outputs = {}
        for li, block in enumerate(model.transformer.h):
            attn_out = block.attn(block.ln_1(x))
            component_outputs[('attn', li)] = attn_out.clone()
            x = x + attn_out
            mlp_out = block.mlp(block.ln_2(x))
            component_outputs[('mlp', li)] = mlp_out.clone()
            x = x + mlp_out
            block_outputs.append(x.clone())

        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
    return logits, block_outputs, component_outputs


def trace(
    editor,
    prompt: str,
    subject: str,
    target: Optional[str] = None,
    noise_std: float = 0.1,
    n_noise: int = 10,
) -> CausalTraceResult:
    """Causal tracing: measure indirect effect of each (layer, position).

    Args:
        editor: ModelEditor instance.
        prompt: Input prompt string.
        subject: Subject string within prompt.
        target: Target token string. If None, uses the model's top prediction.
        noise_std: Standard deviation of Gaussian noise for corruption.
        n_noise: Number of noise samples to average over.

    Returns:
        CausalTraceResult with scores matrix (n_layers, seq_len).
    """
    model = editor.model
    device = next(model.parameters()).device
    model.eval()

    # Tokenize
    tokens = editor.tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    seq_len = len(tokens)
    n_layers = model.config.n_layer

    # Find subject positions
    subject_positions, last_subject_pos = editor.find_subject_tokens(prompt, subject)

    # Clean run
    clean_logits, clean_block_outputs, _ = _clean_forward_with_cache(model, input_ids)
    # Last token logits -> predicted token
    last_logits = clean_logits[0, -1, :]
    if target is None:
        target_id = last_logits.argmax().item()
    else:
        target_tokens = editor.tokenizer.encode(target)
        # Use first token of target (handle space-prefixed tokens)
        if len(target_tokens) == 0:
            # Try with space prefix
            target_tokens = editor.tokenizer.encode(" " + target)
        target_id = target_tokens[0]

    clean_prob = F.softmax(last_logits, dim=-1)[target_id].item()

    # Accumulate scores over noise samples
    scores = torch.zeros(n_layers, seq_len, device=device)
    n_embd = model.config.n_embd

    for _ in range(n_noise):
        # Generate noise
        noise = torch.randn(1, seq_len, n_embd, device=device) * noise_std
        # Zero out noise for non-subject positions
        mask = torch.zeros(1, seq_len, 1, device=device)
        for p in subject_positions:
            mask[0, p, 0] = 1.0
        noise = noise * mask

        # Corrupted run
        corrupt_logits, _ = _corrupt_forward(model, input_ids, subject_positions, noise)
        corrupt_prob = F.softmax(corrupt_logits[0, -1, :], dim=-1)[target_id].item()

        # Get clean block outputs with this specific run (they don't change, but
        # we already have them from above)

        # Restore loop
        for layer in range(n_layers):
            for pos_idx in range(seq_len):
                restored_logits = _corrupt_and_restore_forward(
                    model, input_ids, subject_positions, noise,
                    clean_block_outputs, layer, pos_idx,
                )
                restored_prob = F.softmax(
                    restored_logits[0, -1, :], dim=-1
                )[target_id].item()
                scores[layer, pos_idx] += (restored_prob - corrupt_prob)

    scores /= n_noise

    # Find peak
    peak_idx = scores.argmax()
    peak_layer = (peak_idx // seq_len).item()

    return CausalTraceResult(
        scores=scores.cpu(),
        layers=list(range(n_layers)),
        components=['block'],
        peak_layer=peak_layer,
        peak_component='block',
    )


def trace_components(
    editor,
    prompt: str,
    subject: str,
    target: Optional[str] = None,
    noise_std: float = 0.1,
    n_noise: int = 10,
) -> Dict[str, CausalTraceResult]:
    """Causal tracing separated by MLP and attention components.

    Returns:
        Dict with 'mlp' and 'attn' keys, each a CausalTraceResult.
    """
    model = editor.model
    device = next(model.parameters()).device
    model.eval()

    tokens = editor.tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    seq_len = len(tokens)
    n_layers = model.config.n_layer
    n_embd = model.config.n_embd

    subject_positions, _ = editor.find_subject_tokens(prompt, subject)

    # Clean run with component cache
    clean_logits, _, clean_component_outputs = _clean_forward_with_cache(model, input_ids)
    last_logits = clean_logits[0, -1, :]
    if target is None:
        target_id = last_logits.argmax().item()
    else:
        target_tokens = editor.tokenizer.encode(target)
        if len(target_tokens) == 0:
            target_tokens = editor.tokenizer.encode(" " + target)
        target_id = target_tokens[0]

    mlp_scores = torch.zeros(n_layers, seq_len, device=device)
    attn_scores = torch.zeros(n_layers, seq_len, device=device)

    for _ in range(n_noise):
        noise = torch.randn(1, seq_len, n_embd, device=device) * noise_std
        mask = torch.zeros(1, seq_len, 1, device=device)
        for p in subject_positions:
            mask[0, p, 0] = 1.0
        noise = noise * mask

        corrupt_logits, _ = _corrupt_forward(model, input_ids, subject_positions, noise)
        corrupt_prob = F.softmax(corrupt_logits[0, -1, :], dim=-1)[target_id].item()

        for component, score_mat in [('mlp', mlp_scores), ('attn', attn_scores)]:
            for layer in range(n_layers):
                for pos_idx in range(seq_len):
                    restored_logits = _corrupt_and_restore_component_forward(
                        model, input_ids, subject_positions, noise,
                        clean_component_outputs, layer, pos_idx, component,
                    )
                    restored_prob = F.softmax(
                        restored_logits[0, -1, :], dim=-1
                    )[target_id].item()
                    score_mat[layer, pos_idx] += (restored_prob - corrupt_prob)

    mlp_scores /= n_noise
    attn_scores /= n_noise

    mlp_peak = (mlp_scores.argmax() // seq_len).item()
    attn_peak = (attn_scores.argmax() // seq_len).item()

    return {
        'mlp': CausalTraceResult(
            scores=mlp_scores.cpu(),
            layers=list(range(n_layers)),
            components=['mlp'],
            peak_layer=mlp_peak,
            peak_component='mlp',
        ),
        'attn': CausalTraceResult(
            scores=attn_scores.cpu(),
            layers=list(range(n_layers)),
            components=['attn'],
            peak_layer=attn_peak,
            peak_component='attn',
        ),
    }


def find_critical_layer(result: CausalTraceResult) -> int:
    """Return layer with maximum indirect effect at the subject's last token.

    Since we don't know which token is the last subject token from just the
    result, we use the overall peak layer (max across all positions).

    Args:
        result: CausalTraceResult from trace().

    Returns:
        Layer index with highest indirect effect.
    """
    # Max over all positions per layer, then take argmax layer
    max_per_layer = result.scores.max(dim=1).values
    return max_per_layer.argmax().item()


def find_critical_layer_range(
    result: CausalTraceResult, threshold: float = 0.5
) -> Tuple[int, int]:
    """Return range of layers above threshold * max_effect.

    Args:
        result: CausalTraceResult from trace().
        threshold: Fraction of max effect to use as cutoff.

    Returns:
        (start_layer, end_layer) inclusive range.
    """
    max_per_layer = result.scores.max(dim=1).values
    max_effect = max_per_layer.max().item()
    cutoff = threshold * max_effect

    above = (max_per_layer >= cutoff).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        peak = max_per_layer.argmax().item()
        return (peak, peak)
    return (above[0].item(), above[-1].item())


def plot_trace(result: CausalTraceResult, save_path: Optional[str] = None):
    """Plot causal trace heatmap.

    Args:
        result: CausalTraceResult with scores matrix.
        save_path: If provided, save PNG to this path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scores = result.scores.numpy()
    n_layers, seq_len = scores.shape

    fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.8), max(4, n_layers * 0.4)))
    im = ax.imshow(scores, aspect='auto', origin='lower', cmap='hot')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_title(f'Causal Trace ({result.peak_component})')
    fig.colorbar(im, ax=ax, label='Indirect Effect')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
