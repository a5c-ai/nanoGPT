"""Steering vectors for inference-time behavioral control (T7.1-T7.3)."""

import contextlib
from typing import List, Optional

import torch
import torch.nn as nn

from .data_structures import SteeringVector
from .edit_core import ModelEditor


def compute_steering_vector(
    editor: ModelEditor,
    positive_texts: List[str],
    negative_texts: List[str],
    layer: int,
    aggregation: str = "mean_seq",
) -> SteeringVector:
    """Compute a steering vector from contrastive text pairs.

    Args:
        editor: ModelEditor with model and tokenizer.
        positive_texts: Texts representing the desired direction.
        negative_texts: Texts representing the opposite direction.
        layer: Transformer block index to extract activations from.
        aggregation: 'mean_seq' averages over sequence positions;
                     'last_token' uses only the last token position.

    Returns:
        SteeringVector with the computed direction.
    """
    if aggregation not in ("mean_seq", "last_token"):
        raise ValueError(f"Unknown aggregation: {aggregation}")

    model = editor.model
    device = next(model.parameters()).device
    model.eval()

    def _collect(texts: List[str]) -> torch.Tensor:
        vecs = []
        for text in texts:
            tokens = editor.tokenizer.encode(text)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            cache = editor.cache_activations(input_ids, [layer], ["block"])
            act = cache[f"block.{layer}"]  # (1, seq_len, n_embd)
            if aggregation == "mean_seq":
                vecs.append(act.squeeze(0).mean(dim=0))  # (n_embd,)
            else:  # last_token
                vecs.append(act.squeeze(0)[-1])  # (n_embd,)
        return torch.stack(vecs).mean(dim=0)  # (n_embd,)

    with torch.no_grad():
        pos_mean = _collect(positive_texts)
        neg_mean = _collect(negative_texts)

    vector = (pos_mean - neg_mean).cpu()

    return SteeringVector(
        vector=vector,
        layer=layer,
        aggregation=aggregation,
        metadata={
            "n_positive": len(positive_texts),
            "n_negative": len(negative_texts),
        },
    )


class SteeringHook:
    """Context manager that adds a steering vector to a transformer block output.

    Usage:
        with SteeringHook(model, sv, alpha=1.0):
            output = model(input_ids)
    """

    def __init__(self, model: nn.Module, vector: SteeringVector, alpha: float = 1.0):
        self.model = model
        self.vector = vector
        self.alpha = alpha
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None

    def activate(self):
        """Register the forward hook on the target layer."""
        layer_module = self.model.transformer.h[self.vector.layer]
        device = next(self.model.parameters()).device
        sv = self.vector.vector.to(device)
        alpha = self.alpha

        def hook_fn(module, input, output):
            return output + alpha * sv

        self._handle = layer_module.register_forward_hook(hook_fn)

    def deactivate(self):
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deactivate()
        return False


@contextlib.contextmanager
def multi_layer_steer(
    model: nn.Module,
    vectors: List[SteeringVector],
    alpha: float = 1.0,
):
    """Context manager that activates steering hooks on multiple layers.

    Args:
        model: The nanoGPT model.
        vectors: List of SteeringVector objects (possibly at different layers).
        alpha: Scaling factor applied to all vectors.
    """
    hooks = [SteeringHook(model, v, alpha) for v in vectors]
    try:
        for h in hooks:
            h.activate()
        yield hooks
    finally:
        for h in hooks:
            h.deactivate()


def save_steering_vector(sv: SteeringVector, path: str) -> None:
    """Save a SteeringVector to a .pt file."""
    torch.save(
        {
            "vector": sv.vector,
            "layer": sv.layer,
            "aggregation": sv.aggregation,
            "metadata": sv.metadata,
        },
        path,
    )


def load_steering_vector(path: str) -> SteeringVector:
    """Load a SteeringVector from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return SteeringVector(
        vector=data["vector"],
        layer=data["layer"],
        aggregation=data["aggregation"],
        metadata=data.get("metadata", {}),
    )
