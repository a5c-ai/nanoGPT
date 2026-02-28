"""Task arithmetic operations for model merging and editing (T6.1-T6.3).

Implements task vector extraction, application, TIES-Merging, DARE sparsification,
and persistence utilities.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .data_structures import TaskVector

# Weight tying: in nanoGPT, lm_head.weight is tied to transformer.wte.weight.
# When computing task vectors we skip lm_head.weight to avoid double-counting.
_TIED_KEY = "lm_head.weight"
_TIED_SOURCE = "transformer.wte.weight"


def _is_tied(key: str) -> bool:
    """Check if a state_dict key corresponds to a tied weight that should be skipped."""
    return key == _TIED_KEY


# ---------------------------------------------------------------------------
# T6.1 -- Core operations
# ---------------------------------------------------------------------------

def extract_task_vector(
    base_state_dict: Dict[str, torch.Tensor],
    finetuned_state_dict: Dict[str, torch.Tensor],
    base_model_name: str = "",
    metadata: Optional[Dict] = None,
) -> TaskVector:
    """Compute task vector as W_ft - W_base for each parameter.

    Handles weight tying by skipping lm_head.weight (tied to wte.weight).
    """
    vector_dict: Dict[str, torch.Tensor] = {}
    for key in base_state_dict:
        if _is_tied(key):
            continue
        if key not in finetuned_state_dict:
            continue
        vector_dict[key] = finetuned_state_dict[key].to(torch.float32) - base_state_dict[key].to(torch.float32)
    return TaskVector(
        vector_dict=vector_dict,
        base_model_name=base_model_name,
        metadata=metadata or {},
    )


def apply_task_vector(
    model: nn.Module,
    task_vector: TaskVector,
    alpha: float = 1.0,
) -> None:
    """Add alpha * task_vector to model weights in-place.

    Handles weight tying: updates wte.weight (which is shared with lm_head.weight).
    """
    sd = model.state_dict()
    for key, delta in task_vector.vector_dict.items():
        if key in sd:
            sd[key] = sd[key].to(delta.dtype) + alpha * delta.to(sd[key].device)
    model.load_state_dict(sd, strict=False)


def negate_task_vector(task_vector: TaskVector) -> TaskVector:
    """Return a new TaskVector with all deltas negated."""
    negated = {k: -v for k, v in task_vector.vector_dict.items()}
    return TaskVector(
        vector_dict=negated,
        base_model_name=task_vector.base_model_name,
        metadata={**task_vector.metadata, "negated": True},
    )


def add_task_vectors(vectors: List[TaskVector]) -> TaskVector:
    """Element-wise sum of multiple task vectors."""
    if not vectors:
        raise ValueError("Need at least one task vector")
    all_keys = set()
    for v in vectors:
        all_keys.update(v.vector_dict.keys())

    result: Dict[str, torch.Tensor] = {}
    for key in all_keys:
        tensors = [v.vector_dict[key] for v in vectors if key in v.vector_dict]
        result[key] = sum(tensors[1:], tensors[0].clone())

    return TaskVector(
        vector_dict=result,
        base_model_name=vectors[0].base_model_name,
        metadata={"operation": "add", "num_vectors": len(vectors)},
    )


# ---------------------------------------------------------------------------
# T6.2 -- Advanced merging
# ---------------------------------------------------------------------------

def ties_merge(vectors: List[TaskVector], k: float = 0.2) -> TaskVector:
    """TIES-Merging: Trim, Elect sign, Disjoint merge.

    Args:
        vectors: List of task vectors to merge.
        k: Fraction of largest-magnitude elements to keep per vector (trim threshold).

    Returns:
        Merged TaskVector.
    """
    if not vectors:
        raise ValueError("Need at least one task vector")

    all_keys = set()
    for v in vectors:
        all_keys.update(v.vector_dict.keys())

    result: Dict[str, torch.Tensor] = {}

    for key in all_keys:
        deltas = [v.vector_dict[key] for v in vectors if key in v.vector_dict]
        if not deltas:
            continue

        # Step 1: Trim -- zero out smallest (1-k) fraction of each vector
        trimmed = []
        for d in deltas:
            flat = d.abs().flatten()
            num_keep = max(1, int(k * flat.numel()))
            _, top_indices = flat.topk(num_keep)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[top_indices] = True
            mask = mask.view(d.shape)
            trimmed.append(d * mask)

        # Step 2: Elect sign -- majority vote per element
        signs = torch.stack([torch.sign(t) for t in trimmed], dim=0)  # (n_vectors, *shape)
        sign_sum = signs.sum(dim=0)
        elected_sign = torch.sign(sign_sum)  # +1 or -1; 0 where tied

        # Step 3: Disjoint merge -- average only elements with agreeing sign
        stacked = torch.stack(trimmed, dim=0)
        agrees = (torch.sign(stacked) == elected_sign.unsqueeze(0))  # broadcast
        # Zero out disagreeing elements
        masked = stacked * agrees.float()
        # Count agreeing non-zero elements per position
        counts = agrees.float().sum(dim=0).clamp(min=1)
        merged = masked.sum(dim=0) / counts

        result[key] = merged

    return TaskVector(
        vector_dict=result,
        base_model_name=vectors[0].base_model_name,
        metadata={"operation": "ties_merge", "k": k, "num_vectors": len(vectors)},
    )


def dare_sparsify(
    task_vector: TaskVector,
    p: float = 0.9,
    rescale: bool = True,
    seed: Optional[int] = None,
) -> TaskVector:
    """DARE: Drop And REscale sparsification.

    Randomly drops p fraction of delta elements (sets to 0).
    If rescale, multiplies remaining elements by 1/(1-p).

    Args:
        task_vector: Input task vector.
        p: Drop probability (fraction of elements to zero out).
        rescale: Whether to rescale remaining elements by 1/(1-p).
        seed: Optional random seed for reproducibility.

    Returns:
        Sparsified TaskVector.
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    result: Dict[str, torch.Tensor] = {}
    scale = 1.0 / (1.0 - p) if rescale else 1.0

    for key, delta in task_vector.vector_dict.items():
        mask = torch.bernoulli(torch.full_like(delta, 1.0 - p), generator=gen).bool()
        sparsified = delta * mask.float()
        if rescale:
            sparsified = sparsified * scale
        result[key] = sparsified

    return TaskVector(
        vector_dict=result,
        base_model_name=task_vector.base_model_name,
        metadata={
            **task_vector.metadata,
            "operation": "dare_sparsify",
            "p": p,
            "rescale": rescale,
        },
    )


# ---------------------------------------------------------------------------
# T6.3 -- Persistence
# ---------------------------------------------------------------------------

def save_task_vector(task_vector: TaskVector, path: str) -> None:
    """Save task vector to a .pt file."""
    payload = {
        "vector_dict": task_vector.vector_dict,
        "base_model_name": task_vector.base_model_name,
        "metadata": task_vector.metadata,
    }
    torch.save(payload, path)


def load_task_vector(path: str) -> TaskVector:
    """Load task vector from a .pt file."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return TaskVector(
        vector_dict=payload["vector_dict"],
        base_model_name=payload["base_model_name"],
        metadata=payload.get("metadata", {}),
    )
