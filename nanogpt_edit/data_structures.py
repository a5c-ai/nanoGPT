"""Data structures for nanogpt_edit toolkit."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class EditRequest:
    """A request to edit a factual association in the model."""
    subject: str
    prompt: str
    target_new: str
    target_old: Optional[str] = None


@dataclass
class EditResult:
    """Result of applying an edit to the model."""
    success: bool
    efficacy: float
    delta_norm: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalTraceResult:
    """Result of causal tracing analysis."""
    scores: torch.Tensor  # (n_layers, seq_len) or similar
    layers: List[int]
    components: List[str]
    peak_layer: int
    peak_component: str


@dataclass
class TaskVector:
    """A task vector representing the difference between fine-tuned and base weights."""
    vector_dict: Dict[str, torch.Tensor]
    base_model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SteeringVector:
    """A steering vector for inference-time behavioral control."""
    vector: torch.Tensor
    layer: int
    aggregation: str = "mean"
    metadata: Dict[str, Any] = field(default_factory=dict)
