"""nanogpt_edit: Surgical model editing toolkit for nanoGPT."""

from .data_structures import (
    CausalTraceResult,
    EditRequest,
    EditResult,
    SteeringVector,
    TaskVector,
)
from .edit_core import ModelEditor
from .causal_trace import trace, trace_components, find_critical_layer, plot_trace
from .rome import rome_edit
from .memit import memit_edit, MEMIT_HPARAMS_124M
from .evaluation import eval_full, eval_efficacy
from .task_arithmetic import extract_task_vector, apply_task_vector, ties_merge, dare_sparsify
from .steering import compute_steering_vector, SteeringHook, multi_layer_steer, save_steering_vector, load_steering_vector

__all__ = [
    "ModelEditor",
    "EditRequest",
    "EditResult",
    "CausalTraceResult",
    "TaskVector",
    "SteeringVector",
    "trace",
    "trace_components",
    "find_critical_layer",
    "plot_trace",
    "rome_edit",
    "memit_edit",
    "MEMIT_HPARAMS_124M",
    "eval_full",
    "eval_efficacy",
    "extract_task_vector",
    "apply_task_vector",
    "ties_merge",
    "dare_sparsify",
    "compute_steering_vector",
    "SteeringHook",
    "multi_layer_steer",
    "save_steering_vector",
    "load_steering_vector",
]
