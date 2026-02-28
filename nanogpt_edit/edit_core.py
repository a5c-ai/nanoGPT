"""Core model editing infrastructure for nanoGPT."""

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ModelEditor:
    """Provides surgical editing capabilities for nanoGPT models.

    Supports parameter access, activation caching, weight deltas,
    snapshots/restore, and subject token identification.

    The model is expected to follow the nanoGPT architecture:
        model.transformer.h[layer].mlp.c_fc
        model.transformer.h[layer].mlp.c_proj
        model.transformer.h[layer].attn.c_attn
        model.transformer.h[layer].attn.c_proj
        model.transformer.h[layer].ln_1
        model.transformer.h[layer].ln_2
    """

    # Templates for generating varied context prompts
    CONTEXT_TEMPLATES = [
        "The {} is",
        "People know that {}",
        "{} is known for",
        "Everyone has heard of {}",
        "In summary, {}",
        "{} is famous because",
        "The history of {} shows",
        "Experts agree that {}",
        "It is well known that {}",
        "According to sources, {}",
        "Many people believe {}",
        "{} has been described as",
        "When asked about {}, most say",
        "The significance of {} is",
        "One interesting fact about {} is",
    ]

    def __init__(self, model: nn.Module, tokenizer: Any):
        """Initialize the ModelEditor.

        Args:
            model: A nanoGPT model instance.
            tokenizer: A tokenizer (e.g. tiktoken or HF tokenizer) with encode/decode.
        """
        self.model = model
        self.tokenizer = tokenizer
        self._snapshots: List[Dict[str, torch.Tensor]] = []
        self._delta_history: List[Tuple[int, str, torch.Tensor]] = []

        # Detect torch.compile
        if hasattr(model, '_orig_mod'):
            warnings.warn(
                "Model appears to be torch.compiled. Some editing operations "
                "may not work correctly. Call editor.reset_compile() if needed."
            )

    def _resolve_module(self, layer: int, component: str) -> nn.Module:
        """Resolve a component string to an nn.Module.

        Args:
            layer: Transformer block index.
            component: Dot-separated path within the block, e.g. 'mlp.c_proj',
                       'attn.c_attn', 'ln_1', 'mlp', 'attn'.

        Returns:
            The resolved nn.Module.
        """
        block = self.model.transformer.h[layer]
        parts = component.split('.')
        module = block
        for part in parts:
            module = getattr(module, part)
        return module

    def get_parameter(self, layer: int, component: str) -> nn.Parameter:
        """Get a weight parameter from a specific layer and component.

        Args:
            layer: Transformer block index.
            component: Dot-separated path, e.g. 'mlp.c_proj'. The .weight
                       suffix is added automatically.

        Returns:
            The weight nn.Parameter.
        """
        module = self._resolve_module(layer, component)
        return module.weight

    def reset_compile(self):
        """Reset torch dynamo state to allow editing a compiled model."""
        torch._dynamo.reset()

    def cache_activations(
        self,
        input_ids: torch.Tensor,
        layers: List[int],
        components: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass and cache activations at specified locations.

        Supported components:
            'block'     - output of the full transformer block
            'mlp'       - output of the MLP sub-layer
            'attn'      - output of the attention sub-layer
            'mlp_input' - input to mlp.c_proj (i.e. after c_fc + GELU)
            'ln2'       - output of ln_2

        Args:
            input_ids: Token IDs, shape (batch, seq_len).
            layers: List of layer indices.
            components: List of component names to cache.

        Returns:
            Dict keyed like 'mlp.5', 'attn.3', etc. with activation tensors.
        """
        cache: Dict[str, torch.Tensor] = {}
        hooks = []

        try:
            for layer_idx in layers:
                for comp in components:
                    key = f"{comp}.{layer_idx}"

                    if comp == 'block':
                        module = self.model.transformer.h[layer_idx]
                    elif comp == 'mlp':
                        module = self.model.transformer.h[layer_idx].mlp
                    elif comp == 'attn':
                        module = self.model.transformer.h[layer_idx].attn
                    elif comp == 'mlp_input':
                        # Capture input[0] to c_proj (after c_fc + GELU)
                        module = self.model.transformer.h[layer_idx].mlp.c_proj
                    elif comp == 'ln2':
                        module = self.model.transformer.h[layer_idx].ln_2
                    else:
                        raise ValueError(f"Unknown component: {comp}")

                    if comp == 'mlp_input':
                        def make_hook(k):
                            def hook_fn(mod, inp, out):
                                cache[k] = inp[0].detach().clone()
                            return hook_fn
                    else:
                        def make_hook(k):
                            def hook_fn(mod, inp, out):
                                cache[k] = out.detach().clone()
                            return hook_fn

                    h = module.register_forward_hook(make_hook(key))
                    hooks.append(h)

            with torch.no_grad():
                self.model(input_ids)

        finally:
            for h in hooks:
                h.remove()

        return cache

    def apply_delta(
        self,
        layer: int,
        component: str,
        delta: torch.Tensor,
        record: bool = True,
    ):
        """Add a delta tensor to a weight parameter.

        Args:
            layer: Transformer block index.
            component: Component path, e.g. 'mlp.c_proj'.
            delta: Tensor to add, must match weight shape.
            record: If True, record delta for later undo.
        """
        param = self.get_parameter(layer, component)
        with torch.no_grad():
            param.add_(delta.to(param.device))
        if record:
            self._delta_history.append((layer, component, delta.detach().cpu().clone()))

    def snapshot(self) -> int:
        """Save a deep copy of the current model state_dict to CPU.

        Returns:
            Index of the saved snapshot.
        """
        sd = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self._snapshots.append(sd)
        return len(self._snapshots) - 1

    def restore(self, snapshot_idx: int = -1):
        """Restore model weights from a previously saved snapshot.

        Args:
            snapshot_idx: Index of the snapshot to restore (default: last).
        """
        sd = self._snapshots[snapshot_idx]
        device = next(self.model.parameters()).device
        self.model.load_state_dict({k: v.to(device) for k, v in sd.items()})

    def undo_last(self):
        """Subtract the last recorded delta to undo the most recent edit."""
        if not self._delta_history:
            raise RuntimeError("No delta history to undo.")
        layer, component, delta = self._delta_history.pop()
        param = self.get_parameter(layer, component)
        with torch.no_grad():
            param.sub_(delta.to(param.device))

    def find_subject_tokens(
        self, prompt: str, subject: str
    ) -> Tuple[List[int], int]:
        """Find the token positions of a subject within a prompt.

        Uses BPE-aware sliding window: tokenizes the full prompt, then
        tokenizes the subject in context to find matching token subsequence.

        Args:
            prompt: The full prompt string.
            subject: The subject string to locate.

        Returns:
            (positions, last_pos): List of token positions and the last position.
        """
        prompt_tokens = self.tokenizer.encode(prompt)

        # Find subject in prompt string to get surrounding context
        idx = prompt.find(subject)
        if idx == -1:
            raise ValueError(f"Subject '{subject}' not found in prompt '{prompt}'")

        # Use character-level alignment: decode each token to find which
        # tokens correspond to the subject span in the original string.
        # Build a mapping from character positions to token indices.
        char_pos = 0
        token_char_spans = []
        for ti, tok in enumerate(prompt_tokens):
            decoded = self.tokenizer.decode([tok])
            start = char_pos
            end = char_pos + len(decoded)
            token_char_spans.append((start, end))
            char_pos = end

        subject_start_char = idx
        subject_end_char = idx + len(subject)

        positions = []
        for ti, (cs, ce) in enumerate(token_char_spans):
            # Token overlaps with subject span
            if cs < subject_end_char and ce > subject_start_char:
                positions.append(ti)

        if not positions:
            raise ValueError(
                f"Could not map subject '{subject}' to token positions in '{prompt}'"
            )

        last_pos = positions[-1]
        return positions, last_pos

    def generate_context_prompts(self, subject: str, n: int = 10) -> List[str]:
        """Generate varied context prompts containing the subject.

        Args:
            subject: The subject to embed in templates.
            n: Number of prompts to generate (max 15).

        Returns:
            List of prompt strings.
        """
        prompts = [t.format(subject) for t in self.CONTEXT_TEMPLATES[:n]]
        return prompts
