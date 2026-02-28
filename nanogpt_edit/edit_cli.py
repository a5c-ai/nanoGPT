"""CLI interface for nanogpt_edit toolkit (T8.1)."""

import argparse
import json
import sys
import os


def _load_model_and_editor(checkpoint_path=None):
    """Load GPT-2 model and create ModelEditor.

    If checkpoint_path is provided, load from that checkpoint.
    Otherwise, load fresh GPT-2 124M from nanoGPT.
    """
    import torch
    # Add parent dir to path so we can import model
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from model import GPT, GPTConfig
    from nanogpt_edit.edit_core import ModelEditor
    import tiktoken

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            config_args = checkpoint.get("config", {})
            if isinstance(config_args, dict):
                config = GPTConfig(**config_args)
            else:
                config = config_args
            model = GPT(config)
            state_dict = checkpoint["model"]
            # Strip unwanted prefix
            unwanted_prefix = "_orig_mod."
            for k in list(state_dict.keys()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        else:
            # Assume it's a raw state dict with config
            model = GPT.from_pretrained("gpt2")
    else:
        model = GPT.from_pretrained("gpt2")

    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    editor = ModelEditor(model, tokenizer)
    return model, tokenizer, editor


def cmd_info(args):
    """Print model info."""
    model, tokenizer, editor = _load_model_and_editor(getattr(args, "checkpoint", None))
    config = model.config
    print(f"Model info:")
    print(f"  n_layer:    {config.n_layer}")
    print(f"  n_head:     {config.n_head}")
    print(f"  n_embd:     {config.n_embd}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  block_size: {config.block_size}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {total:,}")


def cmd_trace(args):
    """Run causal tracing."""
    from nanogpt_edit import trace, find_critical_layer, plot_trace

    model, tokenizer, editor = _load_model_and_editor(getattr(args, "checkpoint", None))
    result = trace(editor, args.prompt, args.subject, noise_std=0.1, n_noise=3)
    critical = find_critical_layer(result)
    print(f"Causal trace complete.")
    print(f"  Peak layer: {result.peak_layer}")
    print(f"  Critical layer (max indirect effect): {critical}")

    if args.output_png:
        plot_trace(result, save_path=args.output_png)
        print(f"  Saved plot to: {args.output_png}")

    if args.layer is not None:
        # Show score at specific layer
        scores_at_layer = result.scores[args.layer]
        print(f"  Scores at layer {args.layer}: max={scores_at_layer.max().item():.4f}")


def cmd_rome_edit(args):
    """Apply a ROME edit."""
    import torch
    from nanogpt_edit import rome_edit, EditRequest

    model, tokenizer, editor = _load_model_and_editor(getattr(args, "checkpoint", None))
    request = EditRequest(
        subject=args.subject,
        prompt=args.prompt,
        target_new=args.target_new,
    )
    result = rome_edit(editor, request)
    print(f"ROME edit result:")
    print(f"  success:  {result.success}")
    print(f"  efficacy: {result.efficacy:.4f}")
    print(f"  delta_norm: {result.delta_norm:.4f}")
    print(f"  metadata: {result.metadata}")

    if args.checkpoint_out:
        torch.save({"model": model.state_dict(), "config": vars(model.config)}, args.checkpoint_out)
        print(f"  Saved checkpoint to: {args.checkpoint_out}")


def cmd_memit_edit(args):
    """Apply MEMIT edits from a JSON file."""
    import torch
    from nanogpt_edit import memit_edit, EditRequest

    model, tokenizer, editor = _load_model_and_editor(getattr(args, "checkpoint", None))

    with open(args.edits_json, "r") as f:
        edits_data = json.load(f)

    requests = [
        EditRequest(
            subject=e["subject"],
            prompt=e["prompt"],
            target_new=e["target_new"],
        )
        for e in edits_data
    ]

    results = memit_edit(editor, requests)
    for i, r in enumerate(results):
        print(f"Edit {i}: success={r.success}, efficacy={r.efficacy:.4f}, delta_norm={r.delta_norm:.4f}")

    if args.checkpoint_out:
        torch.save({"model": model.state_dict(), "config": vars(model.config)}, args.checkpoint_out)
        print(f"Saved checkpoint to: {args.checkpoint_out}")


def cmd_task_vector(args):
    """Task vector operations: extract, apply, negate."""
    import torch
    from nanogpt_edit import extract_task_vector, apply_task_vector
    from nanogpt_edit.task_arithmetic import negate_task_vector, save_task_vector, load_task_vector

    if args.operation == "extract":
        if not args.base_checkpoint or not args.ft_checkpoint:
            print("Error: --base-checkpoint and --ft-checkpoint required for extract")
            sys.exit(1)
        base_sd = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        ft_sd = torch.load(args.ft_checkpoint, map_location="cpu", weights_only=False)
        if "model" in base_sd:
            base_sd = base_sd["model"]
        if "model" in ft_sd:
            ft_sd = ft_sd["model"]
        tv = extract_task_vector(base_sd, ft_sd)
        if args.save:
            save_task_vector(tv, args.save)
            print(f"Saved task vector to: {args.save}")
        else:
            print(f"Extracted task vector with {len(tv.vector_dict)} parameters")

    elif args.operation == "apply":
        if not args.base_checkpoint:
            print("Error: --base-checkpoint required for apply")
            sys.exit(1)
        if not args.save:
            print("Error: --save required for apply (output checkpoint)")
            sys.exit(1)
        model, tokenizer, editor = _load_model_and_editor(args.base_checkpoint)
        tv = load_task_vector(args.ft_checkpoint)
        alpha = args.alpha if args.alpha is not None else 1.0
        apply_task_vector(model, tv, alpha=alpha)
        torch.save({"model": model.state_dict(), "config": vars(model.config)}, args.save)
        print(f"Applied task vector (alpha={alpha}), saved to: {args.save}")

    elif args.operation == "negate":
        if not args.ft_checkpoint:
            print("Error: --ft-checkpoint required (task vector file) for negate")
            sys.exit(1)
        tv = load_task_vector(args.ft_checkpoint)
        neg = negate_task_vector(tv)
        out = args.save or args.ft_checkpoint.replace(".pt", "_negated.pt")
        save_task_vector(neg, out)
        print(f"Negated task vector saved to: {out}")

    else:
        print(f"Unknown operation: {args.operation}")
        sys.exit(1)


def cmd_steering(args):
    """Compute a steering vector."""
    from nanogpt_edit import compute_steering_vector
    from nanogpt_edit.steering import save_steering_vector

    model, tokenizer, editor = _load_model_and_editor(getattr(args, "checkpoint", None))

    positive_texts = args.positive_texts
    negative_texts = args.negative_texts
    layer = args.layer
    alpha = args.alpha if args.alpha is not None else 1.0

    sv = compute_steering_vector(editor, positive_texts, negative_texts, layer)
    print(f"Steering vector computed:")
    print(f"  layer: {sv.layer}")
    print(f"  norm:  {sv.vector.norm().item():.4f}")
    print(f"  alpha: {alpha}")

    if args.save:
        save_steering_vector(sv, args.save)
        print(f"  Saved to: {args.save}")


def cmd_eval(args):
    """Run evaluation on test cases."""
    from nanogpt_edit import eval_full

    model, tokenizer, editor = _load_model_and_editor(args.checkpoint)

    with open(args.test_cases, "r") as f:
        test_cases = json.load(f)

    for i, tc in enumerate(test_cases):
        results = eval_full(editor, tc)
        print(f"\nTest case {i}: {tc.get('prompt', 'N/A')}")
        print(f"  Efficacy: {results['efficacy']}")
        if results.get("paraphrase"):
            print(f"  Paraphrase: {results['paraphrase']}")
        if results.get("generation"):
            gen = results["generation"]
            print(f"  Fluent: {gen.get('fluent')}, Contains target: {gen.get('contains_target')}")


def build_parser():
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="nanogpt_edit",
        description="Surgical model editing toolkit for nanoGPT",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    p_info = subparsers.add_parser("info", help="Print model info (n_layers, n_heads, n_embd, vocab_size)")
    p_info.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")

    # trace
    p_trace = subparsers.add_parser("trace", help="Run causal tracing")
    p_trace.add_argument("--prompt", type=str, required=True)
    p_trace.add_argument("--subject", type=str, required=True)
    p_trace.add_argument("--layer", type=int, default=None, help="Show scores at specific layer")
    p_trace.add_argument("--output-png", type=str, default=None, help="Save trace heatmap PNG")
    p_trace.add_argument("--checkpoint", type=str, default=None)

    # rome-edit
    p_rome = subparsers.add_parser("rome-edit", help="Apply a ROME edit")
    p_rome.add_argument("--prompt", type=str, required=True)
    p_rome.add_argument("--subject", type=str, required=True)
    p_rome.add_argument("--target-new", type=str, required=True)
    p_rome.add_argument("--checkpoint", type=str, default=None, help="Input checkpoint")
    p_rome.add_argument("--checkpoint-out", type=str, default=None, help="Save edited model")

    # memit-edit
    p_memit = subparsers.add_parser("memit-edit", help="Apply MEMIT edits from JSON")
    p_memit.add_argument("--edits-json", type=str, required=True, help="JSON file with list of edits")
    p_memit.add_argument("--checkpoint", type=str, default=None)
    p_memit.add_argument("--checkpoint-out", type=str, default=None, help="Save edited model")

    # task-vector
    p_tv = subparsers.add_parser("task-vector", help="Task vector operations")
    p_tv.add_argument("--operation", type=str, required=True, choices=["extract", "apply", "negate"])
    p_tv.add_argument("--base-checkpoint", type=str, default=None)
    p_tv.add_argument("--ft-checkpoint", type=str, default=None)
    p_tv.add_argument("--alpha", type=float, default=None)
    p_tv.add_argument("--save", type=str, default=None, help="Output path")

    # steering
    p_steer = subparsers.add_parser("steering", help="Compute a steering vector")
    p_steer.add_argument("--positive-texts", type=str, nargs="+", required=True)
    p_steer.add_argument("--negative-texts", type=str, nargs="+", required=True)
    p_steer.add_argument("--layer", type=int, required=True)
    p_steer.add_argument("--alpha", type=float, default=None)
    p_steer.add_argument("--save", type=str, default=None, help="Save steering vector to .pt")
    p_steer.add_argument("--checkpoint", type=str, default=None)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation on test cases")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--test-cases", type=str, required=True, help="JSON file with test cases")
    p_eval.add_argument("--method", type=str, choices=["rome", "memit"], default="rome")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "info": cmd_info,
        "trace": cmd_trace,
        "rome-edit": cmd_rome_edit,
        "memit-edit": cmd_memit_edit,
        "task-vector": cmd_task_vector,
        "steering": cmd_steering,
        "eval": cmd_eval,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn is None:
        parser.print_help()
        sys.exit(1)

    cmd_fn(args)


if __name__ == "__main__":
    main()
