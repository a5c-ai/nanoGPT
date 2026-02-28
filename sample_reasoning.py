"""
Structured reasoning generation: two-phase sampling with thinking and answer phases.
"""

import argparse
import os
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT
from tokenizer_utils import ReasoningTokenizer, get_tokenizer

# ANSI color codes
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"

THINK_END_ID = 50258
ANSWER_END_ID = 50260


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def generate_reasoning(model, tokenizer, prompt, max_tokens, device):
    """Two-phase generation: thinking (temp=0.7) then answer (temp=0.3)."""
    # Encode prompt with <think> prefix to start thinking phase
    input_ids = tokenizer.encode(prompt + "<think>")
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]

    think_budget = max_tokens // 2
    answer_budget = max_tokens - think_budget

    # Phase 1: thinking (temp=0.7) until </think>
    result = model.generate(
        x, think_budget, temperature=0.7, top_k=200,
        stop_tokens={THINK_END_ID},
    )
    if isinstance(result, dict):
        thinking_ids = result['token_ids'][0].tolist()[len(input_ids):]
    else:
        thinking_ids = result[0].tolist()[len(input_ids):]

    # Remove stop token from thinking if present
    if thinking_ids and thinking_ids[-1] == THINK_END_ID:
        thinking_ids = thinking_ids[:-1]

    # Build context for answer phase: original + thinking + </think><answer>
    answer_prefix = input_ids + thinking_ids + [THINK_END_ID, 50259]
    x2 = torch.tensor(answer_prefix, dtype=torch.long, device=device)[None, ...]

    # Phase 2: answer (temp=0.3) until </answer>
    result2 = model.generate(
        x2, answer_budget, temperature=0.3, top_k=200,
        stop_tokens={ANSWER_END_ID},
    )
    if isinstance(result2, dict):
        answer_ids = result2['token_ids'][0].tolist()[len(answer_prefix):]
    else:
        answer_ids = result2[0].tolist()[len(answer_prefix):]

    # Remove stop token from answer if present
    if answer_ids and answer_ids[-1] == ANSWER_END_ID:
        answer_ids = answer_ids[:-1]

    thinking_text = tokenizer.decode(thinking_ids)
    answer_text = tokenizer.decode(answer_ids)
    return thinking_text, answer_text


def display_output(thinking_text, answer_text, show_thinking):
    """Display structured output with colors and formatting."""
    if show_thinking:
        print(f"\n{DIM}{CYAN}--- Thinking ---{RESET}")
        print(f"{DIM}{thinking_text}{RESET}")
        print(f"{DIM}{CYAN}--- End Thinking ---{RESET}\n")

    print(f"{BOLD}{GREEN}Answer:{RESET}")
    print(answer_text)
    print()


def main():
    parser = argparse.ArgumentParser(description="Structured reasoning generation")
    parser.add_argument("--checkpoint", type=str, default="out-sft/ckpt.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt for reasoning")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--show-thinking", action="store_true", default=True,
                        help="Show thinking phase output (default: True)")
    parser.add_argument("--hide-thinking", action="store_true",
                        help="Hide thinking phase output")
    args = parser.parse_args()

    show_thinking = args.show_thinking and not args.hide_thinking

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = 'bfloat16' if device_type == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)
    tokenizer = get_tokenizer()

    print(f"Generating with prompt: {args.prompt}")
    with torch.no_grad():
        with ctx:
            thinking, answer = generate_reasoning(
                model, tokenizer, args.prompt, args.max_tokens, args.device,
            )

    display_output(thinking, answer, show_thinking)


if __name__ == "__main__":
    main()
