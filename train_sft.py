"""
SFT (Supervised Fine-Tuning) training script for the reasoning model.

Adapts from train.py for JSONL-based training with on-the-fly tokenization,
completion-only loss masking, and stage-aware checkpointing.

Usage:
  Single GPU:
    python train_sft.py

  DDP (4 GPUs):
    torchrun --standalone --nproc_per_node=4 train_sft.py

  Override defaults via CLI:
    python train_sft.py --init_from=gpt2 --learning_rate=2e-5 --batch_size=4

  Custom data path:
    python train_sft.py --data_path=data/multi_cot/train.jsonl --val_data_path=data/multi_cot/eval/math.jsonl
"""

import os
import json
import time
import math
import random
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from tokenizer_utils import ReasoningTokenizer

# -----------------------------------------------------------------------------
# default config values for SFT
# I/O
out_dir = 'out-sft'
eval_interval = 200
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'gpt2'  # 'gpt2', 'gpt2-medium', 'resume'
# wandb logging
wandb_log = False
wandb_project = 'nanogpt-sft'
wandb_run_name = 'sft'
# data
data_path = 'data/gsm8k_cot/train.jsonl'
val_data_path = 'data/gsm8k_cot/val.jsonl'
# training
batch_size = 4
gradient_accumulation_steps = 8
block_size = 1024
max_iters = 2000
# model
dropout = 0.1
# optimizer
learning_rate = 2e-5
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0
# lr schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 2e-6
# DDP
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # torch.compile; off by default for SFT
seed = 1337
# stage metadata for checkpointing
stage = 'sft'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
_original_data_path = data_path
_original_val_data_path = val_data_path
exec(open('configurator.py').read())  # overrides from command line or config file
# If data_path was changed but val_data_path was not explicitly overridden,
# derive val_data_path from data_path by replacing train.jsonl -> val.jsonl
if data_path != _original_data_path and val_data_path == _original_val_data_path:
    _derived_val = data_path.replace('train.jsonl', 'val.jsonl')
    if _derived_val != data_path and os.path.exists(_derived_val):
        val_data_path = _derived_val
        print(f"Auto-derived val_data_path from data_path: {val_data_path}")
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Tokenizer
tokenizer = ReasoningTokenizer()
eot_token = tokenizer.eot_id  # 50256

# -----------------------------------------------------------------------------
# Data loading

def load_jsonl(path):
    """Load JSONL file, each line has {prompt, thinking, answer}."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

print(f"Loading training data from {data_path}")
train_data = load_jsonl(data_path)
print(f"Loaded {len(train_data)} training examples")

val_data = []
if os.path.exists(val_data_path):
    print(f"Loading validation data from {val_data_path}")
    val_data = load_jsonl(val_data_path)
    print(f"Loaded {len(val_data)} validation examples")
else:
    print(f"No validation data found at {val_data_path}, will use training data for eval")


def tokenize_example(example):
    """Tokenize a single example and compute loss mask.

    Returns (input_ids, targets, loss_mask) as lists.
    Loss mask is 0 for prompt tokens, 1 for thinking+answer tokens.
    """
    prompt_tokens = tokenizer.base.encode(example['prompt'])
    # Build completion: <think>thinking</think><answer>answer</answer>
    completion_tokens = tokenizer.encode_reasoning_example(
        example['prompt'], example['thinking'], example['answer']
    )
    # completion_tokens includes prompt + thinking + answer
    # The prompt portion is completion_tokens[:len(prompt_tokens)]
    prompt_len = len(prompt_tokens)
    all_tokens = completion_tokens  # already includes prompt

    # Truncate to block_size + 1 (need +1 for targets shift)
    if len(all_tokens) > block_size:
        all_tokens = all_tokens[:block_size]

    # input_ids: all_tokens[:-1], targets: all_tokens[1:]
    input_ids = all_tokens[:-1]
    targets = all_tokens[1:]

    # Loss mask: 0 for positions predicting prompt tokens, 1 for completion tokens
    # Position i in targets corresponds to predicting token at position i+1 in all_tokens
    # Prompt tokens are at positions 0..prompt_len-1 in all_tokens
    # So targets predicting prompt tokens are at positions 0..prompt_len-2
    # Completion starts at position prompt_len-1 in targets (predicting token at prompt_len in all_tokens)
    loss_mask = [0] * min(prompt_len - 1, len(targets)) + [1] * max(0, len(targets) - (prompt_len - 1))

    return input_ids, targets, loss_mask


def get_batch(split):
    """Get a batch of tokenized examples with right-padding."""
    data = train_data if split == 'train' else (val_data if val_data else train_data)
    indices = [random.randint(0, len(data) - 1) for _ in range(batch_size)]

    batch_input_ids = []
    batch_targets = []
    batch_loss_masks = []

    for idx in indices:
        input_ids, targets, loss_mask = tokenize_example(data[idx])
        batch_input_ids.append(input_ids)
        batch_targets.append(targets)
        batch_loss_masks.append(loss_mask)

    # Determine max length in this batch (capped at block_size)
    max_len = min(max(len(ids) for ids in batch_input_ids), block_size)

    # Right-pad to max_len
    padded_inputs = []
    padded_targets = []
    padded_masks = []
    for inp, tgt, msk in zip(batch_input_ids, batch_targets, batch_loss_masks):
        seq_len = min(len(inp), max_len)
        pad_len = max_len - seq_len
        padded_inputs.append(inp[:seq_len] + [eot_token] * pad_len)
        padded_targets.append(tgt[:seq_len] + [-1] * pad_len)  # -1 = ignore index
        padded_masks.append(msk[:seq_len] + [0] * pad_len)

    x = torch.tensor(padded_inputs, dtype=torch.long)
    y = torch.tensor(padded_targets, dtype=torch.long)
    m = torch.tensor(padded_masks, dtype=torch.float32)

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        m = m.pin_memory().to(device, non_blocking=True)
    else:
        x, y, m = x.to(device), y.to(device), m.to(device)

    return x, y, m

# -----------------------------------------------------------------------------
# Model init

iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=block_size,
                  bias=True, vocab_size=None, dropout=dropout)

if init_from == 'resume':
    print(f"Resuming SFT training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"Unknown init_from: {init_from}. Use 'gpt2', 'gpt2-medium', etc. or 'resume'.")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# GradScaler for float16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

# Compile
if compile:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Evaluation

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, loss_mask=M)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# LR schedule (cosine with warmup)

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# Logging

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Training loop

raw_model = model.module if ddp else model
X, Y, M = get_batch('train')
t0 = time.time()
local_iter_num = 0

print(f"Starting SFT training | init_from={init_from} | max_iters={max_iters}")
print(f"batch_size={batch_size} | grad_accum={gradient_accumulation_steps} | lr={learning_rate}")

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and checkpoint
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, losses['val'])
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'stage': stage,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # Forward/backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y, loss_mask=M)
            loss = loss / gradient_accumulation_steps
        X, Y, M = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

# Save final checkpoint
if master_process:
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
        'stage': stage,
    }
    print(f"saving final checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

if ddp:
    destroy_process_group()

print("SFT training complete.")
