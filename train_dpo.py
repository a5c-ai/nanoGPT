"""
DPO (Direct Preference Optimization) training script.

Implements DPO loss with a frozen reference model:
  loss = -log sigmoid(beta * (log_pi_chosen - log_pi_rejected - log_ref_chosen + log_ref_rejected))

Reads preference pairs from JSONL: {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
  Single GPU:
    python train_dpo.py --init_from=out-sft/ckpt.pt

  Override defaults via CLI:
    python train_dpo.py --init_from=out-sft/ckpt.pt --learning_rate=1e-6 --beta=0.1

  With modern architecture (XL model):
    python train_dpo.py --init_from=out-sft/ckpt.pt --model_size=gpt2-xl --modern_arch=True --gradient_checkpointing=True
"""

import os
import json
import time
import math
import copy
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from tokenizer_utils import ReasoningTokenizer

# -----------------------------------------------------------------------------
# default config values for DPO
# I/O
out_dir = 'out-dpo'
eval_interval = 100
log_interval = 10
eval_iters = 20
eval_only = False
always_save_checkpoint = True
init_from = 'out-sft/ckpt.pt'  # path to SFT checkpoint or 'gpt2', 'gpt2-xl', etc.
# wandb logging
wandb_log = False
wandb_project = 'nanogpt-dpo'
wandb_run_name = 'dpo'
# data
data_path = 'data/dpo/train.jsonl'
val_data_path = 'data/dpo/val.jsonl'
# model
model_size = 'gpt2'  # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
modern_arch = False  # enables RoPE + RMSNorm + SwiGLU
gradient_checkpointing = False  # activation checkpointing to save VRAM
dropout = 0.0  # no dropout for DPO
# DPO hyperparameters
beta = 0.1  # DPO temperature parameter
# training
batch_size = 4
gradient_accumulation_steps = 4
block_size = 1024
max_iters = 500
# optimizer
learning_rate = 1e-6
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0
# lr schedule
decay_lr = True
warmup_iters = 20
lr_decay_iters = 500
min_lr = 1e-7
# DDP
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False
seed = 42
# stage metadata
stage = 'dpo'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read())  # overrides from command line or config file

# Model-size-aware defaults for DPO
_dpo_model_defaults = {
    'gpt2':        {'batch_size': 4, 'gradient_accumulation_steps': 4},
    'gpt2-medium': {'batch_size': 2, 'gradient_accumulation_steps': 8},
    'gpt2-large':  {'batch_size': 1, 'gradient_accumulation_steps': 16},
    'gpt2-xl':     {'batch_size': 1, 'gradient_accumulation_steps': 32},
}
if model_size in _dpo_model_defaults:
    _defaults = _dpo_model_defaults[model_size]
    if model_size in ('gpt2-large', 'gpt2-xl'):
        if batch_size == 4:
            batch_size = _defaults['batch_size']
            print(f"Auto-setting batch_size={batch_size} for model_size={model_size}")
        if gradient_accumulation_steps == 4:
            gradient_accumulation_steps = _defaults['gradient_accumulation_steps']
            print(f"Auto-setting gradient_accumulation_steps={gradient_accumulation_steps} for model_size={model_size}")

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
    """Load JSONL file with {prompt, chosen, rejected} entries for DPO."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

print(f"Loading DPO training data from {data_path}")
if os.path.exists(data_path):
    train_data = load_jsonl(data_path)
    print(f"Loaded {len(train_data)} DPO training pairs")
else:
    print(f"WARNING: No DPO data found at {data_path}, creating dummy data for smoke test")
    train_data = [
        {"prompt": "What is 2+2?", "chosen": "<think>Simple addition.</think><answer>4</answer>", "rejected": "<think>Hmm.</think><answer>5</answer>"},
    ]

val_data = []
if os.path.exists(val_data_path):
    print(f"Loading DPO validation data from {val_data_path}")
    val_data = load_jsonl(val_data_path)
    print(f"Loaded {len(val_data)} DPO validation pairs")
else:
    print(f"No DPO validation data found at {val_data_path}, will use training data for eval")

def tokenize_preference_pair(example):
    """Tokenize a preference pair for DPO.

    Returns:
        chosen_ids, chosen_targets, chosen_mask: token IDs, targets, loss mask for chosen
        rejected_ids, rejected_targets, rejected_mask: same for rejected
    """
    prompt_tokens = tokenizer.base.encode(example['prompt'])
    prompt_len = len(prompt_tokens)

    def _encode_completion(prompt, completion):
        # Encode prompt + completion together
        full_text = prompt + completion
        all_tokens = tokenizer.base.encode(full_text)
        if len(all_tokens) > block_size:
            all_tokens = all_tokens[:block_size]
        input_ids = all_tokens[:-1]
        targets = all_tokens[1:]
        # Loss mask: 0 for prompt, 1 for completion
        loss_mask = [0] * min(prompt_len - 1, len(targets)) + [1] * max(0, len(targets) - (prompt_len - 1))
        return input_ids, targets, loss_mask

    chosen_ids, chosen_targets, chosen_mask = _encode_completion(example['prompt'], example['chosen'])
    rejected_ids, rejected_targets, rejected_mask = _encode_completion(example['prompt'], example['rejected'])

    return chosen_ids, chosen_targets, chosen_mask, rejected_ids, rejected_targets, rejected_mask


def pad_and_stack(batch_items, pad_value=0):
    """Pad a list of lists to the same length and stack into a tensor."""
    max_len = min(max(len(x) for x in batch_items), block_size)
    padded = []
    for x in batch_items:
        seq_len = min(len(x), max_len)
        pad_len = max_len - seq_len
        padded.append(x[:seq_len] + [pad_value] * pad_len)
    return torch.tensor(padded, dtype=torch.long)


def get_batch(split):
    """Get a batch of DPO preference pairs."""
    data = train_data if split == 'train' else (val_data if val_data else train_data)
    indices = [random.randint(0, len(data) - 1) for _ in range(batch_size)]

    all_chosen_ids, all_chosen_targets, all_chosen_masks = [], [], []
    all_rejected_ids, all_rejected_targets, all_rejected_masks = [], [], []

    for idx in indices:
        c_ids, c_tgt, c_mask, r_ids, r_tgt, r_mask = tokenize_preference_pair(data[idx])
        all_chosen_ids.append(c_ids)
        all_chosen_targets.append(c_tgt)
        all_chosen_masks.append(c_mask)
        all_rejected_ids.append(r_ids)
        all_rejected_targets.append(r_tgt)
        all_rejected_masks.append(r_mask)

    # Pad chosen and rejected separately
    chosen_x = pad_and_stack(all_chosen_ids, pad_value=eot_token)
    chosen_y = pad_and_stack(all_chosen_targets, pad_value=-1)
    chosen_m = pad_and_stack(all_chosen_masks, pad_value=0).float()
    rejected_x = pad_and_stack(all_rejected_ids, pad_value=eot_token)
    rejected_y = pad_and_stack(all_rejected_targets, pad_value=-1)
    rejected_m = pad_and_stack(all_rejected_masks, pad_value=0).float()

    if device_type == 'cuda':
        chosen_x = chosen_x.pin_memory().to(device, non_blocking=True)
        chosen_y = chosen_y.pin_memory().to(device, non_blocking=True)
        chosen_m = chosen_m.pin_memory().to(device, non_blocking=True)
        rejected_x = rejected_x.pin_memory().to(device, non_blocking=True)
        rejected_y = rejected_y.pin_memory().to(device, non_blocking=True)
        rejected_m = rejected_m.pin_memory().to(device, non_blocking=True)
    else:
        chosen_x, chosen_y, chosen_m = chosen_x.to(device), chosen_y.to(device), chosen_m.to(device)
        rejected_x, rejected_y, rejected_m = rejected_x.to(device), rejected_y.to(device), rejected_m.to(device)

    return chosen_x, chosen_y, chosen_m, rejected_x, rejected_y, rejected_m

# -----------------------------------------------------------------------------
# Model init

model_args = {}

if init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    if modern_arch:
        override_args['modern_arch'] = True
    model = GPT.from_pretrained(init_from, override_args)
    if gradient_checkpointing:
        model.config.gradient_checkpointing = True
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    if modern_arch:
        model_args['modern_arch'] = True
    if gradient_checkpointing:
        model_args['gradient_checkpointing'] = True
    model_args['dropout'] = dropout
elif os.path.isfile(init_from):
    print(f"Loading model from checkpoint: {init_from}")
    checkpoint = torch.load(init_from, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    model_args['dropout'] = dropout
    # Propagate modern_arch from checkpoint or CLI
    if 'modern_arch' in checkpoint_model_args:
        model_args['modern_arch'] = checkpoint_model_args['modern_arch']
    if modern_arch:
        model_args['modern_arch'] = True
    if gradient_checkpointing:
        model_args['gradient_checkpointing'] = True
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    checkpoint = None  # free memory
else:
    raise ValueError(f"Unknown init_from: {init_from}. Use 'gpt2', 'gpt2-xl', etc. or a checkpoint path.")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# -----------------------------------------------------------------------------
# Create frozen reference model (deep copy)
print("Creating frozen reference model...")
ref_model = copy.deepcopy(model)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
# Disable gradient checkpointing for ref model (not needed since no backward)
if hasattr(ref_model.config, 'gradient_checkpointing'):
    ref_model.config.gradient_checkpointing = False
print("Reference model created and frozen.")

# -----------------------------------------------------------------------------
# GradScaler for float16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer (only for policy model)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Compile
if compile_model:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# -----------------------------------------------------------------------------
# DPO Loss

def compute_log_probs_masked(model_obj, input_ids, targets, mask):
    """Compute masked per-sequence log probabilities.

    Returns sum of log_probs over completion tokens for each sequence in the batch.
    """
    log_probs = model_obj.compute_log_probs(input_ids, targets)  # (B, T)
    # Sum log-probs over completion tokens per sequence
    per_seq_lp = (log_probs * mask).sum(dim=-1)  # (B,)
    return per_seq_lp


def dpo_loss(pi_chosen_lp, pi_rejected_lp, ref_chosen_lp, ref_rejected_lp):
    """Compute DPO loss.

    loss = -log sigmoid(beta * (log_pi_chosen - log_pi_rejected - log_ref_chosen + log_ref_rejected))

    Args:
        pi_chosen_lp: (B,) sum of log probs for chosen under policy
        pi_rejected_lp: (B,) sum of log probs for rejected under policy
        ref_chosen_lp: (B,) sum of log probs for chosen under reference
        ref_rejected_lp: (B,) sum of log probs for rejected under reference

    Returns:
        loss: scalar
        reward_margin: mean(pi_chosen - pi_rejected) for monitoring
        accuracy: fraction where chosen is preferred
    """
    pi_diff = pi_chosen_lp - pi_rejected_lp
    ref_diff = ref_chosen_lp - ref_rejected_lp
    logits = beta * (pi_diff - ref_diff)

    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        reward_margin = pi_diff.mean().item()
        accuracy = (logits > 0).float().mean().item()

    return loss, reward_margin, accuracy

# -----------------------------------------------------------------------------
# Evaluation

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            c_x, c_y, c_m, r_x, r_y, r_m = get_batch(split)
            with ctx:
                pi_chosen = compute_log_probs_masked(raw_model, c_x, c_y, c_m)
                pi_rejected = compute_log_probs_masked(raw_model, r_x, r_y, r_m)
                ref_chosen = compute_log_probs_masked(ref_model, c_x, c_y, c_m)
                ref_rejected = compute_log_probs_masked(ref_model, r_x, r_y, r_m)
                loss, _, _ = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected)
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

iter_num = 0
best_val_loss = 1e9
t0 = time.time()

print(f"Starting DPO training | init_from={init_from} | max_iters={max_iters}")
print(f"batch_size={batch_size} | grad_accum={gradient_accumulation_steps} | beta={beta} | lr={learning_rate}")

while iter_num < max_iters:
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
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'stage': stage,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # Forward/backward with gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_margin = 0.0
    total_acc = 0.0

    for micro_step in range(gradient_accumulation_steps):
        c_x, c_y, c_m, r_x, r_y, r_m = get_batch('train')

        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            # Policy log-probs
            pi_chosen = compute_log_probs_masked(raw_model, c_x, c_y, c_m)
            pi_rejected = compute_log_probs_masked(raw_model, r_x, r_y, r_m)

            # Reference log-probs (no grad needed)
            with torch.no_grad():
                ref_chosen = compute_log_probs_masked(ref_model, c_x, c_y, c_m)
                ref_rejected = compute_log_probs_masked(ref_model, r_x, r_y, r_m)

            loss, margin, acc = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * gradient_accumulation_steps
        total_margin += margin
        total_acc += acc

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    avg_loss = total_loss / gradient_accumulation_steps
    avg_margin = total_margin / gradient_accumulation_steps
    avg_acc = total_acc / gradient_accumulation_steps

    if iter_num % log_interval == 0 and master_process:
        print(f"iter {iter_num}: loss {avg_loss:.4f}, margin {avg_margin:.3f}, "
              f"acc {avg_acc:.3f}, time {dt*1000:.2f}ms, lr {lr:.2e}")

    if wandb_log and master_process:
        wandb.log({
            "iter": iter_num,
            "train/dpo_loss": avg_loss,
            "train/reward_margin": avg_margin,
            "train/accuracy": avg_acc,
            "lr": lr,
        })

    iter_num += 1

# Save final checkpoint
if master_process:
    ckpt = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
        'stage': stage,
    }
    print(f"saving final checkpoint to {out_dir}")
    torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))

if ddp:
    destroy_process_group()

print("DPO training complete.")
