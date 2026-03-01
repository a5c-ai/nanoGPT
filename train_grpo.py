"""
GRPO (Group Relative Policy Optimization) training script with DAPO stability tricks.

Implements the full GRPO loop:
  1. Sample prompts from train_prompts.jsonl
  2. Generate G completions per prompt
  3. Compute rewards via reward.py
  4. Compute group-relative advantages (zero mean within each group)
  5. Clipped surrogate loss with DAPO enhancements
  6. Gradient update

DAPO tricks (T13):
  - Clip-Higher: asymmetric clipping (eps_low=0.2, eps_high=0.28)
  - Entropy bonus with decaying coefficient (0.01 -> 0.001)
  - Dynamic sampling: skip zero-variance reward groups
  - Token-level loss normalization

Usage:
  Single GPU:
    python train_grpo.py

  DDP (4 GPUs):
    torchrun --standalone --nproc_per_node=4 train_grpo.py

  Override defaults via CLI:
    python train_grpo.py --init_from=out-sft/ckpt.pt --learning_rate=3e-6

  Custom data path:
    python train_grpo.py --data_path=data/multi_cot/train.jsonl

  Multi-domain mode (uses general_accuracy_reward for non-math domains):
    python train_grpo.py --data_path=data/multi_cot/train.jsonl --multi_domain=True
"""

import os
import json
import time
import math
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from tokenizer_utils import ReasoningTokenizer
from reward import compute_rewards, compute_rewards_multi

# -----------------------------------------------------------------------------
# default config values for GRPO
# I/O
out_dir = 'out-grpo'
eval_interval = 50
log_interval = 1
eval_iters = 20
eval_only = False
always_save_checkpoint = True
init_from = 'out-sft/ckpt.pt'  # path to SFT checkpoint
# wandb logging
wandb_log = False
wandb_project = 'nanogpt-grpo'
wandb_run_name = 'grpo'
# data
data_path = 'data/gsm8k_cot/train.jsonl'
multi_domain = False  # use general_accuracy_reward for non-math domains
# GRPO hyperparameters
group_size = 8          # G: completions per prompt
batch_size = 4          # number of prompts per iteration
max_gen_tokens = 512    # max new tokens during generation
gen_temperature = 1.0   # temperature for rollout generation
gen_top_k = None        # top-k for generation (None = no top-k)
micro_batch_size = 8    # micro-batch for log-prob recompute
# clipping (DAPO Clip-Higher)
clip_eps_low = 0.2      # lower clip epsilon (standard PPO)
clip_eps_high = 0.28    # upper clip epsilon for positive advantages (DAPO)
# entropy bonus (DAPO)
entropy_coeff_start = 0.01   # initial entropy bonus coefficient
entropy_coeff_end = 0.001    # final entropy bonus coefficient
# reward weights
accuracy_weight = 1.0
format_weight = 0.5
length_weight = 0.1
# optimizer
learning_rate = 3e-6
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0
# lr schedule
decay_lr = True
warmup_iters = 20
lr_decay_iters = 500
min_lr = 3e-7
# training
max_iters = 500
block_size = 1024
dropout = 0.0  # no dropout during RL
# DDP
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False
seed = 42
# stage metadata
stage = 'grpo'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read())  # overrides from command line or config file
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
stop_tokens = {tokenizer.answer_end_id, eot_token}  # stop on </answer> or EOT

# -----------------------------------------------------------------------------
# Data loading

def load_prompts(path):
    """Load JSONL file with {prompt, answer} entries for GRPO."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

print(f"Loading prompt data from {data_path}")
train_prompts = load_prompts(data_path)
print(f"Loaded {len(train_prompts)} training prompts")

def sample_prompt_batch():
    """Sample a batch of prompts with their ground truth answers and domains."""
    indices = [random.randint(0, len(train_prompts) - 1) for _ in range(batch_size)]
    prompts = [train_prompts[i]['prompt'] for i in indices]
    answers = [train_prompts[i]['answer'] for i in indices]
    domains = [train_prompts[i].get('domain', None) for i in indices]
    return prompts, answers, domains

# -----------------------------------------------------------------------------
# Left-pad encoding for batched generation

def left_pad_encode(prompts):
    """Encode prompts and left-pad for batched generation.

    Returns:
        prompt_ids: (B, max_prompt_len) tensor, left-padded with eot_token
        prompt_lengths: list of original prompt lengths
    """
    encoded = [tokenizer.encode(p) for p in prompts]
    prompt_lengths = [len(e) for e in encoded]
    max_len = min(max(prompt_lengths), block_size)

    padded = []
    for tokens in encoded:
        tokens = tokens[-max_len:]  # truncate from left if too long
        pad_len = max_len - len(tokens)
        padded.append([eot_token] * pad_len + tokens)

    prompt_ids = torch.tensor(padded, dtype=torch.long, device=device)
    return prompt_ids, prompt_lengths

# -----------------------------------------------------------------------------
# Model init

print(f"Loading model from {init_from}")
checkpoint = torch.load(init_from, map_location=device)
checkpoint_model_args = checkpoint['model_args']
model_args = {}
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
model_args['dropout'] = dropout
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

# GradScaler for float16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
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

# Entropy coefficient decay (linear)
def get_entropy_coeff(it):
    """Linearly decay entropy coefficient from start to end over training."""
    frac = min(it / max(max_iters, 1), 1.0)
    return entropy_coeff_start + frac * (entropy_coeff_end - entropy_coeff_start)

# -----------------------------------------------------------------------------
# GRPO loss computation with DAPO tricks

def compute_grpo_loss(new_log_probs, old_log_probs, advantages, completion_mask, logits):
    """Compute GRPO clipped surrogate loss with DAPO enhancements.

    Args:
        new_log_probs: (B, T) per-token log-probs under current policy
        old_log_probs: (B, T) per-token log-probs under old policy (from generation)
        advantages: (B,) per-sequence advantages (already group-normalized)
        completion_mask: (B, T) binary mask (1 for completion tokens, 0 for prompt/pad)
        logits: (B, T, V) logits from current policy (for entropy computation)

    Returns:
        policy_loss: scalar, the clipped surrogate loss
        entropy: scalar, mean entropy over completion tokens
        clip_fraction: scalar, fraction of clipped tokens
    """
    # Importance ratio per token
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)  # (B, T)

    # Expand advantages to token level
    adv = advantages.unsqueeze(1)  # (B, 1)

    # Unclipped surrogate
    surr1 = ratio * adv  # (B, T)

    # DAPO Clip-Higher: asymmetric clipping
    # For positive advantages: clip to [1 - eps_low, 1 + eps_high]
    # For negative advantages: clip to [1 - eps_low, 1 + eps_low] (symmetric)
    ratio_clipped = torch.where(
        adv >= 0,
        torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high),
        torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_low),
    )
    surr2 = ratio_clipped * adv  # (B, T)

    # PPO-clip objective (take min for positive adv, max for negative adv = min of both)
    per_token_loss = -torch.min(surr1, surr2)  # (B, T)

    # Token-level loss normalization (DAPO): normalize by total completion tokens
    total_completion_tokens = completion_mask.sum() + 1e-8
    policy_loss = (per_token_loss * completion_mask).sum() / total_completion_tokens

    # Clip fraction for monitoring
    with torch.no_grad():
        clipped = (ratio_clipped != ratio).float()
        clip_fraction = (clipped * completion_mask).sum() / total_completion_tokens

    # Entropy computation over completion tokens
    probs = F.softmax(logits, dim=-1)
    log_probs_all = F.log_softmax(logits, dim=-1)
    entropy_per_token = -(probs * log_probs_all).sum(dim=-1)  # (B, T)
    entropy = (entropy_per_token * completion_mask).sum() / total_completion_tokens

    return policy_loss, entropy, clip_fraction

# -----------------------------------------------------------------------------
# Logging setup

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Training loop

iter_num = 0
best_mean_reward = -float('inf')
t0 = time.time()

print(f"Starting GRPO training | init_from={init_from} | max_iters={max_iters}")
print(f"batch_size={batch_size} | group_size={group_size} | lr={learning_rate}")
print(f"DAPO: clip_eps_low={clip_eps_low}, clip_eps_high={clip_eps_high}")
print(f"DAPO: entropy_coeff {entropy_coeff_start} -> {entropy_coeff_end}")

while iter_num < max_iters:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    entropy_coeff = get_entropy_coeff(iter_num)

    # =========================================================================
    # Step 1: Sample prompts
    # =========================================================================
    prompts, gt_answers, prompt_domains = sample_prompt_batch()

    # =========================================================================
    # Step 2: Encode and left-pad prompts
    # =========================================================================
    prompt_ids, prompt_lengths = left_pad_encode(prompts)
    B = prompt_ids.size(0)
    T_prompt = prompt_ids.size(1)

    # Expand prompts for group generation: (B, T) -> (B*G, T)
    expanded_prompts = prompt_ids.repeat_interleave(group_size, dim=0)  # (B*G, T_prompt)
    expanded_prompt_lengths = []
    for pl in prompt_lengths:
        expanded_prompt_lengths.extend([pl] * group_size)

    # =========================================================================
    # Step 3: Generate G completions per prompt
    # =========================================================================
    model.eval()
    with torch.no_grad():
        gen_result = raw_model.generate(
            expanded_prompts,
            max_new_tokens=max_gen_tokens,
            temperature=gen_temperature,
            top_k=gen_top_k,
            stop_tokens=stop_tokens,
            collect_logprobs=True,
        )
    model.train()

    gen_token_ids = gen_result['token_ids']      # (B*G, T_prompt + T_gen)
    gen_log_probs = gen_result['log_probs']       # (B*G, T_gen)
    gen_lengths = gen_result['lengths']            # (B*G,)

    total_seq_len = gen_token_ids.size(1)
    T_gen = total_seq_len - T_prompt

    # =========================================================================
    # Step 4: Decode completions and compute rewards
    # =========================================================================
    completions = []
    for i in range(B * group_size):
        comp_tokens = gen_token_ids[i, T_prompt:T_prompt + gen_lengths[i].item()].tolist()
        completions.append(tokenizer.decode(comp_tokens))

    # Expand ground truth answers to match B*G
    gt_expanded = []
    for ans in gt_answers:
        gt_expanded.extend([ans] * group_size)

    # Expand domains if available (for multi-domain mode)
    domains_expanded = None
    if multi_domain:
        domains_expanded = []
        for domain in prompt_domains:
            domains_expanded.extend([domain] * group_size)

    if multi_domain:
        rewards_list = compute_rewards_multi(
            completions, gt_expanded,
            accuracy_weight=accuracy_weight,
            format_weight=format_weight,
            length_weight=length_weight,
            domains=domains_expanded,
        )
    else:
        rewards_list = compute_rewards(
            completions, gt_expanded,
            accuracy_weight=accuracy_weight,
            format_weight=format_weight,
            length_weight=length_weight,
        )
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)  # (B*G,)

    # Compute per-completion accuracy for logging
    from reward import accuracy_reward, general_accuracy_reward
    accuracies = []
    for comp, gt in zip(completions, gt_expanded):
        if multi_domain:
            accuracies.append(general_accuracy_reward(comp, gt))
        else:
            accuracies.append(accuracy_reward(comp, gt))
    mean_accuracy = sum(accuracies) / len(accuracies)

    # =========================================================================
    # Step 5: Group-relative advantages
    # =========================================================================
    R = rewards.view(B, group_size)  # (B, G)
    group_mean = R.mean(dim=1, keepdim=True)    # (B, 1)
    group_std = R.std(dim=1, keepdim=True)       # (B, 1)

    # DAPO: Dynamic sampling -- skip zero-variance groups
    valid_groups = (group_std.squeeze(1) > 1e-6)  # (B,)
    num_valid_groups = valid_groups.sum().item()

    if num_valid_groups == 0:
        # All groups have zero variance -- skip this iteration
        if master_process and iter_num % log_interval == 0:
            print(f"iter {iter_num}: skipping -- all groups have zero reward variance")
        iter_num += 1
        continue

    # Compute advantages with group normalization
    advantages = (R - group_mean) / (group_std + 1e-8)  # (B, G)

    # Zero out advantages for invalid (zero-variance) groups
    advantages = advantages * valid_groups.unsqueeze(1).float()  # (B, G)
    advantages = advantages.view(B * group_size)  # (B*G,)

    # =========================================================================
    # Step 6: Build input/target tensors for log-prob recompute
    # =========================================================================
    # We need input_ids and target_ids for compute_log_probs
    # input_ids: full sequence except last token
    # target_ids: full sequence except first token
    # We only care about completion portion for the loss

    # Clamp sequence length to block_size
    max_seq_len = min(total_seq_len, block_size + 1)
    seq_ids = gen_token_ids[:, :max_seq_len]  # (B*G, max_seq_len)

    input_ids = seq_ids[:, :-1]    # (B*G, max_seq_len-1)
    target_ids = seq_ids[:, 1:]    # (B*G, max_seq_len-1)
    T = input_ids.size(1)

    # Completion mask: 1 for completion tokens, 0 for prompt tokens and padding
    completion_mask = torch.zeros(B * group_size, T, dtype=torch.float32, device=device)
    for i in range(B * group_size):
        # Completion starts at T_prompt (in the target indexing, that's T_prompt - 1)
        comp_start = T_prompt - 1  # target position where completion begins
        comp_end = min(T_prompt - 1 + gen_lengths[i].item(), T)
        if comp_start < T and comp_end > comp_start:
            completion_mask[i, comp_start:comp_end] = 1.0

    # Old log-probs: align with the target positions
    # gen_log_probs covers positions T_prompt to T_prompt + T_gen - 1 in the full sequence
    # In target indexing, these correspond to positions (T_prompt - 1) to (T_prompt - 1 + T_gen - 1)
    old_log_probs_full = torch.zeros(B * group_size, T, dtype=torch.float32, device=device)
    gen_lp_len = gen_log_probs.size(1)
    lp_start = T_prompt - 1
    lp_end = min(lp_start + gen_lp_len, T)
    actual_lp_len = lp_end - lp_start
    if actual_lp_len > 0:
        old_log_probs_full[:, lp_start:lp_end] = gen_log_probs[:, :actual_lp_len]

    # =========================================================================
    # Step 7: Micro-batched log-prob recompute + loss + backward
    # =========================================================================
    optimizer.zero_grad(set_to_none=True)

    total_policy_loss = 0.0
    total_entropy = 0.0
    total_clip_frac = 0.0
    num_micro_batches = 0

    total_samples = B * group_size
    for mb_start in range(0, total_samples, micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, total_samples)
        mb_input = input_ids[mb_start:mb_end]
        mb_target = target_ids[mb_start:mb_end]
        mb_old_lp = old_log_probs_full[mb_start:mb_end]
        mb_adv = advantages[mb_start:mb_end]
        mb_mask = completion_mask[mb_start:mb_end]

        if ddp:
            model.require_backward_grad_sync = (mb_start + micro_batch_size >= total_samples)

        with ctx:
            # Forward pass to get logits and new log-probs
            mb_new_lp = raw_model.compute_log_probs(mb_input, mb_target)  # (mb, T)

            # Also get logits for entropy computation
            # Reuse forward pass -- compute_log_probs doesn't return logits,
            # so we do a separate forward for logits (or approximate entropy from log_probs)
            # For efficiency, compute entropy from a separate forward
            b_mb, t_mb = mb_input.size()
            pos = torch.arange(0, t_mb, dtype=torch.long, device=device)
            tok_emb = raw_model.transformer.wte(mb_input)
            pos_emb = raw_model.transformer.wpe(pos)
            x = raw_model.transformer.drop(tok_emb + pos_emb)
            for block in raw_model.transformer.h:
                x = block(x)
            x = raw_model.transformer.ln_f(x)
            mb_logits = raw_model.lm_head(x)  # (mb, T, V)

            # Recompute log-probs from logits (more accurate, shares computation)
            mb_log_probs_all = F.log_softmax(mb_logits, dim=-1)
            target_clamped = mb_target.clamp(min=0)
            mb_new_lp = mb_log_probs_all.gather(2, target_clamped.unsqueeze(-1)).squeeze(-1)
            mb_new_lp = mb_new_lp * (mb_target != -1).float()

            # Compute loss
            policy_loss, entropy, clip_frac = compute_grpo_loss(
                mb_new_lp, mb_old_lp, mb_adv, mb_mask, mb_logits
            )

            # Total loss with entropy bonus
            loss = policy_loss - entropy_coeff * entropy

            # Scale by number of micro-batches for accumulation
            n_micro = math.ceil(total_samples / micro_batch_size)
            loss = loss / n_micro

        scaler.scale(loss).backward()

        total_policy_loss += policy_loss.item()
        total_entropy += entropy.item()
        total_clip_frac += clip_frac.item()
        num_micro_batches += 1

    # Gradient clipping and optimizer step
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    # =========================================================================
    # Step 8: Logging
    # =========================================================================
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    avg_policy_loss = total_policy_loss / max(num_micro_batches, 1)
    avg_entropy = total_entropy / max(num_micro_batches, 1)
    avg_clip_frac = total_clip_frac / max(num_micro_batches, 1)
    mean_reward = rewards.mean().item()
    mean_gen_length = gen_lengths.float().mean().item()
    valid_groups_frac = num_valid_groups / B

    if iter_num % log_interval == 0 and master_process:
        print(
            f"iter {iter_num}: reward {mean_reward:.3f} | acc {mean_accuracy:.3f} | "
            f"entropy {avg_entropy:.3f} | gen_len {mean_gen_length:.0f} | "
            f"clip_frac {avg_clip_frac:.3f} | valid_groups {valid_groups_frac:.2f} | "
            f"loss {avg_policy_loss:.4f} | lr {lr:.2e} | time {dt*1000:.0f}ms"
        )

    if wandb_log and master_process:
        wandb.log({
            "iter": iter_num,
            "train/reward": mean_reward,
            "train/accuracy": mean_accuracy,
            "train/entropy": avg_entropy,
            "train/gen_length": mean_gen_length,
            "train/clip_fraction": avg_clip_frac,
            "train/valid_groups_frac": valid_groups_frac,
            "train/policy_loss": avg_policy_loss,
            "train/entropy_coeff": entropy_coeff,
            "lr": lr,
        })

    # =========================================================================
    # Step 9: Checkpointing
    # =========================================================================
    if (iter_num % eval_interval == 0 or iter_num == max_iters - 1) and master_process:
        if mean_reward > best_mean_reward or always_save_checkpoint:
            best_mean_reward = max(best_mean_reward, mean_reward)
            ckpt = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_mean_reward': best_mean_reward,
                'config': config,
                'stage': stage,
            }
            print(f"saving checkpoint to {out_dir} (reward={mean_reward:.3f})")
            torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))

    iter_num += 1

# Save final checkpoint
if master_process:
    ckpt = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_mean_reward': best_mean_reward,
        'config': config,
        'stage': stage,
    }
    print(f"saving final checkpoint to {out_dir}")
    torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))

if ddp:
    destroy_process_group()

print("GRPO training complete.")
