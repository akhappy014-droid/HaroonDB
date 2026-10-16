"""
Train Haroon-2.5-HK from random initialization.

Usage:
python train.py --steps 3000

Data:
data/train.txt

Output:
checkpoints/model.pt
"""

import argparse
import math
import os
import random

import torch

from model import HaroonGPT, GPTConfig
from tokenizer import ByteTokenizer

def evaluate(model, source, config, device, batch_size, batches=8):
model.eval()

if len(source) <= config.block_size:
    return float("inf")

losses = []

with torch.inference_mode():
    for _ in range(batches):
        ix = torch.randint(
            0,
            len(source) - config.block_size,
            (batch_size,),
        )

        x = torch.stack([
            source[i:i + config.block_size]
            for i in ix.tolist()
        ]).to(device)

        y = torch.stack([
            source[i + 1:i + config.block_size + 1]
            for i in ix.tolist()
        ]).to(device)

        _, loss = model(x, y)
        losses.append(loss.item())

return sum(losses) / len(losses)

def main():
parser = argparse.ArgumentParser()

parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--eval-interval", type=int, default=100)
parser.add_argument("--eval-batches", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

if args.steps < 1:
    raise ValueError("--steps must be positive")

if args.batch_size < 1 or args.eval_batches < 1:
    raise ValueError("Batch sizes must be positive")

random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

os.makedirs("checkpoints", exist_ok=True)

data_path = "data/train.txt"

if not os.path.isfile(data_path):
    raise FileNotFoundError(f"Missing {data_path}")

with open(data_path, "r", encoding="utf-8") as file:
    text = file.read()

if not text.strip():
    raise ValueError("data/train.txt is empty")

tokenizer = ByteTokenizer()
ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

if len(ids) < 1000:
    raise ValueError(
        f"Training data contains only {len(ids)} bytes. "
        "Add more useful text before training."
    )

# Split on a byte boundary. This is a simple baseline;
# a future dataset pipeline should split by complete documents.
split = int(len(ids) * 0.9)

train_ids = ids[:split]
val_ids = ids[split:]

config = GPTConfig()
config.vocab_size = tokenizer.vocab_size

if len(train_ids) <= config.block_size + 1:
    raise ValueError("Training data is too short for block_size")

if len(val_ids) <= config.block_size + 1:
    raise ValueError(
        "Validation data is too short. Add more training text."
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HaroonGPT(config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=0.05,
    betas=(0.9, 0.95),
)

total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Haroon-2.5-HK | device={device} | params={total_params:,}")
print(
    f"Bytes={len(ids):,} | "
    f"train={len(train_ids):,} | "
    f"validation={len(val_ids):,}"
)

best_val = float("inf")
steps_without_improvement = 0
checkpoint_path = "checkpoints/model.pt"

def get_batch(source):
    max_start = len(source) - config.block_size - 1

    ix = torch.randint(
        0,
        max_start + 1,
        (args.batch_size,),
    )

    x = torch.stack([
        source[i:i + config.block_size]
        for i in ix.tolist()
    ])

    y = torch.stack([
        source[i + 1:i + config.block_size + 1]
        for i in ix.tolist()
    ])

    return x.to(device), y.to(device)

for step in range(1, args.steps + 1):
    model.train()

    # Linear warmup followed by cosine decay.
    warmup_steps = min(100, max(1, args.steps // 10))

    if step <= warmup_steps:
        lr = args.lr * step / warmup_steps
    else:
        progress = (
            (step - warmup_steps)
            / max(1, args.steps - warmup_steps)
        )

        lr = args.lr * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        )

    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_ids)

    _, loss = model(x, y)

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite training loss at step {step}"
        )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    should_evaluate = (
        step == 1
        or step % args.eval_interval == 0
        or step == args.steps
    )

    if not should_evaluate:
        continue

    val_loss = evaluate(
        model,
        val_ids,
        config,
        device,
        args.batch_size,
        args.eval_batches,
    )

    print(
        f"step {step}/{args.steps} "
        f"train_loss={loss.item():.4f} "
        f"val_loss={val_loss:.4f} "
        f"lr={lr:.6g}"
    )

    if val_loss < best_val:
        best_val = val_loss
        steps_without_improvement = 0

        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    key: getattr(config, key)
                    for key in (
                        "vocab_size",
                        "block_size",
                        "n_embd",
                        "n_head",
                        "n_layer",
                        "dropout",
                    )
                },
                "step": step,
                "val_loss": best_val,
                "model_name": "Haroon-2.5-HK",
            },
            checkpoint_path,
        )

        print(f"Saved best checkpoint: {checkpoint_path}")

    else:
        steps_without_improvement += 1

        if steps_without_improvement >= args.patience:
            print("Early stopping: validation loss stopped improving.")
            break

print(f"Training finished. Best validation loss: {best_val:.4f}")

if name == "main":
main()
