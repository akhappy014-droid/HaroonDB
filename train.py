"""
Train Haroon-2.5-HK from random initialization.

Usage:
    python train.py --steps 3000

Input:
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


def make_batch(source, block_size, batch_size, device):
    if len(source) < 2:
        raise ValueError("Dataset is too short")

    # Allow shorter contexts when the validation dataset
    # is smaller than the configured block size.
    context_size = min(block_size, len(source) - 1)

    max_start = len(source) - context_size - 1

    if max_start < 0:
        raise ValueError("Not enough data to build a batch")

    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,),
    )

    x = torch.stack([
        source[i:i + context_size]
        for i in starts.tolist()
    ])

    y = torch.stack([
        source[i + 1:i + context_size + 1]
        for i in starts.tolist()
    ])

    return x.to(device), y.to(device)


def evaluate(
    model,
    source,
    config,
    batch_size,
    device,
    batches=8,
):
    model.eval()

    if len(source) < 2:
        return float("inf")

    losses = []

    with torch.inference_mode():
        for _ in range(batches):
            x, y = make_batch(
                source,
                config.block_size,
                batch_size,
                device,
            )

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

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    if args.eval_interval < 1:
        raise ValueError("--eval-interval must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs("checkpoints", exist_ok=True)

    data_path = "data/train.txt"

    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Missing training file: {data_path}"
        )

    with open(
        data_path,
        "r",
        encoding="utf-8",
    ) as file:
        text = file.read()

    if not text.strip():
        raise ValueError("data/train.txt is empty")

    tokenizer = ByteTokenizer()

    ids = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
    )

    if len(ids) < 1000:
        raise ValueError(
            f"Only {len(ids)} bytes found. "
            "Add more varied training text first."
        )

    # Simple sequential split.
    # For a larger dataset, split by complete documents instead.
    split = int(len(ids) * 0.9)

    train_ids = ids[:split]
    val_ids = ids[split:]

    if len(train_ids) < 300:
        raise ValueError(
            "Training portion is too small for reliable training."
        )

    if len(val_ids) < 2:
        raise ValueError(
            "Validation portion is too small."
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = GPTConfig()
    config.vocab_size = tokenizer.vocab_size

    model = HaroonGPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Haroon-2.5-HK | device={device} "
        f"| params={total_params:,}"
    )

    print(
        f"Bytes={len(ids):,} "
        f"| train={len(train_ids):,} "
        f"| validation={len(val_ids):,}"
    )

    best_val = float("inf")
    steps_without_improvement = 0

    checkpoint_path = "checkpoints/model.pt"

    warmup_steps = min(
        100,
        max(1, args.steps // 10),
    )

    for step in range(1, args.steps + 1):
        model.train()

        if step <= warmup_steps:
            lr = args.lr * step / warmup_steps
        else:
            progress = (
                (step - warmup_steps)
                / max(1, args.steps - warmup_steps)
            )

            lr = args.lr * (
                0.1
                + 0.9
                * 0.5
                * (1 + math.cos(math.pi * progress))
            )

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = make_batch(
            train_ids,
            config.block_size,
            args.batch_size,
            device,
        )

        _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Invalid training loss at step {step}"
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
            args.batch_size,
            device,
            args.eval_batches,
        )

        print(
            f"step {step}/{args.steps} "
            f"train_loss={loss.item():.4f} "
            f"val_loss={val_loss:.4f} "
            f"lr={lr:.7f}"
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

            print(
                f"Saved best checkpoint: {checkpoint_path}"
            )

        else:
            steps_without_improvement += 1

            if (
                steps_without_improvement
                >= args.patience
            ):
                print(
                    "Early stopping: validation loss "
                    "stopped improving."
                )
                break

    print(
        f"Training finished. "
        f"Best validation loss={best_val:.4f}"
    )


if __name__ == "__main__":
    main()
