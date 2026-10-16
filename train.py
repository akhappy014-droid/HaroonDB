"""
Train Haroon-2.5-HK from scratch.
Usage: python train.py --steps 3000
The initial dataset is intentionally small; add licensed/public-domain training data
to data/train.txt for meaningful language and Roblox Luau competence.
"""
import argparse, os, time
import torch
from model import HaroonGPT, GPTConfig
from tokenizer import ByteTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    path = "data/train.txt"
    if not os.path.exists(path):
        raise FileNotFoundError("Missing data/train.txt")
    text = open(path, "r", encoding="utf-8").read()
    tok = ByteTokenizer()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    if len(ids) < 100:
        raise ValueError("Training data is too small. Add more text to data/train.txt.")
    split = max(1, int(len(ids) * 0.9))
    train_ids, val_ids = ids[:split], ids[split:]
    if len(val_ids) < 2:
        val_ids = train_ids[-max(2, len(train_ids)//10):]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig()
    config.vocab_size = tok.vocab_size
    model = HaroonGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"Haroon-2.5-HK | device={device} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Bytes={len(ids):,} | train={len(train_ids):,} | validation={len(val_ids):,}")
    def batch(source):
        ix = torch.randint(0, max(1, len(source)-config.block_size-1), (args.batch_size,))
        x = torch.stack([source[i:i+config.block_size] for i in ix])
        y = torch.stack([source[i+1:i+config.block_size+1] for i in ix])
        return x.to(device), y.to(device)
    best = float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        x, y = batch(train_ids)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(val_ids)
                _, vl = model(vx, vy)
            print(f"step {step}/{args.steps} train_loss={loss.item():.4f} val_loss={vl.item():.4f}")
            if vl.item() < best:
                best = vl.item()
                torch.save({
                    "model": model.state_dict(),
                    "config": {k: getattr(config, k) for k in
                               ["vocab_size","block_size","n_embd","n_head","n_layer","dropout"]},
                    "step": step, "val_loss": best, "model_name": "Haroon-2.5-HK"
                }, "checkpoints/model.pt")
    print("Training finished. Saved checkpoints/model.pt")

if __name__ == "__main__":
    main()
