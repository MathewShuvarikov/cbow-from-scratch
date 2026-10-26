
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from cbow.data import tokenize, build_vocab, encode_tokens, create_training_pairs
from cbow.train import train_cbow
from cbow.metrics import theoretical_loss_bound

import argparse


ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Train CBOW on Alice in Wonderland"
)

parser.add_argument("--window-size", type=int, default=2)
parser.add_argument("--embedding-dim", type=int, default=30)
parser.add_argument("--learning-rate", type=float, default=0.05)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--max-tokens", type=int, default=None)

parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("artifacts/alice_full"),
)

args = parser.parse_args()

if args.epochs < 1:
    parser.error("--epochs must be positive")
if args.embedding_dim < 1:
    parser.error("--embedding-dim must be positive")
if args.window_size < 1:
    parser.error("--window-size must be positive")
if args.learning_rate <= 0:
    parser.error("--learning-rate must be positive")
if args.max_tokens is not None and args.max_tokens < 1:
    parser.error("--max-tokens must be positive")

OUT = (ROOT / args.out_dir).resolve()
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load and clean corpus
text = (ROOT / "data" / "raw" / "alice.txt").read_text(encoding="utf-8")

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"

start = text.find(start_marker)
end = text.find(end_marker)

if start == -1 or end == -1 or end <= start:
    raise ValueError("Project Gutenberg markers not found")

start = text.find("\n", start) + 1
clean_text = text[start:end].strip()

# 2. Prepare entire corpus
tokens = tokenize(clean_text)

if args.max_tokens is not None:
    tokens = tokens[:args.max_tokens]

if len(tokens) <= 2 * args.window_size:
    raise ValueError(
        "Corpus is too short for the selected window size"
    )

vocab = build_vocab(tokens)
frequencies = Counter(tokens)

window_size = args.window_size
embedding_dim = args.embedding_dim
learning_rate = args.learning_rate
epochs = args.epochs

token_ids = encode_tokens(tokens, vocab)
training_pairs = create_training_pairs(token_ids, window_size)

print(f"Tokens: {len(tokens)}")
print(f"Vocabulary: {len(vocab)}")
print(f"Training pairs: {len(training_pairs)}")

# 3. Train
W_in, W_out, epoch_losses, epoch_end_losses = train_cbow(
    training_pairs=training_pairs,
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    learning_rate=learning_rate,
    epochs=epochs,
    seed=args.seed,
    shuffle=args.shuffle,
)

# 4. Save model
np.savez_compressed(
    OUT / "model.npz",
    W_in=W_in,
    W_out=W_out,
    epoch_losses=np.array(epoch_losses),
    epoch_end_losses=np.array(epoch_end_losses),
)

# 5. Save vocabulary and data
(OUT / "vocab.json").write_text(
    json.dumps(vocab, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

(OUT / "frequencies.json").write_text(
    json.dumps(dict(frequencies), ensure_ascii=False, indent=2),
    encoding="utf-8"
)

np.save(OUT / "token_ids.npy", np.array(token_ids, dtype=np.int32))

# 6. Save training metadata
metadata = {
    "tokens": len(tokens),
    "vocab_size": len(vocab),
    "training_pairs": len(training_pairs),
    "window_size": window_size,
    "embedding_dim": embedding_dim,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "initial_epoch_loss": float(epoch_losses[0]),
    "final_epoch_loss": float(epoch_losses[-1]),
    "final_epoch_end_loss": float(epoch_end_losses[-1]),
    "seed": args.seed,
    "shuffle": args.shuffle,
    "max_tokens": args.max_tokens,
    "empirical_loss_bound": float(theoretical_loss_bound(training_pairs)),
    "initialization_seed": 42
}

(OUT / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# 7. Save loss curve
plt.figure(figsize=(9, 5))
plt.plot(range(1, epochs + 1), epoch_losses , label="Training Loss")
plt.plot(
    range(1, epochs + 1),
    epoch_end_losses,
    label="Epoch-end Training Loss",
)
plt.legend()
plt.title("CBOW — Alice in Wonderland (Full Corpus)")
plt.xlabel("Epoch")
plt.ylabel("Mean Training Cross-Entropy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "loss_curve.png", dpi=150)
plt.close()

print(f"First epoch loss: {epoch_losses[0]:.6f}")
print(f"Final epoch loss: {epoch_losses[-1]:.6f}")
print(f"Saved to: {OUT}")
