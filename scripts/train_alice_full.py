
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from cbow.data import tokenize, build_vocab, encode_tokens, create_training_pairs
from cbow.train import train_cbow
from cbow.metrics import theoretical_loss_bound


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "alice_full"
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
vocab = build_vocab(tokens)
frequencies = Counter(tokens)

window_size = 2
embedding_dim = 20
learning_rate = 0.05
epochs = 30

token_ids = encode_tokens(tokens, vocab)
training_pairs = create_training_pairs(token_ids, window_size)

print(f"Tokens: {len(tokens)}")
print(f"Vocabulary: {len(vocab)}")
print(f"Training pairs: {len(training_pairs)}")

# 3. Train
W_in, W_out, epoch_losses = train_cbow(
    training_pairs=training_pairs,
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    learning_rate=learning_rate,
    epochs=epochs
)

# 4. Save model
np.savez_compressed(
    OUT / "model.npz",
    W_in=W_in,
    W_out=W_out,
    epoch_losses=np.array(epoch_losses)
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
    "empirical_loss_bound": float(theoretical_loss_bound(training_pairs)),
    "initialization_seed": 42
}

(OUT / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# 7. Save loss curve
plt.figure(figsize=(9, 5))
plt.plot(range(1, epochs + 1), epoch_losses)
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
