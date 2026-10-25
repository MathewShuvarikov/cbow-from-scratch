
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from cbow.evaluation import most_similar


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "alice_full"

# 1. Load saved model
with np.load(OUT / "model.npz") as model:
    W_in = model["W_in"]

vocab = json.loads((OUT / "vocab.json").read_text(encoding="utf-8"))

frequencies = Counter(
    json.loads((OUT / "frequencies.json").read_text(encoding="utf-8"))
)

# 2. Evaluate nearest neighbors
focus_words = [
    "alice", "rabbit", "queen", "king", "cat",
    "mouse", "little", "said", "hatter", "garden"
]

results = {}

for word in focus_words:
    if word not in vocab:
        continue

    neighbors = most_similar(word, vocab, W_in, top_k=10)
    results[word] = neighbors

    print(f"\n{word} (frequency={frequencies[word]}):")

    for neighbor, score in neighbors:
        print(
            f"  {neighbor:18s} "
            f"cosine={score:.3f} "
            f"freq={frequencies[neighbor]}"
        )

(OUT / "neighbors.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# 3. L2-normalize embeddings before PCA
norms = np.linalg.norm(W_in, axis=1, keepdims=True)
E = W_in / np.maximum(norms, 1e-12)

# Fit PCA using words occurring at least 5 times
frequent_ids = [
    vocab[word]
    for word, count in frequencies.items()
    if count >= 5
]

pca = PCA(n_components=2)
pca.fit(E[frequent_ids])

coordinates = pca.transform(E)

# 4. Choose words to label
# Stopwords are excluded ONLY from visualization,
# not from training.
stopwords = {
    "the", "and", "to", "a", "it", "she", "i",
    "of", "in", "was", "that", "you", "as",
    "her", "at", "on", "for", "but", "had",
    "is", "with", "not", "be", "so", "he",
    "they", "this", "there", "if", "were",
    "have", "my", "his", "or", "an"
}

plot_words = list(dict.fromkeys(
    [w for w in focus_words if w in vocab] +
    [w for w, _ in frequencies.most_common() if w not in stopwords]
))[:50]

plot_ids = [vocab[word] for word in plot_words]

# 5. Plot PCA
fig, ax = plt.subplots(figsize=(14, 10))

# Background: frequently occurring words
ax.scatter(
    coordinates[frequent_ids, 0],
    coordinates[frequent_ids, 1],
    s=10,
    alpha=0.15
)

# Selected words with labels
ax.scatter(
    coordinates[plot_ids, 0],
    coordinates[plot_ids, 1],
    s=35,
    alpha=0.8
)

for word, idx in zip(plot_words, plot_ids):
    ax.annotate(
        word,
        (coordinates[idx, 0], coordinates[idx, 1]),
        fontsize=9,
        xytext=(4, 4),
        textcoords="offset points"
    )

explained = pca.explained_variance_ratio_

ax.set_title("CBOW Word Embeddings — PCA")
ax.set_xlabel(f"PC1 ({explained[0]:.1%} explained variance)")
ax.set_ylabel(f"PC2 ({explained[1]:.1%} explained variance)")
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUT / "embeddings_pca.png", dpi=180)
plt.close()

print("\nPCA explained variance:", explained.sum())
print("Evaluation complete.")
