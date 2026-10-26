import numpy as np

def most_similar(
    word: str,
    vocab: dict[str, int],
    embeddings: np.ndarray,
    top_k: int = 5,
    *,
    frequencies: dict[str, int] | None = None,
    min_frequency: int = 1,
) -> list[tuple[str, float]]:

    if top_k < 1:
        raise ValueError("top_k must be positive")

    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")

    if min_frequency > 1 and frequencies is None:
        raise ValueError(
            "frequencies are required for min_frequency > 1"
        )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(vocab):
        raise ValueError(
            "embeddings must have shape "
            "(vocab_size, embedding_dim)"
        )

    if word not in vocab:
        raise KeyError(f"{word!r} is not in the vocabulary")

    word_idx = vocab[word]

    norms = np.linalg.norm(embeddings, axis=1)

    if norms[word_idx] == 0:
        raise ValueError("query word has a zero embedding")

    # Exclude zero vectors and the query word
    eligible = norms > 0
    eligible[word_idx] = False

    # Optional frequency filtering
    if frequencies is not None:
        for token, idx in vocab.items():
            if frequencies.get(token, 0) < min_frequency:
                eligible[idx] = False

    candidate_ids = np.flatnonzero(eligible)

    scores = (
        embeddings[candidate_ids] @ embeddings[word_idx]
    ) / (norms[candidate_ids] * norms[word_idx])

    order = np.argsort(-scores, kind="stable")[:top_k]

    inverse_vocab = {
        idx: token for token, idx in vocab.items()
    }

    return [
        (inverse_vocab[int(candidate_ids[i])], float(scores[i]))
        for i in order
    ]