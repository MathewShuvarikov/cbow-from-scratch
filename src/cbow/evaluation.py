import numpy as np

def most_similar(
    word: str,
    vocab: dict[str, int],
    embeddings: np.ndarray, # V x Edim, 
    top_k: int = 5
) -> list[tuple[str, float]]:

    word_idx = vocab[word]
    word_embedding = embeddings[word_idx] # (Edim, )

    scalar_product = word_embedding @ embeddings.T # (V, )
    norms = np.linalg.norm(embeddings, axis=1, ord=2)

    cos_sim = scalar_product / (norms*norms[word_idx])

    top_k_elements = np.argsort(cos_sim)[::-1]
    top_k_elements = top_k_elements[top_k_elements!=word_idx]
    top_k_elements = top_k_elements[:top_k]


    inverse_vocab = {value: key for key, value in vocab.items()}

    result = []
    for i in top_k_elements:
        result.append((inverse_vocab[i], float(cos_sim[i])))

    return result
