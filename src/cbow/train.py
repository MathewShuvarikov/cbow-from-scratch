import numpy as np
from cbow.model import (initialize_weights, 
                        create_context_vector, 
                        compute_hidden, 
                        compute_logits, 
                        softmax, 
                        cross_entropy_from_logits,
                        backward,
                        update_weights,
                        compute_loss)
from tqdm import tqdm

def evaluate_cbow_loss(
    training_pairs: list[tuple[list[int], int]],
    vocab_size: int,
    W_in: np.ndarray,
    W_out: np.ndarray,
) -> float:

    if not training_pairs:
        raise ValueError("training_pairs must not be empty")

    total_loss = 0.0

    for context, target_id in training_pairs:
        x = create_context_vector(context, vocab_size)

        total_loss += compute_loss(
            x, target_id, W_in, W_out
        )

    return total_loss / len(training_pairs)

def train_cbow(
    training_pairs: list[tuple[list[int], int]],
    vocab_size: int,
    embedding_dim: int,
    learning_rate: float,
    epochs: int,
    seed: int = 42,
    shuffle: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[float],
    list[float],
]:

    W_in, W_out = initialize_weights(
    vocab_size,
    embedding_dim,
    seed=seed
    )

    rng = np.random.default_rng(seed)

    epoch_losses = []
    epoch_end_losses = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_agg = []

        if shuffle:
            indices = rng.permutation(len(training_pairs))
        else:
            indices = range(len(training_pairs))

        for index in indices:
            context, target_id = training_pairs[index]

            # 2. Context vector
            x = create_context_vector(context, vocab_size)

            # 3. Hidden representation
            h = compute_hidden(x, W_in)

            # 4. Logits
            z = compute_logits(h, W_out)

            # 5. Softmax
            y_hat = softmax(z)

            # 6. Cross-Entropy
            loss = cross_entropy_from_logits(z, target_id)

            loss_agg.append(loss)

            dW_in, dW_out = backward(x, h, y_hat, target_id, W_out)

            W_in, W_out = update_weights(W_in, W_out, dW_in, dW_out, learning_rate)

        epoch_losses.append(float(np.mean(loss_agg)))
        epoch_end_losses.append(
        evaluate_cbow_loss(
            training_pairs,
            vocab_size,
            W_in,
            W_out,
        )
    )

    return W_in, W_out, epoch_losses, epoch_end_losses