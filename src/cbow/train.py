import numpy as np
from cbow.model import (initialize_weights, 
                        create_context_vector, 
                        compute_hidden, 
                        compute_logits, 
                        softmax, 
                        cross_entropy,
                        backward,
                        update_weights)
from tqdm import tqdm

def train_cbow(
    training_pairs: list[tuple[list[int], int]],
    vocab_size: int,
    embedding_dim: int,
    learning_rate: float,
    epochs: int
) -> tuple[np.ndarray, np.ndarray, list[float]]:

    W_in, W_out = initialize_weights(vocab_size, embedding_dim)

    epoch_losses = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_agg = []
        for context, target_id in training_pairs:

            # 2. Context vector
            x = create_context_vector(context, vocab_size)

            # 3. Hidden representation
            h = compute_hidden(x, W_in)

            # 4. Logits
            z = compute_logits(h, W_out)

            # 5. Softmax
            y_hat = softmax(z)

            # 6. Cross-Entropy
            loss = cross_entropy(y_hat, target_id)

            loss_agg.append(loss)

            dW_in, dW_out = backward(x, h, y_hat, target_id, W_out)

            W_in, W_out = update_weights(W_in, W_out, dW_in, dW_out, learning_rate)

        epoch_losses.append(float(np.mean(loss_agg)))

    return W_in, W_out, epoch_losses