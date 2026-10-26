import numpy as np

def initialize_weights(
    vocab_size: int,
    embedding_dim: int,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    
    rng = np.random.default_rng(seed)
    W_in = rng.normal(size=(vocab_size, embedding_dim), loc=0, scale=0.01)
    W_out = rng.normal(size=(embedding_dim, vocab_size), loc=0, scale=0.01)

    return W_in, W_out

def create_context_vector(
    context: list[int],
    vocab_size: int
) -> np.ndarray:

    if not context:
        raise ValueError("context must not be empty")

    x = np.zeros(shape=(1, vocab_size))

    for i in context:
        x[0, i] += 1/len(context) 

    return x

def compute_hidden(
    context_vector: np.ndarray,
    W_in: np.ndarray
) -> np.ndarray:

    h = context_vector @ W_in

    return h

def compute_logits(
    hidden: np.ndarray,
    W_out: np.ndarray
) -> np.ndarray:

    z = hidden @ W_out

    return z

def softmax(logits: np.ndarray) -> np.ndarray:

    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    y_hat = exp_logits / np.sum(exp_logits)

    return y_hat

def cross_entropy(
    y_hat: np.ndarray,
    target_id: int
) -> float:

    return float(-np.log(y_hat[0, target_id]))

def cross_entropy_from_logits(
    logits: np.ndarray,
    target_id: int
) -> float:

    max_logit = np.max(logits)

    log_sum_exp = max_logit + np.log(
        np.sum(np.exp(logits - max_logit))
    )

    return float(log_sum_exp - logits[0, target_id])

def compute_logits_gradient(
    y_hat: np.ndarray,
    target_id: int
) -> np.ndarray:

    y_true = np.zeros_like(y_hat)
    y_true[0, target_id] = 1

    return y_hat - y_true

def compute_output_gradient(
    hidden: np.ndarray,
    logits_gradient: np.ndarray
) -> np.ndarray:

    return hidden.T @ logits_gradient

def compute_hidden_gradient(
    logits_gradient: np.ndarray,
    W_out: np.ndarray
) -> np.ndarray:

    return logits_gradient @ W_out.T 

def compute_input_gradient(
    context_vector: np.ndarray,
    hidden_gradient: np.ndarray
) -> np.ndarray:

    return context_vector.T @ hidden_gradient

def backward(
    context_vector: np.ndarray,
    hidden: np.ndarray,
    y_hat: np.ndarray,
    target_id: int,
    W_out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    logits_gradient = compute_logits_gradient(y_hat, target_id)
    hidden_gradient = compute_hidden_gradient(logits_gradient, W_out)
    dW_in = compute_input_gradient(context_vector, hidden_gradient)
    dW_out = compute_output_gradient(hidden, logits_gradient)

    return dW_in, dW_out


def update_weights(
    W_in: np.ndarray,
    W_out: np.ndarray,
    dW_in: np.ndarray,
    dW_out: np.ndarray,
    learning_rate: float
) -> tuple[np.ndarray, np.ndarray]:

    W_in_new = W_in - learning_rate*dW_in
    W_out_new = W_out - learning_rate*dW_out 

    return W_in_new, W_out_new


def compute_loss(
    context_vector: np.ndarray,
    target_id: int,
    W_in: np.ndarray,
    W_out: np.ndarray
) -> float:

    hidden = compute_hidden(context_vector, W_in)
    logits = compute_logits(hidden, W_out)

    return cross_entropy_from_logits(logits, target_id)
