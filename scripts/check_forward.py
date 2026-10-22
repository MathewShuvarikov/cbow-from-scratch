import numpy as np
from cbow.data import (
    tokenize,
    build_vocab,
    encode_tokens,
    create_training_pairs
)

from cbow.model import (
    initialize_weights,
    compute_hidden,
    compute_logits,
    softmax,
    compute_loss,
    create_context_vector
)

text = "The cat sits on the mat"

tokens = tokenize(text)
vocab = build_vocab(tokens)
token_ids = encode_tokens(tokens, vocab)
training_pairs = create_training_pairs(token_ids, window_size=1)

print("Tokens:", tokens)
print("Vocabulary:", vocab)
print("Token IDs:", token_ids)
print("Training pairs:", training_pairs)

context, target_id = training_pairs[0]

vocab_size = len(vocab)
embedding_dim = 3

# 1. Weights initialization
W_in, W_out = initialize_weights(vocab_size, embedding_dim)

# 2. Context vector
x = create_context_vector(context, vocab_size)

# 3. Hidden representation
h = compute_hidden(x, W_in)

# 4. Logits
z = compute_logits(h, W_out)

# 5. Softmax
y_hat = softmax(z)

# 6. Cross-Entropy
loss = compute_loss(x, target_id, W_in, W_out)

print("Context:", x)
print("Hidden:", h)
print("Logits:", z)
print("Probabilities:", y_hat)
print("Loss:", loss)

assert np.isclose(loss, -np.log(y_hat[0, target_id]))