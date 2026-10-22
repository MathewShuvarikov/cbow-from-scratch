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
    backward,
    create_context_vector,
    compute_loss
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

# 6. Backward

dW_in, dW_out = backward(hidden=h, y_hat=y_hat, context_vector=x, target_id=target_id, W_out=W_out)

print(dW_in)
print(dW_out)

# 7. Derivative estimation with Loss
grad_analytic = dW_out[0, 0]

E = np.zeros_like(W_out)
E[0, 0] = 1
eps = 10e-6
L_0 = compute_loss(x, target_id, W_in, (W_out-eps*E))
L_1 = compute_loss(x, target_id, W_in, (W_out+eps*E))

grad_num = (L_1-L_0)/(2*eps)
print(grad_num)

print("Analytical:", grad_analytic)
print("Numerical:", grad_num)
print("Absolute error:", abs(grad_analytic - grad_num))

# 7.1
grad_analytic = dW_in[0, 0]

E = np.zeros_like(W_in)
E[0, 0] = 1
eps = 10e-6
L_0 = compute_loss(x, target_id, (W_in-eps*E), W_out)
L_1 = compute_loss(x, target_id, (W_in+eps*E), W_out)

grad_num = (L_1-L_0)/(2*eps)
print(grad_num)

print("Analytical:", grad_analytic)
print("Numerical:", grad_num)
print("Absolute error:", abs(grad_analytic - grad_num))