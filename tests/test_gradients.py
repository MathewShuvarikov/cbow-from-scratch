import numpy as np
import pytest

from cbow.model import (
    backward,
    compute_hidden,
    compute_logits,
    compute_loss,
    create_context_vector,
    softmax,
)


def numerical_gradient(loss_fn, weights, eps=1e-5):
    gradient = np.zeros_like(weights)

    for index in np.ndindex(weights.shape):
        weights_plus = weights.copy()
        weights_minus = weights.copy()

        weights_plus[index] += eps
        weights_minus[index] -= eps

        gradient[index] = (
            loss_fn(weights_plus) - loss_fn(weights_minus)
        ) / (2 * eps)

    return gradient

@pytest.mark.parametrize(
    "context,target_id",
    [
        ([0, 2], 1),
        ([0, 0, 2], 3),
    ],
)
def test_backward_matches_finite_differences(context, target_id):
    W_in = np.array([
        [0.2, -0.1, 0.3],
        [-0.4, 0.5, 0.1],
        [0.2, 0.7, -0.2],
        [0.3, -0.2, 0.4],
    ])

    W_out = np.array([
        [0.2, -0.1, 0.4, 0.3],
        [-0.3, 0.5, -0.2, 0.1],
        [0.1, 0.2, -0.4, 0.6],
    ])

    x = create_context_vector(context, vocab_size=4)
    h = compute_hidden(x, W_in)
    y_hat = softmax(compute_logits(h, W_out))

    # Analytical gradients
    dW_in, dW_out = backward(
        x, h, y_hat, target_id, W_out
    )

    # Numerical gradients
    numeric_dW_in = numerical_gradient(
        lambda w: compute_loss(x, target_id, w, W_out),
        W_in
    )

    numeric_dW_out = numerical_gradient(
        lambda w: compute_loss(x, target_id, W_in, w),
        W_out
    )

    np.testing.assert_allclose(
        dW_in, numeric_dW_in,
        rtol=1e-5, atol=1e-7
    )

    np.testing.assert_allclose(
        dW_out, numeric_dW_out,
        rtol=1e-5, atol=1e-7
    )