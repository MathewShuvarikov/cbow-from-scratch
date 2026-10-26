import numpy as np
import pytest

from cbow.train import train_cbow, evaluate_cbow_loss
from cbow.model import (
    initialize_weights,
    create_context_vector,
    compute_loss,
)

TRAINING_PAIRS = [
    ([0, 1], 2),
    ([0, 2], 1),
    ([1, 3], 0),
    ([2, 3], 1),
    ([0, 3], 2),
    ([1, 2], 3),
]

PARAMS = {
    "training_pairs": TRAINING_PAIRS,
    "vocab_size": 4,
    "embedding_dim": 3,
    "learning_rate": 0.5,
    "epochs": 5,
    "seed": 42,
}


@pytest.mark.parametrize("shuffle", [False, True])
def test_training_reproducibility(shuffle):

    first = train_cbow(**PARAMS, shuffle=shuffle)
    second = train_cbow(**PARAMS, shuffle=shuffle)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_shuffling_changes_training():

    W_in_fixed, _, _, _ = train_cbow(
        **PARAMS,
        shuffle=False
    )

    W_in_shuffled, _, _, _ = train_cbow(
        **PARAMS,
        shuffle=True
    )

    assert not np.array_equal(
        W_in_fixed,
        W_in_shuffled
    )

def test_evaluate_uniform_model():
    W_in = np.zeros((4, 3))
    W_out = np.zeros((3, 4))

    loss = evaluate_cbow_loss(
        TRAINING_PAIRS,
        vocab_size=4,
        W_in=W_in,
        W_out=W_out,
    )

    np.testing.assert_allclose(loss, np.log(4))


def test_epoch_end_loss_matches_final_model():
    W_in, W_out, _, epoch_end_losses = train_cbow(
        **PARAMS,
        shuffle=True,
    )

    expected = evaluate_cbow_loss(
        TRAINING_PAIRS,
        vocab_size=4,
        W_in=W_in,
        W_out=W_out,
    )

    np.testing.assert_allclose(
        epoch_end_losses[-1],
        expected,
    )

def test_training_logs_correct_loss():
    pairs = [([0, 2], 1)]

    W_in, W_out = initialize_weights(
        vocab_size=4,
        embedding_dim=3,
        seed=42,
    )

    x = create_context_vector([0, 2], vocab_size=4)

    expected_loss = compute_loss(x, 1, W_in, W_out)

    _, _, epoch_losses, _ = train_cbow(
        training_pairs=pairs,
        vocab_size=4,
        embedding_dim=3,
        learning_rate=0.1,
        epochs=1,
        seed=42,
        shuffle=False,
    )

    np.testing.assert_allclose(
        epoch_losses[0],
        expected_loss,
        rtol=0,
        atol=1e-10,
    )