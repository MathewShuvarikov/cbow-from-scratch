import numpy as np
import pytest

from cbow.model import (
    softmax,
    cross_entropy,
    create_context_vector,
)


def test_softmax_probabilities():
    logits = np.array([[1.0, 2.0, 3.0]])

    probabilities = softmax(logits)

    assert probabilities.shape == logits.shape
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    np.testing.assert_allclose(probabilities.sum(), 1.0)


def test_softmax_numerical_stability():
    logits = np.array([[1000.0, 1001.0, 1002.0]])

    probabilities = softmax(logits)
    expected = softmax(np.array([[0.0, 1.0, 2.0]]))

    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities, expected)


def test_cross_entropy():
    probabilities = np.array([[0.1, 0.7, 0.2]])

    loss = cross_entropy(probabilities, target_id=1)

    np.testing.assert_allclose(loss, -np.log(0.7))


def test_cross_entropy_uniform_distribution():
    probabilities = np.full((1, 4), 0.25)

    loss = cross_entropy(probabilities, target_id=2)

    np.testing.assert_allclose(loss, np.log(4))


def test_context_vector_with_duplicates():
    context = [0, 2, 0, 3]

    result = create_context_vector(context, vocab_size=4)

    expected = np.array([[0.5, 0.0, 0.25, 0.25]])

    np.testing.assert_allclose(result, expected)


def test_empty_context_raises_error():
    with pytest.raises(ValueError, match="context must not be empty"):
        create_context_vector([], vocab_size=4)