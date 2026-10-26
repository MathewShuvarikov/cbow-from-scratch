import numpy as np
import pytest

from cbow.evaluation import most_similar


VOCAB = {
    "cat": 0,
    "dog": 1,
    "car": 2,
    "empty": 3,
}

EMBEDDINGS = np.array([
    [1.0, 0.0],
    [0.8, 0.6],
    [-1.0, 0.0],
    [0.0, 0.0],
])


def test_neighbors_exclude_query_and_zero_vectors():
    result = most_similar(
        "cat", VOCAB, EMBEDDINGS, top_k=10
    )

    assert [word for word, _ in result] == ["dog", "car"]

    np.testing.assert_allclose(
        [score for _, score in result],
        [0.8, -1.0],
    )


def test_frequency_filter():
    frequencies = {
        "cat": 20,
        "dog": 1,
        "car": 12,
        "empty": 100,
    }

    result = most_similar(
        "cat",
        VOCAB,
        EMBEDDINGS,
        frequencies=frequencies,
        min_frequency=5,
    )

    assert [word for word, _ in result] == ["car"]


def test_zero_query():
    with pytest.raises(ValueError, match="zero embedding"):
        most_similar("empty", VOCAB, EMBEDDINGS)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="top_k"):
        most_similar("cat", VOCAB, EMBEDDINGS, top_k=0)

    with pytest.raises(ValueError, match="frequencies"):
        most_similar(
            "cat", VOCAB, EMBEDDINGS, min_frequency=5
        )