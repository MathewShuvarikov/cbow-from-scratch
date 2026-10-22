from cbow.data import create_training_pairs

def test_training_pairs_values():
    token_ids = [0, 1, 2, 3, 0, 4]
    expected_result = [
    ([0, 2], 1),
    ([1, 3], 2),
    ([2, 0], 3),
    ([3, 4], 0)
    ]
    window_size = 1 
    result = create_training_pairs(token_ids, window_size)

    assert result == expected_result


def test_training_pairs_short_sequence():
    token_ids = [0, 1]
    window_size = 2
    expected_result = []

    result = create_training_pairs(token_ids, window_size)
    assert result == expected_result

