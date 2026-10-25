import re

# def tokenize(text: str) -> list[str]:
#     return text.lower().split()

def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", text.lower())

def build_vocab(tokens: list[str]) -> dict[str, int]:
    vocab = {}
    next_id = 0

    for token in tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1

    return vocab

def encode_tokens(
    tokens: list[str],
    vocab: dict[str, int]
) -> list[int]:
    return [vocab[token] for token in tokens]


def create_training_pairs(
    token_ids: list[int],
    window_size: int
) -> list[tuple[list[int], int]]:

    if window_size < 1:
        raise ValueError("window_size must be positive")

    training_list = []

    for i in range(window_size, len(token_ids)-window_size):
        left_context = token_ids[i-window_size:i]
        right_context = token_ids[i+1:i+window_size+1]

        training_list.append((left_context + right_context, token_ids[i]))

    return training_list