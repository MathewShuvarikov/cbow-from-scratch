from pathlib import Path
from collections import Counter
from cbow.data import (
    tokenize, build_vocab, encode_tokens, create_training_pairs
)
from cbow.train import train_cbow
from cbow.metrics import theoretical_loss_bound
import matplotlib.pyplot as plt

text = Path("data/raw/alice.txt").read_text(encoding="utf-8")

start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***")
end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***")

start = text.find("\n", start) + 1
clean_text = text[start:end].strip()

tokens = tokenize(clean_text)
tokens = tokens[:5000]
vocab = build_vocab(tokens)

print(f'The number of tokens: {len(tokens)}')
print(f'The length of yhe vocabulary: {len(vocab)}')

frequency_tokens = Counter(tokens)
print('Top-20 most frequent tokens', frequency_tokens.most_common(20))

# cbow applying 
embedding_dim = 20
window_size = 2
learning_rate = 0.05
epochs = 100

token_ids = encode_tokens(tokens, vocab)
training_pairs = create_training_pairs(token_ids, window_size)

W_in, W_out, epoch_losses = train_cbow(training_pairs, 
                                       vocab_size=len(vocab), 
                                       embedding_dim=embedding_dim, 
                                       learning_rate=learning_rate,
                                       epochs=epochs)

theor_loss_bound = theoretical_loss_bound(training_pairs)
print(f"Initial Loss: {epoch_losses[0]:.6f}")
print(f"Final Loss: {epoch_losses[-1]:.6f}")
print(f"Minimum Loss: {min(epoch_losses):.6f}")
print(f"Theoretical Loss Bound: {theor_loss_bound:.6f}")

plt.plot(range(1, epochs + 1), epoch_losses, label="Training Loss")

plt.axhline(
    y=theor_loss_bound,
    linestyle="--",
    label=f"Theoretical Lower Bound: {theor_loss_bound:.3f}"
)

plt.title("CBOW Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy')
plt.legend()
plt.savefig("assets/alice_loss_plot.png", dpi=150, bbox_inches="tight")