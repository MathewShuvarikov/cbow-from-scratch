from cbow.train import train_cbow
from cbow.data import (tokenize,
                       build_vocab, 
                       encode_tokens, 
                       create_training_pairs)
from cbow.metrics import theoretical_loss_bound
import matplotlib.pyplot as plt
import numpy as np

text = "The cat sits on the mat"

embedding_dim = 3
window_size = 1
learning_rate = 0.1
epochs = 500

tokens = tokenize(text)
vocab = build_vocab(tokens)
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
plt.savefig("assets/loss_plot.png", dpi=150, bbox_inches="tight")