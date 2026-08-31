# Continuous Bag of Words — Mathematics

## 1. Problem formulation
We need to create a numerical representation of a word (embedding) using the context to the left and to the right from this word in text

## 2. Notation
## Notation

Let $\mathcal{V}$ be the vocabulary and

$$
|\mathcal{V}| = V
$$

Let $d$ be the embedding dimension.

The input and output weight matrices are

$$
W_{in} \in \mathbb{R}^{V \times d}
$$

and

$$
W_{out} \in \mathbb{R}^{d \times V}.
$$

For a context of $C$ words,

$$
h =
\frac{1}{C}
\sum_{i=1}^{C}
W_{in}[I_i, :]
$$

where

$$
h \in \mathbb{R}^{1 \times d}.
$$

The logits are

$$
z = hW_{out},
$$

therefore

$$
z \in \mathbb{R}^{1 \times V}.
$$  

Hence,

$$
\hat{y}_j =
\frac{\exp(z_j)}
{\sum_{k=1}^{V} \exp(z_k)}
$$

for $j = 1, \ldots, V$.
## 3. Forward pass




### 3.1 Context embeddings
### 3.2 Context aggregation
### 3.3 Output logits
### 3.4 Softmax
### 3.5 Cross-entropy loss

## 4. Backpropagation

### 4.1 Gradient w.r.t. logits
### 4.2 Gradient w.r.t. output matrix
### 4.3 Gradient w.r.t. hidden representation
### 4.4 Gradient w.r.t. input embeddings

## 5. Parameter update

## 6. Tensor shapes