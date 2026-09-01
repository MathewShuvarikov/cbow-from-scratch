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
## 3.4 Softmax

Let the output logits be

$$
\mathbf{z} = (z_1, z_2, \ldots, z_V),
$$

where $V$ is the vocabulary size.

The softmax probability for class $i$ is

$$
\hat{y}_i
=
\frac{e^{z_i}}
{\sum_{k=1}^{V} e^{z_k}}.
$$

Denote

$$
S = \sum_{k=1}^{V} e^{z_k}.
$$

Then

$$
\hat{y}_i = \frac{e^{z_i}}{S}.
$$

### Derivative of softmax: $i=j$

We first compute

$$
\frac{\partial \hat{y}_i}{\partial z_i}.
$$

Using the quotient rule,

$$
\frac{\partial \hat{y}_i}{\partial z_i}
=
\frac{
e^{z_i}S - e^{z_i}e^{z_i}
}
{S^2}.
$$

Factor out $e^{z_i}$:

$$
=
\frac{
e^{z_i}(S-e^{z_i})
}
{S^2}.
$$

Split the fraction:

$$
=
\frac{e^{z_i}}{S}
\left(
1-\frac{e^{z_i}}{S}
\right).
$$

Therefore,

$$
\boxed{
\frac{\partial \hat{y}_i}{\partial z_i}
=
\hat{y}_i(1-\hat{y}_i)
}
$$

### Derivative of softmax: $i\neq j$

Now consider

$$
\hat{y}_j
=
\frac{e^{z_j}}{S}
$$

and differentiate with respect to $z_i$, where $i\neq j$.

Since $e^{z_j}$ does not depend on $z_i$,

$$
\frac{\partial e^{z_j}}{\partial z_i}=0.
$$

However,

$$
\frac{\partial S}{\partial z_i}=e^{z_i}.
$$

Thus,

$$
\frac{\partial \hat{y}_j}{\partial z_i}
=
-\frac{e^{z_j}e^{z_i}}{S^2}.
$$

Therefore,

$$
\boxed{
\frac{\partial \hat{y}_j}{\partial z_i}
=
-\hat{y}_j\hat{y}_i
}
\qquad i\neq j.
$$

Both cases can be written as one formula using the Kronecker delta:

$$
\boxed{
\frac{\partial \hat{y}_j}{\partial z_i}
=
\hat{y}_j
\left(
\delta_{ij}-\hat{y}_i
\right)
}
$$

where

$$
\delta_{ij}
=
\begin{cases}
1, & i=j,\\
0, & i\neq j.
\end{cases}
$$

## 3.5 Cross-entropy loss

For one training example,

$$
L
=
-\sum_{j=1}^{V}
y_j\ln \hat{y}_j.
$$

Since $\mathbf{y}$ is one-hot, only the true class contributes to the loss.

The derivative with respect to a particular predicted probability $\hat{y}_j$ is

$$
\boxed{
\frac{\partial L}{\partial \hat{y}_j}
=
-\frac{y_j}{\hat{y}_j}
}
$$

# 4. Backpropagation

## 4.1 Gradient w.r.t. logits

We want to compute

$$
\frac{\partial L}{\partial z_i}.
$$

Since every softmax output $\hat{y}_j$ depends on $z_i$, the chain rule gives

$$
\frac{\partial L}{\partial z_i}
=
\sum_{j=1}^{V}
\frac{\partial L}{\partial \hat{y}_j}
\frac{\partial \hat{y}_j}{\partial z_i}.
$$

There are two cases.

For $j=i$:

$$
-\frac{y_i}{\hat{y}_i}
\hat{y}_i(1-\hat{y}_i)
=
-y_i(1-\hat{y}_i).
$$

For $j\neq i$:

$$
-\frac{y_j}{\hat{y}_j}
\left(
-\hat{y}_j\hat{y}_i
\right)
=
y_j\hat{y}_i.
$$

Therefore,

$$
\frac{\partial L}{\partial z_i}
=
-y_i(1-\hat{y}_i)
+
\sum_{j\neq i}
y_j\hat{y}_i.
$$

Since

$$
\sum_{j=1}^{V}y_j=1,
$$

we have

$$
\sum_{j\neq i}y_j
=
1-y_i.
$$

Hence,

$$
\frac{\partial L}{\partial z_i}
=
-y_i(1-\hat{y}_i)
+
\hat{y}_i(1-y_i).
$$

Expanding,

$$
=
-y_i
+
y_i\hat{y}_i
+
\hat{y}_i
-
y_i\hat{y}_i.
$$

The middle terms cancel:

$$
\boxed{
\frac{\partial L}{\partial z_i}
=
\hat{y}_i-y_i
}
$$

Therefore, in vector form,

$$
\boxed{
\frac{\partial L}{\partial \mathbf{z}}
=
\hat{\mathbf{y}}-\mathbf{y}
}
$$

## 4.2 Gradient w.r.t. output matrix

The logits are

$$
\mathbf{z}
=
\mathbf{h}W_{\text{out}}.
$$

Therefore,

$$
\frac{\partial L}{\partial W_{\text{out}}}
=
\mathbf{h}^T
\frac{\partial L}{\partial \mathbf{z}}.
$$

Using

$$
\frac{\partial L}{\partial \mathbf{z}}
=
\hat{\mathbf{y}}-\mathbf{y},
$$

we obtain

$$
\boxed{
\frac{\partial L}{\partial W_{\text{out}}}
=
\mathbf{h}^T
(\hat{\mathbf{y}}-\mathbf{y})
}
$$

## 4.3 Gradient w.r.t. hidden representation

Again,

$$
\mathbf{z}
=
\mathbf{h}W_{\text{out}}.
$$

The gradient with respect to the hidden representation is

$$
\frac{\partial L}{\partial \mathbf{h}}
=
\frac{\partial L}{\partial \mathbf{z}}
W_{\text{out}}^T.
$$

Therefore,

$$
\boxed{
\frac{\partial L}{\partial \mathbf{h}}
=
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^T
}
$$

## 4.4 Gradient w.r.t. input embeddings

In CBOW, the hidden representation is the average of the context embeddings:

$$
\mathbf{h}
=
\frac{1}{C}
\sum_{c=1}^{C}
W_{\text{in}}[w_c],
$$

where $C$ is the number of context words.

For each context embedding,

$$
\frac{\partial \mathbf{h}}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}.
$$

Therefore,

$$
\frac{\partial L}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}
\frac{\partial L}{\partial \mathbf{h}}.
$$

Substituting the gradient with respect to $\mathbf{h}$,

$$
\boxed{
\frac{\partial L}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^T
}
$$

for every word $w_c$ in the context.

Rows of $W_{\text{in}}$ corresponding to words that are not present in the context receive zero gradient for this training example.

# 5. Parameter update

After computing the gradients, gradient descent updates the parameters as

$$
\boxed{
W_{\text{out}}
\leftarrow
W_{\text{out}}
-
\eta
\frac{\partial L}{\partial W_{\text{out}}}
}
$$

and

$$
\boxed{
W_{\text{in}}
\leftarrow
W_{\text{in}}
-
\eta
\frac{\partial L}{\partial W_{\text{in}}}
}
$$

where $\eta$ is the learning rate.

# Batch case

For a batch of $N$ examples,

$$
H\in\mathbb{R}^{N\times d},
$$

$$
W_{\text{out}}\in\mathbb{R}^{d\times V},
$$

and

$$
Z=HW_{\text{out}}
\in\mathbb{R}^{N\times V}.
$$

The target and prediction matrices are

$$
Y,\hat{Y}\in\mathbb{R}^{N\times V}.
$$

If the loss is averaged over the batch,

$$
L
=
\frac{1}{N}
\sum_{n=1}^{N}L_n,
$$

then

$$
\boxed{
\frac{\partial L}{\partial Z}
=
\frac{1}{N}
(\hat{Y}-Y)
}
$$

and

$$
\boxed{
\frac{\partial L}{\partial W_{\text{out}}}
=
\frac{1}{N}
H^T(\hat{Y}-Y)
}
$$

while

$$
\boxed{
\frac{\partial L}{\partial H}
=
\frac{1}{N}
(\hat{Y}-Y)
W_{\text{out}}^T
}
$$

The important point is that we do **not** first average gradients over classes and then over examples.

The matrix

$$
\hat{Y}-Y
$$

already contains the gradient with respect to every logit of every training example.

Aggregation over the batch occurs naturally when the gradients of the shared model parameters are computed.

# 6. Tensor shapes

For one training example:

$$
W_{\text{in}}
\in
\mathbb{R}^{V\times d}
$$

$$
\mathbf{h}
\in
\mathbb{R}^{1\times d}
$$

$$
W_{\text{out}}
\in
\mathbb{R}^{d\times V}
$$

$$
\mathbf{z}
\in
\mathbb{R}^{1\times V}
$$

$$
\hat{\mathbf{y}},
\mathbf{y}
\in
\mathbb{R}^{1\times V}
$$

The main gradients are therefore

$$
\boxed{
\frac{\partial L}{\partial \mathbf{z}}
=
\hat{\mathbf{y}}-\mathbf{y}
}
\qquad
1\times V
$$

$$
\boxed{
\frac{\partial L}{\partial W_{\text{out}}}
=
\mathbf{h}^T(\hat{\mathbf{y}}-\mathbf{y})
}
\qquad
d\times V
$$

$$
\boxed{
\frac{\partial L}{\partial \mathbf{h}}
=
(\hat{\mathbf{y}}-\mathbf{y})W_{\text{out}}^T
}
\qquad
1\times d
$$

and, for every context word,

$$
\boxed{
\frac{\partial L}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^T
}
\qquad
1\times d.
$$
