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

Let the vocabulary size be $V$, the embedding dimension be $d$, and the context size be $C$.

The input embedding matrix is

```math
W_{\text{in}} \in \mathbb{R}^{V \times d}.
```

For each context word $w_c$, where $c = 1,\dots,C$, we select the corresponding row of $W_{\text{in}}$:

```math
\mathbf{e}_c = W_{\text{in}}[w_c].
```

Each context embedding has shape

```math
\mathbf{e}_c \in \mathbb{R}^{1 \times d}.
```


### 3.2 Context aggregation

CBOW combines the context embeddings by taking their average:

```math
\mathbf{h}
=
\frac{1}{C}
\sum_{c=1}^{C}
\mathbf{e}_c.
```

Equivalently,

```math
\mathbf{h}
=
\frac{1}{C}
\sum_{c=1}^{C}
W_{\text{in}}[w_c].
```

The hidden representation has shape

```math
\mathbf{h} \in \mathbb{R}^{1 \times d}.
```


### 3.3 Output logits

The output embedding matrix is

```math
W_{\text{out}}
\in
\mathbb{R}^{d \times V}.
```

The logits are computed as

```math
\mathbf{z}
=
\mathbf{h}W_{\text{out}}.
```

Therefore,

```math
\mathbf{z}
\in
\mathbb{R}^{1 \times V}.
```

For a particular vocabulary element \(i\),

```math
z_i
=
\sum_{r=1}^{d}
h_r W_{\text{out},ri}.
```


### 3.4 Softmax

The logits are converted into probabilities using softmax:

```math
\hat{y}_i
=
\frac{e^{z_i}}
{\sum_{k=1}^{V} e^{z_k}}.
```

Define

```math
S
=
\sum_{k=1}^{V}e^{z_k}.
```

Then

```math
\hat{y}_i
=
\frac{e^{z_i}}{S}.
```


#### Softmax derivative for \(i=j\)

We first compute

```math
\frac{\partial \hat{y}_i}
{\partial z_i}.
```

Using the quotient rule,

```math
\frac{\partial \hat{y}_i}
{\partial z_i}
=
\frac{
e^{z_i}S
-
e^{z_i}e^{z_i}
}
{S^2}.
```

Factor out \(e^{z_i}\):

```math
=
\frac{
e^{z_i}(S-e^{z_i})
}
{S^2}.
```

Split the expression:

```math
=
\frac{e^{z_i}}{S}
\left(
1-\frac{e^{z_i}}{S}
\right).
```

Hence,

```math
\boxed{
\frac{\partial \hat{y}_i}
{\partial z_i}
=
\hat{y}_i(1-\hat{y}_i)
}
```


#### Softmax derivative for \(i \neq j\)

For

```math
\hat{y}_j
=
\frac{e^{z_j}}{S},
```

take the derivative with respect to \(z_i\), where \(i \neq j\).

Since \(e^{z_j}\) does not depend on \(z_i\),

```math
\frac{\partial e^{z_j}}
{\partial z_i}
=
0.
```

However,

```math
\frac{\partial S}
{\partial z_i}
=
e^{z_i}.
```

Therefore,

```math
\frac{\partial \hat{y}_j}
{\partial z_i}
=
-
\frac{
e^{z_j}e^{z_i}
}
{S^2}.
```

Thus,

```math
\boxed{
\frac{\partial \hat{y}_j}
{\partial z_i}
=
-\hat{y}_j\hat{y}_i
}
```

for \(i \neq j\).

Both cases can be written using the Kronecker delta:

```math
\boxed{
\frac{\partial \hat{y}_j}
{\partial z_i}
=
\hat{y}_j
\left(
\delta_{ij}
-
\hat{y}_i
\right)
}
```

where

```math
\delta_{ij}
=
\begin{cases}
1, & i=j, \\
0, & i\neq j.
\end{cases}
```


### 3.5 Cross-entropy loss

For one training example, the cross-entropy loss is

```math
L
=
-
\sum_{j=1}^{V}
y_j
\ln \hat{y}_j.
```

Since \(\mathbf{y}\) is one-hot, only the target word contributes to the loss.

If \(t\) is the index of the target word,

```math
L
=
-\ln \hat{y}_t.
```

The derivative with respect to a particular predicted probability is

```math
\boxed{
\frac{\partial L}
{\partial \hat{y}_j}
=
-
\frac{y_j}{\hat{y}_j}
}
```


## 4. Backpropagation

### 4.1 Gradient w.r.t. logits

We want to find

```math
\frac{\partial L}{\partial z_i}.
```

Since every softmax output \(\hat{y}_j\) depends on \(z_i\), the chain rule gives

```math
\frac{\partial L}{\partial z_i}
=
\sum_{j=1}^{V}
\frac{\partial L}{\partial \hat{y}_j}
\frac{\partial \hat{y}_j}{\partial z_i}.
```


For \(j=i\),

```math
-
\frac{y_i}{\hat{y}_i}
\hat{y}_i
(1-\hat{y}_i)
=
-y_i(1-\hat{y}_i).
```


For \(j \neq i\),

```math
-
\frac{y_j}{\hat{y}_j}
\left(
-\hat{y}_j\hat{y}_i
\right)
=
y_j\hat{y}_i.
```

Therefore,

```math
\frac{\partial L}{\partial z_i}
=
-y_i(1-\hat{y}_i)
+
\sum_{j\neq i}
y_j\hat{y}_i.
```

Since

```math
\sum_{j=1}^{V} y_j = 1,
```

we have

```math
\sum_{j\neq i}y_j
=
1-y_i.
```

Hence,

```math
\frac{\partial L}{\partial z_i}
=
-y_i(1-\hat{y}_i)
+
\hat{y}_i(1-y_i).
```

Expanding,

```math
=
-y_i
+
y_i\hat{y}_i
+
\hat{y}_i
-
y_i\hat{y}_i.
```

The middle terms cancel:

```math
\boxed{
\frac{\partial L}{\partial z_i}
=
\hat{y}_i-y_i
}
```

Therefore, in vector form,

```math
\boxed{
\frac{\partial L}{\partial \mathbf{z}}
=
\hat{\mathbf{y}}
-
\mathbf{y}
}
```


### 4.2 Gradient w.r.t. output matrix

Recall that

```math
\mathbf{z}
=
\mathbf{h}W_{\text{out}}.
```

Therefore,

```math
\frac{\partial L}
{\partial W_{\text{out}}}
=
\mathbf{h}^{T}
\frac{\partial L}{\partial \mathbf{z}}.
```

Substituting the gradient with respect to the logits,

```math
\boxed{
\frac{\partial L}
{\partial W_{\text{out}}}
=
\mathbf{h}^{T}
(\hat{\mathbf{y}}-\mathbf{y})
}
```


### 4.3 Gradient w.r.t. hidden representation

Again,

```math
\mathbf{z}
=
\mathbf{h}W_{\text{out}}.
```

Therefore,

```math
\frac{\partial L}
{\partial \mathbf{h}}
=
\frac{\partial L}
{\partial \mathbf{z}}
W_{\text{out}}^{T}.
```

Hence,

```math
\boxed{
\frac{\partial L}
{\partial \mathbf{h}}
=
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^{T}
}
```


### 4.4 Gradient w.r.t. input embeddings

The hidden representation is

```math
\mathbf{h}
=
\frac{1}{C}
\sum_{c=1}^{C}
W_{\text{in}}[w_c].
```

For every context word,

```math
\frac{\partial \mathbf{h}}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}.
```

Therefore,

```math
\frac{\partial L}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}
\frac{\partial L}
{\partial \mathbf{h}}.
```

Substituting the gradient with respect to \(\mathbf{h}\),

```math
\boxed{
\frac{\partial L}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^{T}
}
```

for every context word $w_c$

Rows of $W_{in}$ corresponding to words that do not occur in the current context receive zero gradient.


## 5. Parameter update

The output matrix is updated as

```math
W_{\text{out}}
\leftarrow
W_{\text{out}}
-
\eta
\frac{\partial L}
{\partial W_{\text{out}}}.
```

The input embedding matrix is updated as

```math
W_{\text{in}}
\leftarrow
W_{\text{in}}
-
\eta
\frac{\partial L}
{\partial W_{\text{in}}}.
```

Here, $\eta$ is the learning rate.


## 6. Tensor shapes

For one training example:

```math
W_{\text{in}}
\in
\mathbb{R}^{V\times d}
```

```math
\mathbf{h}
\in
\mathbb{R}^{1\times d}
```

```math
W_{\text{out}}
\in
\mathbb{R}^{d\times V}
```

```math
\mathbf{z}
\in
\mathbb{R}^{1\times V}
```

```math
\hat{\mathbf{y}},
\mathbf{y}
\in
\mathbb{R}^{1\times V}
```

The corresponding gradients have shapes

```math
\boxed{
\frac{\partial L}
{\partial \mathbf{z}}
=
\hat{\mathbf{y}}-\mathbf{y}
}
\qquad
1\times V
```

```math
\boxed{
\frac{\partial L}
{\partial W_{\text{out}}}
=
\mathbf{h}^{T}
(\hat{\mathbf{y}}-\mathbf{y})
}
\qquad
d\times V
```

```math
\boxed{
\frac{\partial L}
{\partial \mathbf{h}}
=
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^{T}
}
\qquad
1\times d
```

For every context word,

```math
\boxed{
\frac{\partial L}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^{T}
}
\qquad
1\times d
```