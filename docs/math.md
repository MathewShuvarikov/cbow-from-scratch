# Continuous Bag of Words — Mathematics

## 1. Problem formulation

The goal of Continuous Bag of Words (CBOW) is to predict a target word from the words surrounding it.

For each training example, we take a context of $C$ words, map them to their input embeddings, average those embeddings, and use the resulting hidden representation to predict the target word.

The trainable parameters are:

- the input embedding matrix $W_{\text{in}}$;
- the output matrix $W_{\text{out}}$.


## 2. Notation

Let $\mathcal{V}$ be the vocabulary and let

```math
|\mathcal{V}| = V
```

where $V$ is the vocabulary size.

Let $d$ be the embedding dimension.

The input embedding matrix is

```math
W_{\text{in}}
\in
\mathbb{R}^{V \times d}.
```

The output matrix is

```math
W_{\text{out}}
\in
\mathbb{R}^{d \times V}.
```

For one training example:

- $C$ is the number of context words;
- $\mathbf{x} \in \mathbb{R}^{1 \times V}$ is the context vector;
- $\mathbf{h} \in \mathbb{R}^{1 \times d}$ is the hidden representation;
- $\mathbf{z} \in \mathbb{R}^{1 \times V}$ is the vector of logits;
- $\hat{\mathbf{y}} \in \mathbb{R}^{1 \times V}$ is the predicted probability vector;
- $\mathbf{y} \in \mathbb{R}^{1 \times V}$ is the one-hot target vector.


## 3. Forward pass

### 3.1 Context representation

Suppose the context contains words

```math
w_1, w_2, \dots, w_C.
```

Each context word corresponds to one row of $W_{\text{in}}$.

For the $c$-th context word,

```math
\mathbf{e}_c
=
W_{\text{in}}[w_c].
```

Each embedding has shape

```math
\mathbf{e}_c
\in
\mathbb{R}^{1 \times d}.
```

The same context can be represented by a sparse context vector

```math
\mathbf{x}
\in
\mathbb{R}^{1 \times V}.
```

Its $j$-th component is

```math
x_j
=
\frac{
\text{count of vocabulary word } j \text{ in the context}
}{C}.
```

Therefore:

- words not present in the context have $x_j=0$;
- a word appearing once has $x_j=1/C$;
- if a word appears several times, its weight increases accordingly.


### 3.2 Context aggregation

CBOW averages the context embeddings:

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

Using the context vector $\mathbf{x}$, the same operation can be written in matrix form:

```math
\boxed{
\mathbf{h}
=
\mathbf{x}W_{\text{in}}
}
```

Indeed, multiplication by $\mathbf{x}$ selects the context rows of $W_{\text{in}}$ with the correct averaging weights.

The hidden representation has shape

```math
\mathbf{h}
\in
\mathbb{R}^{1 \times d}.
```


### 3.3 Output logits

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

For a particular vocabulary element $i$,

```math
z_i
=
\sum_{r=1}^{d}
h_r
W_{\text{out},ri}.
```


### 3.4 Softmax

The logits are converted into probabilities using softmax:

```math
\hat{y}_i
=
\frac{e^{z_i}}
{\sum_{k=1}^{V}e^{z_k}}.
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


#### Softmax derivative for $i=j$

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

Factor out $e^{z_i}$:

```math
\frac{\partial \hat{y}_i}
{\partial z_i}
=
\frac{
e^{z_i}(S-e^{z_i})
}
{S^2}.
```

Split the fraction:

```math
\frac{\partial \hat{y}_i}
{\partial z_i}
=
\frac{e^{z_i}}{S}
\left(
1-\frac{e^{z_i}}{S}
\right).
```

Therefore,

```math
\boxed{
\frac{\partial \hat{y}_i}
{\partial z_i}
=
\hat{y}_i(1-\hat{y}_i)
}
```


#### Softmax derivative for $i \neq j$

For

```math
\hat{y}_j
=
\frac{e^{z_j}}{S},
```

take the derivative with respect to $z_i$, where $i\neq j$.

Since $e^{z_j}$ does not depend on $z_i$,

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

Hence,

```math
\boxed{
\frac{\partial \hat{y}_j}
{\partial z_i}
=
-\hat{y}_j\hat{y}_i
}
```

for $i\neq j$.

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

Since $\mathbf{y}$ is one-hot, only the target word contributes to the loss.

If $t$ is the index of the target word,

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

We want to compute

```math
\frac{\partial L}
{\partial z_i}.
```

Every softmax output $\hat{y}_j$ depends on $z_i$, so the chain rule gives

```math
\frac{\partial L}
{\partial z_i}
=
\sum_{j=1}^{V}
\frac{\partial L}
{\partial \hat{y}_j}
\frac{\partial \hat{y}_j}
{\partial z_i}.
```

For $j=i$,

```math
-
\frac{y_i}{\hat{y}_i}
\hat{y}_i
(1-\hat{y}_i)
=
-y_i(1-\hat{y}_i).
```

For $j\neq i$,

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
\frac{\partial L}
{\partial z_i}
=
-y_i(1-\hat{y}_i)
+
\sum_{j\neq i}
y_j\hat{y}_i.
```

Since

```math
\sum_{j=1}^{V}y_j
=
1,
```

we have

```math
\sum_{j\neq i}y_j
=
1-y_i.
```

Hence,

```math
\frac{\partial L}
{\partial z_i}
=
-y_i(1-\hat{y}_i)
+
\hat{y}_i(1-y_i).
```

Expanding,

```math
\frac{\partial L}
{\partial z_i}
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
\frac{\partial L}
{\partial z_i}
=
\hat{y}_i-y_i
}
```

Therefore, in vector form,

```math
\boxed{
\frac{\partial L}
{\partial \mathbf{z}}
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
\frac{\partial L}
{\partial \mathbf{z}}.
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

Shape check:

```math
(d\times 1)(1\times V)
=
d\times V.
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

Shape check:

```math
(1\times V)(V\times d)
=
1\times d.
```


### 4.4 Gradient w.r.t. input embeddings

From

```math
\mathbf{h}
=
\frac{1}{C}
\sum_{c=1}^{C}
W_{\text{in}}[w_c],
```

each occurrence of a context word contributes

```math
\frac{\partial \mathbf{h}}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}.
```

Therefore, for each occurrence $w_c$,

```math
\frac{\partial L}
{\partial W_{\text{in}}[w_c]}
=
\frac{1}{C}
\frac{\partial L}
{\partial \mathbf{h}}.
```

Substituting the gradient with respect to $\mathbf{h}$,

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

If the same vocabulary word appears more than once in the context, the gradient contributions for that row are added together.

Rows of $W_{\text{in}}$ corresponding to words not present in the current context receive zero gradient.


#### Full matrix gradient for $W_{\text{in}}$

Using the matrix representation

```math
\mathbf{h}
=
\mathbf{x}W_{\text{in}},
```

the gradient with respect to the entire input embedding matrix is

```math
\frac{\partial L}
{\partial W_{\text{in}}}
=
\mathbf{x}^{T}
\frac{\partial L}
{\partial \mathbf{h}}.
```

Since

```math
\frac{\partial L}
{\partial \mathbf{h}}
=
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^{T},
```

we obtain

```math
\boxed{
\frac{\partial L}
{\partial W_{\text{in}}}
=
\mathbf{x}^{T}
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^{T}
}
```

Shape check:

```math
(V\times 1)(1\times d)
=
V\times d,
```

where

```math
(1\times d)
=
(1\times V)(V\times d).
```

This full matrix formula is equivalent to the row-wise result above.


## 5. Parameter update

Using gradient descent, the output matrix is updated as

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

### 6.1 One training example

```math
W_{\text{in}}
\in
\mathbb{R}^{V\times d}
```

```math
\mathbf{x}
\in
\mathbb{R}^{1\times V}
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

The main gradients are

```math
\boxed{
\frac{\partial L}
{\partial \mathbf{z}}
=
\hat{\mathbf{y}}-\mathbf{y}
}
```

with shape $1\times V$,

```math
\boxed{
\frac{\partial L}
{\partial W_{\text{out}}}
=
\mathbf{h}^{T}
(\hat{\mathbf{y}}-\mathbf{y})
}
```

with shape $d\times V$,

```math
\boxed{
\frac{\partial L}
{\partial \mathbf{h}}
=
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^{T}
}
```

with shape $1\times d$, and

```math
\boxed{
\frac{\partial L}
{\partial W_{\text{in}}}
=
\mathbf{x}^{T}
(\hat{\mathbf{y}}-\mathbf{y})
W_{\text{out}}^{T}
}
```

with shape $V\times d$.


### 6.2 Batch of training examples

For a batch of $N$ training examples, stack the context vectors into the matrix

```math
X
=
\begin{pmatrix}
\mathbf{x}_1 \\
\mathbf{x}_2 \\
\vdots \\
\mathbf{x}_N
\end{pmatrix}
\in
\mathbb{R}^{N\times V}.
```

The hidden representations are

```math
H
=
XW_{\text{in}},
```

where

```math
H
\in
\mathbb{R}^{N\times d}.
```

The logits are

```math
Z
=
HW_{\text{out}},
```

where

```math
Z
\in
\mathbb{R}^{N\times V}.
```

The target and prediction matrices are

```math
Y,
\hat{Y}
\in
\mathbb{R}^{N\times V}.
```

If the loss is averaged over the batch,

```math
L
=
\frac{1}{N}
\sum_{n=1}^{N}
L_n,
```

then

```math
\boxed{
\frac{\partial L}
{\partial Z}
=
\frac{1}{N}
(\hat{Y}-Y)
}
```

The gradient with respect to the output matrix is

```math
\boxed{
\frac{\partial L}
{\partial W_{\text{out}}}
=
\frac{1}{N}
H^{T}
(\hat{Y}-Y)
}
```

The gradient with respect to the hidden representations is

```math
\boxed{
\frac{\partial L}
{\partial H}
=
\frac{1}{N}
(\hat{Y}-Y)
W_{\text{out}}^{T}
}
```

The full gradient with respect to the input embedding matrix is

```math
\boxed{
\frac{\partial L}
{\partial W_{\text{in}}}
=
\frac{1}{N}
X^{T}
(\hat{Y}-Y)
W_{\text{out}}^{T}
}
```

The matrix $\hat{Y}-Y$ already contains the gradient with respect to every logit of every training example.

We do not average first over classes and then over examples. Aggregation over the batch occurs naturally when the gradients of the shared parameter matrices are computed.
