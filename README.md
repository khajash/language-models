# language-models
Repo building and training small language models. Compares recurrent models with transformers

## Language Model
$$
\begin{equation}
L_1(\mathcal{U}) = \sum_i \log P(u_i |u_{i-k}, \dots, u_{i-1};\Theta)
\end{equation}
$$

## Dataset
I am using WikiText2 from `torchtext`. It is already preprocessed having the rare words replaced with the `<unk>` token...

## Transformer Language Model
The Transformer Model architecture used here is similar to that employed in (Liu et al., 2018) and the original GPT network
<img src="./imgs/TransformerDecoderDiagram.png" width="75%" >



### Learning Rate Schedulers
- StepLr
- Inverse Square Root with Warm-up
- Cosine with Warm-up (to be implemented)


---
### Transformer Architecture References

**TODO: finish and refine notes!!**
#### Attention is All You Need
- **Architecture**
  - Encoder-Decoder Transformer model for Machine Translation
  - At beginning of both encoder and decoder models is an embedding layer for each relevant vocabulary and a sinusoidal positional embedding layer. 
    - Embedding multiply by $\sqrt{d_{\text{model}}}$
  - (512 dimensional states with 8 attention heads)
  - **Encoder**
    - Stack of $N = 6$ identical layers consisting of two sublayers: self-attention and feed-forward network. 
    - Around each sublayer is a residual connection.
    - Following the residual connection is layer normalization.
  - **Decoder**
    - Stack of $N = 6$ identical layers consisting of three sublayers: masked self-attention, encoder-decoder attention, and feed-forward network. 
    - Around each sublayer is a residual connection.
    - Following the residual connection is layer normalization.
  - **Position-wise Feed-Forward Networks**
    - Two linear layers with a ReLU activation in between
        $$
        \begin{equation}
        \text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2
        \end{equation}
        $$
    - Input and output dims: $d_{\text{model}}=512$
    - Inner-layer dims: $d_{ff}=2048$
  - **Optimization**
    - Adam optimizer with $\beta_1 = 0.9, \beta_2=0.98$ and $\epsilon=10^{-9}$
    - Used linear warmup with inverse square root decay afterwards
  - Used **bytepair encoding (BPE)** vocabulary with target vocabulary of ~37000 tokens

#### GPT 
- **Architecture**
  - 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads)
  - Position-wise feed forward networks - used 3072 dimensional inner states
- **Optimization**
  - Adam optimizer with max lr of 2.5e-4
  - **lr scheduler**: increased linearly from zero over the first 2000 updates, annealing to 0 using a cosine schedule
  - 100 epochs using minibatches of 64 randomly sampled, contiguous sequences of 512 tokens
- **Weight initialization of $N(0, 0.02)$** is sufficient b/c Layer norm is used throughout
- Used **bytepair encoding (BPE)** vocabulary with 40,000 merges
- Residual, embedding and attention **dropouts with rate of 0.1** for regularization
- Modified version of L2 regularization with $w=0.01$ on all non-bias or gain weights
- GELU activation function
- Used learned position embeddings instead of sinusoidal


---
### Sources
- Attention is All You Need
- GPT
- 