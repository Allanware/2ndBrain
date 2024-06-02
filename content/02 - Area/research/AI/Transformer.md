- [Attention Is All You Need](Attention%20Is%20All%20You%20Need.md)
- [Transformer](Neural%20Networks_Zero%20to%20Hero.md#Transformer%20block), [Self-attention](Neural%20Networks_Zero%20to%20Hero.md#Self-attention) from Andrej's Neural Networks: Zero to Hero
- [Another lecture by Andrej](https://www.youtube.com/watch?v=XfpMkf4rD6E)
- [Lilian Weng's blog on Transformer family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)
# Attention
- [Attentional Interfaces: attention between NNs](Modeling%20Sequence.md#Attentional%20Interfaces%20attention%20between%20NNs)
- Dive into Deep Learning Chapter 9, 10, 11

# The Illustrated Transformer
- [article](https://jalammar.github.io/illustrated-transformer/)

The biggest benefit of Transformer comes from how it lends itself to *parallization*, unlike the sequential RNN. 

The encoding component is a stack of 6 encoders, and the decoding component is a stack of 6 decoders. 
## Encoder
Each encoder have a self-attention layer and a fully connected module. The self-attention layer helps the encoder look at other words in the input sentence as it encodes a specific word.

The bottom-most encoder takes a list of embeddings (the subsequent encoder will takes its outputs):
- the size of the list is a hyperparameter: it is usually the length of the longest sequence. 
- shorter sequences usually are padded with `-inf`s to the end (post padding) to match the size of the list so that the padding vectors are effectively ignored: [Math tricks](Neural%20Networks_Zero%20to%20Hero.md#Math%20tricks)
- Note: the "future tokens" (right context of the current token) are not masked with `-inf` so that every token can attend to every other token in the input sequence. 

The outputs of the self-attention layer are fed to the **same** feed-forward neural network independently, where each vector can be executed in parallel. 

## Self-Attention step by step
1. create three vectors from each of the encoder’s input vectors: a Query vector, a Key vector, and a Value vector by matrix-vector multiplication ($W^Q$, $W^K$, $W^V$ multiply with the input vector $\mathbf x_i$)
2. calculate a score for each input vector (the attention score from $\mathbf x_i$ to every other vector) by taking the dot product of the query vector with the key vector of each input vectors ($q_i$ with $k_1, ..., k_n$)
3. divide the scores by the square root of the dimension of the key vectors, and apply softmax(the softmax of 112 and 96 is 0.999999999 and 0.00000001; if divide them by 8, the resulting softmax will be 0.88 and 0.12)
4. get a weighted sum of value vectors, weighted by the softmax score. 

## Multi-head attention 
It improves the vanilla attention int two ways:
- It enables the model to attend to different positions in multiple ways. 
- With multiple sets of Query/Key/Value weight matrices, it project the inputs into different representation subspaces.
![](../../../multi-headed-self-attention.png)

Note: the self-attention and the FC module is **agnostic** to the sequence length. 
## Position Encoding
To encode position of the input token the transformer adds a vector (of same size to the embedding) to each input embedding. It can be some function of `sin` and `cos` (interleaving `sin` and `cos` in the original paper) because it can scale to unseen lengths of sequences.

The length of position encoding may limit the maximum length of the input, which is the only thing in transformer that is dependent on input sequence length. 

## Residual and layerNorm
After each self-attention, FC, and encoder-decoder attention module (only in decoder), the inputs to those modules are add added back to the output and are normalized based on [LayerNorm](Neural%20Networks_Zero%20to%20Hero.md#LayerNorm). 

## Decoder
The output from the final encoder block consists of a sequence of encoded vectors, each of which corresponds to an input token and reflects not only the individual token but also its relationship with all other tokens. We call them **encoder outputs** or **encoder context vectors**. 

These are fed into the cross-attention module in all decoder blocks as Keys and Values, transformed by distinct linear transformations ($W_{K_i}$, $W_{V_i}$ for $i$ from 1 to num_blocks. )

The first decoder at start takes the embedding of the `<sos>` token, added with the positional encoding. 
### Self-Attention
Each output token can attend to all previous tokens, while future tokens are masked. 
### Cross-Attention
The query (Q) coming from the previous decoder layer's output, and the keys (K) and values (V) coming from the encoder output. This allows each position in the decoder to attend over all positions in the input sequence. 

The outputs are fed in parallel to Feedforward layers. Each module is again augmented with residual connections and layerNorm. 

### The Final linear and Softmax Layer
The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders into a logits vector, the length of which is decided by the size of the vocabulary. The softmax layer then turns those scores into probabilities. 

The output token is then passed back to the first decoder block, after being embedded and combined with positional encoding. 

# Attention? Attention!
- [Lilian Weng's blog on Attention family]
Human visual attention allows us to adjust the focal point to focus on a certain region with “high resolution” while perceiving the surrounding image in “low resolution”,  and do the inference accordingly.

The attention mechanism is introduced to address the limitation of the Seq2Seq model that has fixed context length, proposed by this [paper](https://arxiv.org/pdf/1409.0473.pdf) in the context of machine translation. With the help of the attention, the dependencies between source and target sequences are not restricted by the in-between distance anymore!
### Attention in Seq2Seq
How the decoder attend to all encoders step by step is addressed here: [Attention Mechanism in Decoder](Modeling%20Sequence.md#Attention%20Mechanism%20in%20Decoder).

#### Math formulation of attention in Seq2Seq
Input and output: $$\begin{aligned}
\mathbf{x} &= [x_1, x_2, \dots, x_n] \\
\mathbf{y} &= [y_1, y_2, \dots, y_m]
\end{aligned}$$The encoder is a bidirectional RNN with a forward hidden state $\overrightarrow{\boldsymbol{h}}_i$ and a backward one $\overleftarrow{\boldsymbol{h}}_i$. Encoder hidden state: $$\boldsymbol{h}_i = [\overrightarrow{\boldsymbol{h}}_i^\top; \overleftarrow{\boldsymbol{h}}_i^\top]^\top, i=1,\dots,n$$
The decoder hidden state $\boldsymbol{s}_t=f(\boldsymbol{s}_{t-1}, y_{t-1}, \mathbf{c}_t), t = 1, ..., m$, where
$$\begin{aligned}
\mathbf{c}_t &= \sum_{i=1}^n \alpha_{t,i} \boldsymbol{h}_i & \small{\text{; Context vector for output }y_t}\\
\alpha_{t,i} &= \text{align}(y_t, x_i) & \small{\text{; How well two words }y_t\text{ and }x_i\text{ are aligned.}}\\
&= \frac{\exp(\text{score}(\boldsymbol{s}_{t-1}, \boldsymbol{h}_i))}{\sum_{i'=1}^n \exp(\text{score}(\boldsymbol{s}_{t-1}, \boldsymbol{h}_{i'}))} & \small{\text{; Softmax of some predefined alignment score.}}
\end{aligned}$$
The $score$ function  [

]