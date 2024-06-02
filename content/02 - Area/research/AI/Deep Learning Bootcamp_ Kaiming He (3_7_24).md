[link](https://www.youtube.com/watch?v=D_jt-xO_RmI)
# Why deep learning 
Deep learning is representation learning
So, why representation learning?
A good representation of the raw data is the key to solve complex problems. 
- data: pixels, words, waves, states, DNA, molecules.
The abstraction of different raw data makes computer solve it more easily. 
- bad representation: 3^361 states for go; 256^3\*500\*500 for a 500 by 500 colored image. 
- Good representation: use CNN, for example, to compress/abstract/conceptualize the data
	- keep the same monto-carlo and reinforcement learning algorithms fixed, AlphaGo can do much better by just increasing the power of the NN that learns the representation 

# How to Represent an Image
## Before NN:
Algorithms use edge, and on top of it, orientation, and histogram, and clusters to represent an image. 
- it requires deep expertise to devise representations to solve only simple problems 

## Deep learning:
- multi-level representation: **simple modules composed to a complex function** 
- use data to auto-learn the modules (weights) so that human expertise is replaced. 

## Learning image representation 
### LeNet (1998)
- CNN: translation-invariance
- introduces foundational elements: 
	- convolution
	- pooling
	- fully-connected layer
	- trained by BackProp
- Convolution: local connections (an output neuron is only connected to a small subset of the input neurons (in contrast to FCNN)) + spatial weight-sharing (the weights are shared among all output neurons in their connections with input neurons)
- pooling: sub-sampling:
	- reduce the feature map size. 
	- achieve local invariance (reduce the influence of noises)
Step by step of LeNet:
1. input: 32 \* 32 \* 1 image
2. 1st set of convolution layers, kernel size 5 x 5 => 28 x 28 x 6
3. Subsampling: 14 x 14 x 6
4. 2nd set of convolution layers, kernel size 5 x 5 => 10 x 10 x 16
5. Subsampling: 5 x 5 x 16, then flatten it 
6. 1st set of fc layer (400 x 120): 120 
7. 2nd set of fc layer (120 x 80): 80
8. 3rd set (80 x 10): 10 

### AlexNet (2012)
Scale everything up: data (1.28 million images), model (60 million parameters)
To reduce overfitting:
- data augmentation
- dropout 
Deploy GPU training: data distribution and model distribution

Deeper and Wider (more channels, richer set of features) Neural Network

Use ReLU: better gradient propagation
Deeper

### Visualizing and Understanding Convolution Networks (2013)
- "What input can produce certain feature"
- Understanding representations by visualization
	- set a one-hot feature map
	- back prop to pixels 
- As we progress through the layers, the deeper the layer is, the higher-level representation it has (hard for human expert)
- Also, it leads to [Transfer Learning](#Transfer%20Learning)

### Transfer Learning  Paradigm
- Pre-train on large-scale data to obtain a general representation
- fine-tune on small-scale, more specialized data 

### VGG nets (2014)
- Very deep ConvNets
- simple, modularized design: only 3 x 3 kernels
- stack the same modules, just increases the depth 
-  in this "controlled" way, it provides clear evidence that the deeper, the better
- The catch: it uses stage-wise training: it first trains a 11 layer NN, then add another 2 layers and fine-tune it, and so on so forth: against the end-to-end training philosophy. To fix it, better initialization is needed.  

### Network Initialization (2015)
Intuitively, deep NN suffers from error magnification as the number of layers grow. Formally, during the backprop stage, the gradient of the initial layers will either have 
- the vanishing gradient problem: the network will converge prematurely, or
- the exploding gradient problem: the model will soon diverge
Xavier initialization: help preserve the variance, but assuming the activation function is linear
Kaiming initialization: one half of the Xavier init, to account for the fact that ReLU cuts the signal in half. 
- only works for ReLU; for sigmoid or tanh, use Xavier
- it helps VGG train end-to-end.
- but yet deeper network needs solutions like [ResNet (2015)](#ResNet%20(2015))

### GoogLeNet/Inception (2014)
Deep and economical ConvNets 
- Multiple branches 
- 1x1 bottleneck
	- Each 1x1 convolution filter is essentially a single neuron connected to all the channels at a specific spatial location in the input.
	- When this filter is applied to the input, it performs a weighted sum across all channels at that location, learnsing to capture specific features by combining information across the input channels.
	- By using multiple 1x1 filters, the layer can transform the input into a new space with a reduced number of channels.
- 1x1 shortcut 
But this model breaks the assumption of initialization methods. So they look at normalization methods

### Normalization modules (2015)
Network initialization is a special case of normalization. Input normalization is only valid at the beginning. 
**normalization modules**: keep signals normalized throughout training for all layers: implemented as a normalization layer (normalization + a linear transform)
- BatchNorm
- LayerNorm (the default), etc. 
It helps in mitigating the problem of internal covariate shift, where the distribution of each layer's inputs changes during training, making it difficult to train deep networks efficiently. But, it doesn't inherently provide a mechanism to combat the exponential shrinking of gradients.

### ResNet (2015)
Deep residual learning: enable networks w/ hundreds of layers by **identity shortcut**

Motivation: good init + normalization should enable deeper models, and we could just stack more and more layers. However, the performance degrade after 20 layers, and it is not b/c of overfitting (the training performance is also bad). It is b/c the deep network is hard to train: the optimizer could not find a good solution. **Vanishing Gradients Problem**, for example. 

 A thought experiment: if we have a good performance shallower network, how can we have a similar performance in a deeper one: we can squeeze in identity mapping layers into the shallower one to make it deeper.

Solution: Residual Learning 
- Instead of fitting H(x), a residual block fits F(x), and through skip connection, the output is F(x) + x (+ is element-wise addition), where F(x) can be seen as the "change" to the input, called residual change. 
- By setting the weight layer in the residual block to be 0, the block is essentially a identity mapping of the input x. If that is desired, then training it to set it to be all 0s is easy. Therefore, the residual block "encourages" the optimizer to find solutions to small/incremental changes to the input
- So, by just stacking residual blocks, we could build very deep NN. 
- From now on, people shift from designing in the layer-level to block-level

# Learning Representations for Sequences
## RNN 
- temporally weight-sharing (CNN shares weights spatially): the current input token produces a hidden state, which is passed as input along with the next token. The weight is shared across time steps. 
- Deeper RNN: treat the previous layers effectively as input tokens: the neurons in the deeper layers will depend on both previous layers and previous neurons in the same layer

### Example: Google's Neural Machine Transition System (2016)
- Stacked LSTM units 
- go deep b/c of residual connections 

## CNN for sequence modeling 
- use 1D kernel, and conv along 1-D sequence
- the kernel does not conv the current token with future tokens, only with past tokens (causal convolution) to preserve temporal order, especially in tasks where future information should not influence the current prediction (like time series forecasting or language modeling)
- to have more context, the network should go deeper: in deeper layers, the neuron will have access to more "history"

### Example: DeepMind's WaveNet for audio Generation (2016)
- casual conv with dilation: To expand the receptive field without excessively increasing the number of parameters, dilated convolutions can be used. In dilated convolutions, the kernel is 'stretched' over a larger number of input tokens by skipping inputs at a regular interval.
 
### RNN vs CNN
- RNN neuron can see the full context, but b/c it is not feedforward (it needs to wait on previous neurons), it is not GPU-friendly 
- CNN has limited context, but it is feedforward 
- **Attention**
	- Every node can see every other node 
	- Feedforward 
## Transformer (2017)
- uses attention 
- unlike in FC where every connection is associated with a trainable parameter, attention is parameter-free
- all the layers that have parameters are feed-forward 
	- used to obtain the Q, K, V from the token embeddings and transform them in the MLP block to some other representation
- to go deeper: residual connection + layerNorm

### Example: Generative Pre-trained Transformer (GPT)
### Example: AlphaFold
- 48 transformer blocks
- input: amino acid sequences
- output: protein structures
### Example: Vision Transformer (ViT, 2020)
- Divide a 2D image in non-overlapping 1D sequence of image patches and serve them as inputs to transformers
- 1st time to demonstrate that transformer can be a general structure for many types of data. 