By Andrej Karpathy
- [Video](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Github](https://github.com/karpathy/nn-zero-to-hero)
# 1: building micrograd
- [Micrograd](https://github.com/karpathy/micrograd): a small autograd engine (backprop + dynamically built DAG that only works over scalars) + some NN API on top of it
	- backprop is a general and effective way to do autodiff on any computation DAG. NN weight training is one of its applications. Because the backprop is so generic (you can stack any number of differentiable operations), **it makes the development of NN architectures really easy and modularized**. For example, you can add new parameters and introduce new layers as long as they can be inserted into the existing forward pass pipeline: you do the insertion in the forward pass, and as long as the parameters are tracked by pytorch or any NN implementation, the backprop will take care of the backpass for you. 
- During the backprop process, every gradient calculation should only chain two values together: 
	- the global gradient: dL/dParent (already calculated in the last step), times 
	- the local gradient: dParent/dnode.
- If a node is used multiple times during one forward pass, the overall gradient during one backward pass is the accumulated result of gradient calculations wrt this node. 
- [A simple math model for neuron](https://cs231n.github.io/neural-networks-1/#bio)
- really think about the mathematical expression and its meaning behind derivative/gradient: the influence of this weight towards the output/loss, measured by the ratio of change in output (do) and the very small change (infinitely close to 0) to this weight (dw). If the gradient of this weight is positive, then increasing the weight will increase the output (positive influence)
- to build the backprop DAG, we can use topological sort: it recursively visits nodes and put them into a list based on invariant: every node won't be added to the list until its children (and by recursion, all descendants) are added to the list. Once we have the list, we just call backward prop using the reverse order of the list.  

## Backprop in pytorch
- every torch tensor has a .data and a .grad attribute and use .item() to gets the value (strip out the tensor)
- nn.module has .parameters() that returns all the parameters.

## Build a  2-layer MLP 
- again, NN is just a special application where the micrograd can do forward and backward passes on. 
- in the forward pass, the .data attribute of non-parameters will be updated 
- in the backward pass, the gradients of all nodes (.grad) will be updated with respect to loss **AND** the .data attribute of parameters will be nudged slightly in the step of learning rate towards the direction of negative gradient (to minimize the loss)
- Don't forget to flush out the gradients before go to the next iteration   
	- setting the grad of parameters to be 0. 
	- discarding the grad of intermediate values (weighted sums, ws after activation). In Micrograd, interm values (.data) are freshly created (their grad is set to 0) and destroyed when each forward pass begins. In contrary, the .data and .grad are **persistent** for parameters (that is why we need to reset the grads for parameters while keeping the .data)
	- There are lots of values and grads, the only we care are those of the parameters.  

# Makemore 1.1: Bigram model
- [makemore](https://github.com/karpathy/makemore): make more of something (what something is is defined by the training dataset): an autoregressive (a type of statistical model used for understanding and predicting time-series data, where the key assumption is that the current value of the series is based on a combination of past values) character-level language model. 
	- eg dataset: [name.txt](https://github.com/karpathy/makemore/blob/master/names.txt)
		- 32k names 
- Info packed within a name for a char-level model: the name "emma" tells us: it is likely to start with 'e'; 'm' is likely to follow 'e', ..., 'END' is likely to follow "emma"
- The context it considers is just a sliding window of two chars including "<S\>" and "<E\>"

## Training 
- Iterates through all words to get all the bigrams and count and store the counts in a dictionary
- Another representation/data structure: a 2d tensor array using pytorch, where the rows are the 1st chars of the bigrams, the columns are the 2nd chars. Since our alphabet is of size 28 (26+2), the shape of the tensor array is 28 x 28. The individual entries store the counts of the composed bigrams.

## Sampling
- convert raw counts to probabilities (probability distributions) so that we could do sampling. 
	- torch.multinomial
		- use torch.Generator() to make the result deterministic
		- takes in a tensor of weights (or a probability distribution if they are normalized to 1)
		- by default, replacement=False 
	1. convert the first row of the tensor array to a probability distribution that models the probability distribution of the first char after the start token
	2. Sample from it
	3. Takes the sampled char and goes to the corresponding row and convert that row to a probability distribution and sample from it
	4. break until you sampled the end token

### Code Optimization
- Instead of normalizing whenever sample a char, we can normalize all the rows before sampling
	- use torch.sum()
		- dim: the index to the dimension to be reduced/summed
		- keep_dim=False: by default, not only it reduces the `dim` to 1 (28x28 to 1x28 if `dim=0`), but also it squeezes the dimension out (28). If set to True, it will keep the shape to (1x28)
	- normalize the tensor array by the result of torch.sum(): 
		- [broadcasting semantics/eligibility](https://pytorch.org/docs/stable/notes/broadcasting.html#general-semantics)
			- Each tensor has at least one dimension.
			- When iterating over the dimension sizes, starting at the trailing dimension (move dimensions of the tensor with fewer tensor to the right), the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
			- when broadcasting, the smaller dimension gets copied to align with the larger dimension to get elementwise-ops
		- broadcasting caveat: understand what broadcasting copies before do broadcasting: broadcast a tensor of shape (28, 1) to (28,28) is not necessarily the same as broadcasting a tensor of shape (1,28) or (28) to (28,28). 
			- In the former case, a column tensor is copied along the columns horizontally; in the latter case, a row tensor is copied vertically 
## Log-likelihood
- Likelihood: the product of the probabilities of all the bigrams 
- Log-likelihood: the sum of log probabilities: the higher llh (log-likelihood) is, the higher lh (likelihood) is
	- caveat: if the likelihood of a bigram is 0 (count is 0), then the llh is `-inf`. 
		- use model smoothing: add certain amount of  "fake counts" to all counts so that no llh is `-inf` and in principle every name can be generated. 

### MLE
- The MLE (maximum likelihood estimation) approach maximize the likelihood (or llh) of the data w.r.t model parameters 
- (Averaged) Negative log likelihood as loss function: since we want to minimize the loss, and log-likelihood <= 0 , we negate the llh and tries to make the nllh smaller. 

# Makemore 1.2: Bigram + NN

## Convert to NN
- turn it into a classification problem: give the first char of the bigram (coded as an integer), classify the class of the second char (also coded as an int) using a NN. 
### Input to NN: one-hot encoding
- `torch.nn.functional.one_hot` takes in a tensor of integers (the encoded first chars) and `num_classes`. In our case, `num_classes` = 27. So the input layer is of shape (27, 1)

### Output layer: 
- Connected by the Weight Matrix (27 x 27), the output layer has 27 output neurons. We can interpret the raw value of those neurons as "logits": "log counts" (-inf to inf). We then take the `exp` of those logits to get "counts" (0 to inf). Finally, we normalize the "counts" by their sum to get "probability" (0 to 1, summed to 1), so we can interpret the final output as probability distribution. The final two steps of transformation of the logits is called **softmax**. 
	- all of the above ops are differentiable

## Training 
### Loss function 
- built on top of the result of softmax, we can take the output "probability" for the *true label* as likelihood, and compute log-likelihood and negative llh. 
- compute the sum/avg of negative llh for all bigrams as `loss` (the root of pytorch's computation graph)
### Backward Pass
- before backward pass, make sure to zero-out the gradients of the Weight matrix (setting it to `None`). We don't need to do it for intermediate neurons b/c they are only used to computer gradients of the weights and storing them is a waste of space. To make clear of that, we need to set `require_grad=True` for weights and biases, so that pytorch will retain gradients for them. 
- call `loss.backward()`
	- after the call. `W.grad` will be updated
- update weights by `W.data -= step_size * W.grad`

## Result
- Since the previous approach is already the optimal solution to maximize the likelihood, the gradient descent method is just another way to iteratively come up with (roughly) the same solution. However, the NN is much, much more flexible: the logits, softmax, and loss can stay the same (these are portable, along with data, backward prop, weight updates). The only thing changes is how to get from input data to the logits. 
- Also, what if not bigram, what if is trigram or 10grams: impossible to build the probability table.
  
## Notes 
- The input tensor is one-hot encoding, so only one row of the weight matrix is actually multiplied with the input to get the logits, just as what we did in the 1st approach where we index to the nth table where n is the index to the first char (the input). 
- In the 1st approach, we add some constant value to the count table to do smoothing. As the constant value grows, the probability distribution become uniform. In the NN approach, we could do the same thing through weight regularization: if we make weight to be 0, then the logits will all be 0, and probability/likelihood will be uniform (1/27). If we use the loss function to incentivize the drive of weights to 0, we will achieve the same effects as what smoothing does to 

# Makemore 2: MLP
 - based on the paper *A Neural Probabilistic Language Mode* (2003) by Bengio et al. 
	 - used an embedding of size s (number of neurons fully connected with the input vector)<< 17000.
	 - a hidden layer, fully connected with the final layer (17000 logits), where the most computation happens
	 - a softmax on top of it

## Forward pass
### Input 
- The input context is set to n (3, for example), and we transform the raw data (a list of names) by extracting each name in to a list of (context, label) pairs. In this way, we train a model that classifies inputs into label and use it to generate texts. 
### Embedding 
- two (equivalent) ways to retrieve embeddings:
	1. use one-hot encoding to encode the input char (1x27), and multiply the embedding matrix (27x`dim`) to retrieve the input embedding for this char. The NN implementation of this way is to have a layer of `dim` neurons, and a weight matrix of size 27x`dim`. 
	2. use the index of the input char to index the embedding matrix. Since the context is of size bigger than 1, we could use tensor to index the embedding table to get a tensor of embeddings of all chars in the context. We could even index the table using a ndarray tensor. If we have X contexts of size d, the index is (X,d). And we could use Table\[index\] to get the embeddings for all the example contexts. 

### Hidden layer
- For the sake of example, we set `dim` = 2, and `context_size`=3. So the output from the embedding table will be of shape (num_examples, 3, 2). We need to transform the input to (num_examples, 6) in order to multiply with the hidden layer weights (6, 100), where 100 is the number of neurons in the hidden layer
	- torch.cat will concat a sequence of tensors along a given dimension. We could use `torch.cat([Table[:, 0, :], Table[:, 1, :], Table[:, 2, :]], 1)` to concat the embeddings for the first char, 2nd char, and 3rd char of the context of all examples along the 1st dimension: concat a sequence of three (32,2) tensors along the column.
	- If the context size changes, this won't generalize. `torch.unbind(Table, 1)` will unbind the 1st dimension to a sequence. (num_examples, 3, 2) => a sequence of 3 tensors of shape (num_examples, 2). 
	- There is a better way in below though that makes use of [Torch Internal](#Torch%20Internal)
- Weights w1 shape: (6, 100)
- Bias b1 shape: (100)
- $h = torch.tanh(Table @ W_{1} + b_{1})$
- Output h's shape: (num_examples, 100)

#### Torch Internal
- http://blog.ezyang.com/2019/05/pytorch-internals/
- Torch tensor is stored in memory as a 1d array, no matter what shape it is. `.storage()` will showcase this. Therefore, it isolates storage from the internal attributes (stride, offset, shape) of the view of the tensor. Therefore `.view()` will be really efficient (just some arithmetics done on the attributes) b/c it doesn't involve memory operations. 

### Output layer
- Weight w2 shape: (100, 27)
- Bias b2 shape: (27)
- Logits: $logits=h@W_{2} + b_{2}$
### Loss
- Softmax
- From Softmax, and the associated true labels Y, we get the probabilities corresponding to Y. 
- Instead of getting the product of those probabilities to get likelihood, we sum the log likelihood and negate it to get the negative log-likelihood. 
- We are repeating ourselves from last lecture and reinventing the wheel: [Cross entropy loss in pytorch](#Cross%20entropy%20loss%20in%20pytorch)

### Cross entropy loss in pytorch
- A one step solution to calculate the loss with input logits from the [Output layer](#Output%20layer) and True labels Y: `.cross_entropy()`
	- no intermediate tensors are created, no new memory allocated
	- much simpler backprop (for example, tanh is a congregation of many operations with a one-step backward derivative. If we implement tanh using multiple steps, the backprop will also be in multiple steps)
- more numerically stable: in softmax we take the exponential of logits. If the logit is very large, the result will be `inf`, the resulting probability will be 0s for all others, and `nan` for this very large logit. PyTorch solves it by subtracting the max logit from all logits: if the logit is very negative, the prob will still be stable. 

Then we flush out the gradients before the backward pass, and update the parameters after the backward pass.  
## Training Optimization
### Minibatch
- faster training iteration
- e.g. use `torch.randint` to sample a minibatch
### Learning rate
#### Finding a good initial lr
- find a roughly good range of learning rates 
- interpolate between them, not linearly, but exponentially.
	- 0.001 to 1: linear interpolate using `lre = torch.linspace(-3, 0, 1000)` and get the learning rates using `lr = 10 ** lre` 
- use the 1000 learning rates to train 1000 minibatches and observe the loss curve: find the transition point where the curve goes up. 

#### Learning rate decay 
- as the training progress, we could decay the learning rate 
### Dataset split 
- training split: train the model parameters
- validation split: train the model hyperparameters: size of each layer, number of hidden layers, regularization parameters, etc.
- test split: test the performance of the final model 
- if we train the data using the training split and stabilize the loss, and we run our model on the validation split and get the loss. If the losses are comparable, we are confident that we are not overfitting. We could change the hyperparamerters so that we can have a larger model.  

ML research is similar to research of other kinds: you have to conduct experiment, collect some data and use plots to try to understand the data and verify the hypothesis
## Visualization 
- visualize learning rate vs loss when choosing the initial learning rates 
- visualize step vs loss
- visualize the embeddings


# Makemore 3: activations & gradients, BatchNorm
- Disabling gradient tracking in loss tracking/plotting (no backpass needed) or during inference by explicitly tell pytorch you are not going to run backprop:  
	- `@torch.no_grad()` function decorator 
	- `with torch.no_grad()` context manager, so that torch does not maintain overhead to track their gradients 
	- we could use `.require_grad` to check if a tensor's gradients is tracked. 
## Initialization 
- Inspect the loss after the 1st iteration: in this case it should be $-\log(1/27)$ that reflects a uniform distribution. If the loss is much higher than that, then the NN is initialized to be very confidently incorrect (it chooses the true label with very small probability). Usually it is b/c the logits are having extreme values. Ideally, we want to logits are similar if not the same during initial runs: all 0s for example. 
	- setting the final layer's bias vector to 0, and weight matrix to near 0 (0.1 for example.)
- Saturated acitivation: in the backprop at the tanh "gate", we are propagating back the gradient of `out` using chain rule by effectively scaling the `out.grad` by a factor of $(1 - out.data) ^2$, where `out` is the output neuron of the tanh gate. If we inspect by visualizing the output of the tanh layer (`out`), and it has a lot of -1 (meaning the input to tanh is very negative) and 1 (very positive input): then the gradients are not propagating back. This same issue also applies to `Sigmoid` and `ReLU` where flat regions are present. 
	- If no example can activate this neuron (eg. in ReLU, if the inputs to this ReLU is always negative, then the gradients back to it will never update its downstream): this neuron is dead. This could happen b/c of bad initialization for too high of a learning rate. 
	- to remedy this, we could make the pre-activation close to 0 by initializing the weights and biases of the pre-activation close to 0. 
Bad initialization may not affect shallow networks too much, but it definitely will affect deeper networks. 

## A more systematic Initialization: Kaiming Init

### Motivation
- The output of a matrix multiplication ($y = xw$) will generally have higher variance than both of inputs. Since matrix multiplication involve multiplications and summations, to get some general sense: 
	- Var(A+B)= $Var(A)+Var(B)$
	- Var(A⋅B)=$(E[A])^2⋅σ^2B​+(E[B])^2⋅σ^2A​+σ^2A​⋅σ^2B​$
### Kaiming Init
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (2015)
- The goal of the Kaiming Initialization is to preserve the variance of activations in a NN that uses ReLU (it can also adapt to other non-linear activation functions) by scaling the weights based on the derivation below (under certain assumptions): 
	- It can be [shown](https://pouannes.github.io/blog/initialization/) that: $Var[y_L] = Var[y_1] \left(\prod_{l=2}^{L} \frac{1}{2 n_l} Var[w_l]\right)$. To preserve the variance of each layer's activation, we want to set: 
		- $\forall l, \frac{1}{2 n_l} Var[w_l] = 1$
	- Note: $\frac{1}{2}$ is specific to ReLU, as it is the probability of Y bigger than 0 (the part that produces variance)
- follow from derivation above: W should be initialized as: $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$, where $n_{in}$ is the number of input neurons to multiply W with. Again, 2 is called `gain` and is specific to the activation function NN uses. If not used, then the variances will be scaled/squashed down to 0 as the depth of the NN increases. 
- In the first layer $n_{l} = n_{input}$. The variance of the first layer’s activation also depends on the variance of the input data itself. Together with the convention of input data normalization (normalize input data to have mean 0 and variance 1), when the normalized data is passed through the initialized first layer, the goal of the Kaiming initialization is to maintain the variance of the activation to be 1: the `gain` is here to offset the "squeeze" effect of ReLU by first scaling up the variance of the pre-activation through scaling up weights.  
- It does not talk about bias init. In practice, people just set it to 0. 
- The paper also talks about gradient normalization in the backpass. But we could do without it. 
#### Torch implementation
- `torch.nn.init.kaiming_normal_(tensor, mode, nonlinearity)`
	- tensor: weight W that determines $n_{in}$
	- mode: 
		- fan-in by default: normalize activation during forward pass and preserve the variance of it
		- fa-out: normalize gradient during backpass
	- nonlinearity: the activation function in use that determines the value of `gain`. 
		- tanh's gain: 5/3. B/c tanh also squashes the input as ReLu does, it is taking a value bigger than 1. `torch.nn.init.calculate_gain("tanh")`
- or, `W = torch.randn(n_in, n_out) * gain / (n_in ** 2)`
But now, the initialization process of weights are not so stringent/important with new advances: residual connection, normalization layers, new optimizers (e.g. Adam)

## Batch Normalization
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
### Motivation & Idea
When training in batches, multiplying the batch data with weights and biases will create a distribution. 
- Internal Covariate Shift: The change in the distributions of layers’ inputs presents a problem because the layers need to continuously adapt to the new distribution.
- Saturated activation function (see [[#Initialization]]) with tanh and sigmoid
	- solution: why not make the input to tanh (pre-activation) a unit gaussian so that they are not too extreme. By centering the inputs to activation, the gradient will be higher and thus speeding the learning. 
- Reducing Dependency on Initialization
	- the BatchNorm can be implemented as a small module appended after preactivations and be inserted everywhere in the network.
- Dying ReLU Problem: neurons only output zero for all inputs
	- Batch normalization can mitigate this issue by ensuring that each layer receives inputs with a normalized distribution, reducing the likelihood of consistently negative inputs.

### Algorithm
- Normalize the pre-activations to be unit Gaussian by subtracting the mean and divided by standard deviation (thes operations are differentiable)

However, this along won't work during training, b/c every pre-activation is forced to be unit Gaussian and the backpass won't learn much. So two more learnable parameters (scale and shift) are introduced to each layer that scales and offsets the unit Gaussian (they are initialized to be 1 and 0 so that initially the preactivations are unit Gaussian)

#### Pseudocode of BatchNorm: 
1. $\mu_B \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i$
2. $\sigma^2_B \leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$
3. $\hat{x}_i \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$: normalization done. $\epsilon$ is used to prevent division by zero.
4. $y_i \leftarrow \gamma \hat{x}_i + \beta \equiv BN_{\gamma, \beta}(x_i)$: scale and shift, intr oducing the two new parameters per layer

### Caveat
- Before, although the samples are batched together and fed into the NN, they are independently weighted and transformed into outputs. 
- Now with the BatchNorm module, they are influenced also by the other examples present in the same batch. 
	- The good side effect: each example will shift a bit in each layer in which some noise are added. Thus, effectively, BatchNorm offers some regularization.
	- but still, the idea that examples are not independent is bugging people. Some alternatives have been proposed: layerNorm, groupNorm, instanceNorm, etc. 
- Also, during inference, we are interested in inference inputs individually. To get a single number for mean and standard deviation (in training, we have one for each batch mean and one for each batch sd), we can calculate the mean and sd for the entire training set either:
	- after the training is finished 
	- calculate it during training using a running variable: for example, `bnmean_running = 0.999 * bnmean_running + 0.001 * bnmean_i` , where `bnmean_i` is the `bnmean` of the $i^{th}$ batch, and `0.001` is called momentum (see [[#BatchNorm in pytorch]]). (this is what pytorch does)
- since we are centering the pre-activations, the bias term when calculating the pre-activation does not matter (never gets updated), we can just get rid of it and use the shift parameter of BatchNorm module as the bias. 

### Example: Resnet 
#### Structure of the bottleNeck block 
- declared in `super().__init__()`: 
	1. a `conv` layer: like fully connected layer (without activation), it multiplies weights with inputs. Unlike fc layer, it tries to main the spatial structure of the input images by multiplying the same weights to only patches of inputs independently
		- the `bias` option in the `conv` layer is set to `False`
	2. `norm_layer` which default to be `nn.BatchNorm2d`
	3. `nn.ReLU`
	4. another `conv` layer followed by `BatchNorm` and `ReLU`
	5. another `conv` layer followed by `BatchNorm` 
	6. a residual connection followed by `ReLU`

### Linear layer is pytorch
- https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- options: 
	- fan-in
	- fan-out
	- bias: true or false
- the weights and biases are initialized to to form a uniform distribution: can be overridden using the `init` module of pytorch.

### BatchNorm in pytorch
- options to explain:
	- `momentum`:  default to be 0.1. if batch size is big, then each the statistics of each batch should be very similar. We could therefore set momentum to be big as well. If batch size is small, we should change the default to be some smaller constant. 

## Summing up
- Initialization and BatchNorm aims to control the distribution of pre-activations so that they are homogeneously Gaussian. Failed to do so may cause the activation function (with the builtin non-linearity) to become saturated (leading to vanishing gradient), or to have exploding gradient (if the input to ReLU is very big, in backprop it just passes along the gradient without scaling it). More general, the variances are no longer homogeneous across layers. 
	- an article from Andrej about vanishing gradient/exploding gradient and non-linearities: https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b
- Initialization scales weights and biases so that the variances of pre-activation and therefore activation stays the same. It also prevents at initial iterations the model is very confidentially wrong.
- BatchNorm scales and shifts the pre-activations directly. In practice, people put BatchNorm after the activation layer also works.
- However, initialization fails to control as the depth of the NN and the training iteration increases. As an in-place and real-time control mechanism, BatchNorm prevents the accumulative effects of pre-activation in the forward pass, even when initialization is not carefully configured. 
- But no one likes this layer b/c of the dependence of training data. 

## Visualization & debugging
- activations of each layer after x iterations: we want all the layers are comparable, and are not too concentrated at 0 (lower saturation rate)
- gradients of each layer ...
- gradient of parameters (mean and std): not too concentrated at 0
- update (`lr * grad`) to data ratio: a usual good ratio is 1/1000. And a lot of discrepancy among layers is bad. If most of the layers is below 1/1000, you may want to increase the learning rate. 



# Makemore 4: writing Backprop in tensor
## Manual Backprop a two layer NN 
With a BatchNorm and a softmax layer. The gradients are calculated on all the intermediate values by breaking layers apart (especially the BatchNorm and softmax) into atomic operations. 
- the shape of the gradient should be the same as its corresponding data
- If in the forward pass, the operation is done element-wise. Then in the backprop, applying chain rule is to multiply the local gradient with the previous gradient (our.grad) as dot product to get the current gradient.
	- `dprobs`
- when deriving local gradient, be careful about the implicit broadcasting: use `.shape` to first understand how broadcasting works in this particular case. Again, the shape of the derivative is the same as its data: If in the forward pass, we broadcast the tensor, then in the backpass in calculating the gradient of that tensor,  we need to use `sum` to aggregate the gradients across the broadcasted dimensions (use `keepdim=True` to preserve the reduced dimension if needed). 
	- `dcounts_sum_inv`, `dlogit_maxes`, `b2`, `dbngain`
- Conversely, if the forward pass has an aggregation operation like `sum` , in the backward pass we need to have some broadcasts.
	- `dbndiff2`
- if a node is used more than one time in the graph (on the RHS of an equation more than once), its gradients should be the sum of all "contributions"/sub-gradients. A special case is already mentioned above: the gradient of the broadcasted tensor should be summed along the dimension that is broadcasted. 
	- `dcounts`
- back-propagates gradients from product to its constituents (weights, biases, inner states) is easier than you think: it is just matrix multiplication again (but needs transpose so that the gradients are aligned correctly), the same as what will happen in the scalar case (scalar multiplication). 
	- `h`, `W2`
- It is good practice to consider some toy example before doing the back-propagation. 
- When assigning one tensor variable to another variable, be sure whether you want the other variable stores a shallow (`a = b`) or deep copy (`a = b.clone()`) of the former variable. If you modify`b` and it holds a shallow copy of `a`, changing `b` will also change ``

## Manual Backprop the BatchNorm layer
Without breaking the operations apart into atomic ops: using math to derive the analytical solution to the gradients of the whole layer, so that backprop is more efficient as it calculates the gradient in one go. 
### Bessel's correction in BatchNorm
- An unbiased variance estimation given a sample of size n is the sum of the squared deviations from sample mean divided by *`n-1`*
- But in the original BatchNorm paper, during training they estimate the variance and use it to standardize the pre-activations using the biased variance estimator, the one divided by `n`. However, in test time they use `n-1`tary links:
- WaveNet 2016 from DeepMind https://arxiv.org/abs/1609.03499
- Bengio et al. 2003 M

Built the computation graph. Go backwards one step at a time. Know the fan-outs (what nodes in the forward pass this node influences).
## Manual BackProp the Softmax layer
Similar to the idea in the BatchNorm layer, but apply it to the softmax.
- meaning of `dlogits`: "pull" down the probability of all incorrect classes and "pull" up the probability of the correct class such that the amount of pulling down is the same as the amount up pulling up. (one justification of why we use softmax: so that in the backpass, the gradient is arranged in this nice way)

# Makemore 5: Building a Wavenet
- [the WaveNet paper (2016)](https://arxiv.org/abs/1609.03499)

## Some refactoring on previous code
- instead of plotting the loss of every iteration, plotting the mean of every 1000 iterations  to show a nicer, clear loss decay curve. We can spot a dent in loss when we introduce the learning rate decay. 
- encapsulate the embedding table and the flatten operation of the table into classes that represent layers: the Embedding class an the Flatten class (pytorch also has these two classes)
- encapsulate the list of layers into a class called Sequential

## WaveNet Intro
- The MLP we implemented before is likely to be bottlenecked at the embedding table layer, where it squashes all the context and feed the flattened context to a single layer. 
- in WaveNet, dilated casual convolution layers squash the input by half in each layer as the input progresses through layers. 

## Implementation
- [torch @](https://pytorch.org/docs/stable/generated/torch.matmul.html): if one or two of the operands of torch `@` has more than two dimensions, the last two dimensions are seen as the "matrix" part, and the "prefix" dimensions are seen as the "batch" part: The  batch dimensions are broadcasted. For example, if `input` is a (j×1×n×m) tensor and `other` is a (k×m×p) tensor, `out` will be a (j×k×n×p) tensor.
- To implement the hierarchical dividing (i.e. `block_size` embeddings are grouped into 2 embeddings per group before passing into the first linear layer. The output of tanh are grouped into 2 again before passing to the next linear layer), we need to introduce the "group" dimension (`shape[1]`) to denote the number of groups (eg. 4, 2, 1 in each consecutive layers): use `.view` to change the shape 
	- `torch.squeeze()` squeezes out all dimension of size 1, or if specified the dimension you want to squeeze. 
	- we also need to change the dimension of the LinearLayer accordingly. 

### BatchNorm 
- When we normalize the batch, we should normalize across the "group" as well. `torch.mean` can take a tuple of dimensions averaged across. 

# Build a transformer-based character-level language model
- [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master): this lecture essentially writes the `train.py` and `model.py` from scratch.
- training data: [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Preprocessing 
- download and read in data
- get the vocabulary (in this case, all the unique chars in the text)
### Tokenization
- create a mapping from chars to ints by assigning a number to each char in the vocabulary
	- encode: map text to a list of numbers 
	- decode: map numbers to text
- other tokenizers: 
	- [google's sentencepiece](https://github.com/google/sentencepiece)
	- [openAI's tiktoken](https://github.com/openai/tiktoken)
### Train/test/val split

## Prepare the data 
- "Time" dimension: for any string of length `block_size + 1`, we could construct `block_size` examples/inputs to the transformer, with the context range from 1 to `block_size`. `block_size`: maximum context length for prediction. To do this, we can sample `block_size` length strings (x) and their corresponding labels (y, constructed by offsetting the x by 1 to the right)
	- e.g. "we are" will result in "w" -> "e", "we" -> "we ", "we " -> "a", etc. 
- Batch dimension

## Bigram model in pytorch
- use `nn.Module` to construct the bigramModel class. 
- the only parameters are defined in `__init__`, which is an embedding table of size `vocab_size * vocab_size`: in each row of the table, it stores the "score"/logits for all of the chars in the vocab for a certain char. 
- `forward` is as simple as referencing the embedding table to get the logits for chars. The loss is cross-entropy loss between the predicted logits and the targets. In the training phase, we will provide the targets to calculate the loss; during the inference phase, we do not pass in the targets (Y labels) b/c we only need to get the logits. 

### Training 
- use `torch.optim.AdamW(params, lr)` to initialize an optimizer. Before we only used SGD. 
- the forward pass to get the loss 
- use `.zero_grad` to zero out the gradients from the last iteration before the backward pass
- call `loss.backward()` to get the gradients 
- `optimizer.step()` to update the parameters. 

#### GPU
- If we use GPU in training, we could call `.to("cuda")` to move model and data to GPU

#### Monitor training progress
- for every X iterations, calculate and print out loss: 
	- use `@torch.no_grad()` on top of the function that monitor the loss so that we can prevent pytorch from maintaining gradients. 
	- use `model.eval()` to set the NN to eval mode (BatchNorm, dropout layer has different behaviors in eval mode). After the loss calculation, use `model.train()` to set the NN back to training mode.

## Self-attention 
Besides increasing the maximum context length, we also want the tokens to "talk to each other" under the constraint that the current token can only talk to previous tokens. 

### Math tricks
- one way to aggregate past info is to get a weighted sum of the previous embeddings/channels. We could set `-inf` at the upper triangular part of the weight matrix to prevent giving any weight to future tokens and use softmax to normalize each row of the weights so that they are summed to 1. 
	- to make it data-dependent, the weights (at ith location when it is a vowel char, it may "talk" more with the consonant char two chars before) should be trained. 

### Position embedding 
- people often use position embedding in addition to the embedding for the token value. 

### Implementation of a single-head
- `head_size`: a hyper-parameter that determines the size of the "key" and "query" matrix
- the `key` and `query` is a matrix (`nn.Linear(C, head_size, bias=False)`)
- we pass x to `key` and `query` to obtain `k` and `q`: no "talking" yet
- we then calculate the dot product of `k` and `q`: the talking!
- don't forget to set the weight of "future" context to 0 and normalize the weights: [[#Math tricks]]. 
- Then, instead of getting the weighted sum over `x`, we introduce the `value` layer and we get `v` by passing `x` to `value`. We multiply the weight to `v`.

### Notes
- Attention is a communication mechanism: besides the self-attention block, it can be used in any directed graph: it calculates a weighted sum from all note that point to them with data-dependent weights 
- There is no notion of space in attention. That is why we also encode position. 
- If the task is predicting the next token, it makes sense to impose the constraint that the token should only talk to the previous tokens. However, it the task it different (e.g. sentiment analysis), we can discard this constraint and allow all tokens to communicate (in order to produce a sentiment label, for example). To do that, we could just delete the setting to `-inf` step to make an "encoder" attention block. The former one is thus also called a "decoder" attention block.
- It is called "self-attention" b/c the `k`, `q`, and `v` are all produced from `x`. In "cross-attention", the queries are produced from `x`, but the `k` and `v` come from some other source. 
- "Scaled" attention additional divides the weights by 1/sqrt(head_size). This makes it so when input `q`, `k` are unit variance, weights will be unit variance too and Softmax will stay diffuse and not saturate too much, especially at initialization.

### Multi-head attention
- Multiple heads computed in parallel, and concatenate the results of each head 
### in pytorch
- `register_buffer`: we are using the lower triangular matrix to set the `-inf`, but it does not belong to parameters. This will allow us to maintain this variable. 

## Transformer block
### MLP
- on top of the self-attention, we use a feed forward network that focus on each token individually (by applying independent weights and biases). 

### Block 
- Interspersing **Communication** (attention) with **Computation** (MLP).
- And we can link them together to create a deep NN
- but, we need two more ideas to make the optimization of deep NN work: [[#Residual connection]], [[#Layernorm]]

### Residual connection 
- create a super highway of gradient propagation from loss to input. During initialization, the residual blocks will contribute little, and the gradient will just flow, and they will contribute when the training goes on. 
- in order to match the input and out dimension of the residual block, we need a *projection* layer (a linear layer that adjusts the dimension)

### LayerNorm
In 2D case (batch, feature), BatchNorm normalizes each feature across all batch inputs, so that it distributes unit gaussian across examples. LayerNorm normalizes each example across all features so that each example is unit-gaussian distributed: input examples are no longer inter-dependent. 

In 3D case (batch(n), time(t), feature(d)), for each feature dimension, BatchNorm computes the mean and variance across all examples and all positions in the sequence. It then normalizes it and learns the $\gamma$ and $\beta$ for that feature. In the end, there will be $d$ sets of $\gamma, \beta$ 
- Since BatchNorm normalizes across the batch and sequence dimensions, it can be sensitive to batch size and sequence length variations. This can sometimes lead to issues in tasks with highly variable sequence lengths or in smaller batches where the statistics may not be robust.

On the other hand, for each example sequence, and each element in the sequence, LayerNorm computes the mean and variance across all features.  It then normalizes so that for each sequence element in all sequences its features have a zero-mean, standard deviation 1 distribution. Then it also learns feature-specific $\gamma$ and $\beta$ , but it does not depend on batch size or sequence length. 

In the original transformer paper (2017), the layerNorm is applied after the attention and MLP. It is now more common to use the "pre-norm" formulation: apply layerNorm *before*. 

### Dropout
Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014): in training, randomly drops (by setting them to zero) a subset of neurons
- We can add Dropout layer in the residual block immediately before it gets added back to the input
- also to the self-attention weights  

### Encoder, decoder transformer 
What we have so far is a decoder-only transformer (masked multi-head attention + feed forward). The lower triangular mask constraints the attention to fit the autoregressive, generative regime.  

One part is missing: the encoder and the cross-attention. The original paper introduces the encode part b/c it concerns with the machine translation task: it takes encoded French and decode to English. Therefore, the generated output should be conditioned in some way by the input tokens: it feeds input embedding to another transformer without the masked attention (input tokens are allowed to talk to any input tokens). The output of the encoder transformer is fed into the decoder transformer by generating the keys and values (queries are generated by the decoder transformer)

## How it relates to GPT
Stage 1: pre-training to get a decoder model. 
Stage 2: fine-tuning to "align" it to be an assistant model. 
	1. collect Q & A (prompt is sampled from prompt dataset, and answer is provided by some labeler) data and perform supervised learning 
	2. collect comparison data to train a reward model (labeler will rank the model-generated outputs)
	3. optimize a policy given the reward model using the PPO reinforcement learning algorithm