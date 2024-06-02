---
title: Rob Nowak ML lunch meeting talk
allDay: false
startTime: 13:00
endTime: 14:00
date: 2024-04-04
completed: null
---
# What Kinds of Functions do NN Learn? Theory and Practice

## What ReLU does
## Weight Decay Training 
- reduce weights by subtracting a tiny bit (same as regularization)
- ridge loss (L2 loss) is the same as minimizing "path norm" and "mutli-task" lasso (L1 loss), aka "the natural norm of neural function", b/c the output weight (output of ReLU) is the same as input weight in ReLU with weight decay training and constrained weight norm

The function a NN learns is called Neural Functions. 

### Analysis of Neural Function
- given function f, take one derivative: piece wise linear function; take two derivatives, reveals the output weights 
- The norm of f is the total variation at every direction: NN reduces the high-dimension f to a sparse (a smoother, low-dimensional) solution to each dimension. 

## Representer Theorem 
- the width of each layer of NN can be bounded

We can take advantage of the sparsity of NN layer to compress the number of neurons of each layer to achieve better regularities (lower variation norm) and preserve the same input/output relation. 
## Non-uniqueness of Trained NN
- even with weight-decay
- if in one layer there are two neurons with similar input weights, weight decay will eliminate one of the two neurons. As a byproduct, 
- When training on two/multiple tasks, NN will be forced to find unique solutions 

## Implicit Neural Representations 
- Standard ReLU netowrks suffer from spectral bias and therefore cannot recover/represent high-frequency detail
- Actually, it is not true. They constrain ReLu in groups of 7 to form a B-spline wavelet 