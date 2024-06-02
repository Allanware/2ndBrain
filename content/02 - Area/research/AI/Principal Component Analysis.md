# Dimension Reduction 
Working directly with high-dimensional data comes with some difficulties: 
- hard to analyze
- interpretation is difficult
- visualization is nearly impossible
- storage of the data vectors can be expensive.
However, dimension reduction can exploit the redundancy and the correlation among dimensions, ideally **accurately** and **without losing information**. We can think of dimension reduction as a **compression** technique. 

# Problem Formulation
We consider an i.i.d. data set $\mathbf X = \{\mathbf x_1, ..., \mathbf x_N\}, \mathbf x_n \in \mathbb R^D$. We want to project them into $N$ $M$-dimension vectors where $M << D$. Principal component analysis (PCA) is a **linear** algorithm for dimension reduction. Mathematically, $$\mathbf z_n = \mathbf B^\top \mathbf x_n \in \mathbb{R}^M$$where $$\mathbf B := [\mathbf b_1, \ldots, \mathbf b_M] \in \mathbb{R}^{D \times M}
$$is the projection matrix. We assume that the columns of $B$ are orthonormal so that $\mathbf{b}_i^\top \mathbf{b}_j = 0$ if and only if $i \neq j$ and $\mathbf{b}_i^\top \mathbf{b}_i = 1.$ In other words, $\mathbf b_1, \ldots, \mathbf b_M$ form a basis of the $M$-dimensional space, and $z_{in}$ is the weighted sum of $\mathbf b_i$ (each $\mathbf b$ is of length $D$) and $\mathbf x_n$. 
# Motivation 
Given the linearity framework, PCA strives to achieve two goals: 
1. Maximize the variance (the spread of the data) of the low-dimensional representation/projection of the original data (the **without losing information** aspect: Hotelling, 1933): ![](../../../pca_mv.png)For example, PCA will likely to use $x_1$ as the principal component and discard $x_2$ because $x_1$ (or some small change to it) maximizes the variance. 
2. Minimize the reconstruction error in terms of the squared distance between $\mathbf {x}$ and  $\mathbf {\tilde x}$ (the **accuracy** aspect: Pearson, 1901): $$\frac{1}{N} \sum_{n=1}^N \|\mathbf x_n - \tilde{\mathbf x}_n\|^2$$
**It turns out that there is a *unique* solution that is the key to satisfy both aspects.**

# Maximizing Variance
The objective is to maximize the variance of the projection $\mathbf z_n := [z_{1n}, \ldots, z_{Mn}]$, which are coordinates/codes of a set of $M$ orthonormal basis. In other words, the problem is to choose a subspace that maximize the total variance. 

Since the basis are orthogonal, the total variance of the $\mathbf x_n$'s projections to a $M$-dimensional space is the sum of projection variance to each base vector. $$V_{total} = \sum_{i=1}^M {V_i}$$
Therefore, we can break down the problem by finding the $M$-largest orthonormal vectors as the basis of the subspace. 

We start by seeking vector $\mathbf b_1$ that maximizes the variance of the projection of $\mathbf x_n$ to $\mathbf b_1$, which is the first coordinate of $\mathbf x_n$'s projection $\mathbf z_n$: 
$$z_{1n} = \mathbf b_1^T \mathbf x_n \tag {1}$$
Recall that the projection of one vector on to another is $proj_{\mathbf b_1}{\mathbf x_n} = \frac{\mathbf{x_n} \cdot \mathbf{b_1}}{\|{\mathbf{b_1}}\|^2} \mathbf{b_1}$. We assume $\mathbf b_1$ is a unit vector, i.e. $\|{\mathbf{b_1}}\|^2 = 1$, and $z_{1n}$ is the coordinate in the direction of $\mathbf b_1$, we can omit $\mathbf b_1$ in the end. 

By exploiting the i.i.d assumption of data ($Var(\mathbf x_i + \mathbf x_j) = Var(\mathbf x_i) + Var(\mathbf x_j)$), and without loss of generality, by assuming $\mathbb{E}_\mathbf x[\mathbf x] = 0$ (see [Step 1 Center the data](#Step%201%20Center%20the%20data)), we obtain:
$$V_1 := V[z_1] = \frac{1}{N} \sum_{n=1}^N z_{1n}^2$$
By substitution using (1): 
$$\begin{align}
V_1 &= \frac{1}{N} \sum_{n=1}^N (\mathbf {b}_1^ \top \mathbf x_n)^2 = \frac{1}{N} \sum_{n=1}^N \mathbf {b}_1^\top \mathbf x_n \mathbf {x}_n^\top \mathbf b_1  \\
&= \mathbf {b}_1^\top \left( \frac{1}{N} \sum_{n=1}^N \mathbf x_n \mathbf {x}_n^\top\right) \mathbf b_1 = \mathbf{b}_1^\top \mathbf S \mathbf b_1
\end{align}
$$
Where $S$ is the covariance matrix of $\mathbf X$ (again, we here assume $\mathbb{E}_\mathbf x[\mathbf x] = 0$).
To maximize $V_1$, $\mathbf b_1$ is the eigenvector associated with largest eigenvalue $\lambda_1$ of $\mathbf S$ ($\mathbf S\mathbf b_1 = \lambda_1\mathbf b_1$), and $V_1 = \lambda_1$. Since we assume $\mathbf b_1$ to be a unit vector, $\mathbf b_1$ becomes the first orthonormal base of the subspace $U$ inside $\mathbb R^D$. Using this method, we find the first principal component.

We then subtract out the variance that already explained by the $\mathbf b_1$ direction and this gets us a new covariance matrix $\mathbf {\hat S}$, which has the same eigenvectors and eigenvalues as $\mathbf S$, except that the eigenvalue associated with $\mathbf b_1$ is 0. So again, we select the eigenvector $\mathbf b_2$ that associates with the largest eigenvalue $\lambda_2$. By induction, the first $M$ eigenvectors of $\mathbf S$ (ranked by their associating eigenvalues) are the orthonormal vectors that form the basis of the subspace $U$ we want that maximizes the total projection variance.

Therefore, we can find all principal components in parallel by first calculating the covariance matrix $\mathbf S$, and applying eigen-decomposition and extracting the first $M$ eigenvalues and eigenvectors. 
# Step-by-step

## Step 0: Get data 

## Step 1: Data Preprocessing 
### Center the data?
In principle, we don't need to center the data, as the variance we are maximizing does not depend on $\mathbf {\bar X}: \operatorname{Var}(\mathbf z) = \operatorname{Var}(\mathbf{B}^{\top}(\mathbf{x} - \boldsymbol{\mu})) = \operatorname{Var}(\mathbf{B}^{\top}\mathbf{x} - \mathbf{B}^{\top}\boldsymbol{\mu}) = \operatorname{Var}(\mathbf{B}^{\top}\mathbf{x})$. 

We will also verify this in the example shown later, where we compare the result of applying PCA to both non-centered $\mathbf X$ and centered $\mathbf X ^ \prime$. But in a lot of tutorials they ask you to center the data by subtracting $\bar x_i, i=1, \dots, D$, $D$ is the number of dimensions. Why? Because people often either use $\mathbf X^T\mathbf X/(n−1)$ to calculate covariance matrix (by omitting the sample mean which effectively assuming mean=0) or they use singular value decomposition to calculate PCs directly, which assumes the data is centered. Either way, it is always safer to center the data. Often in implementation there is library function you can call to center and standardize the data at the same time.
### Standardize the data?
Standardization divides the data by standard deviation $\sigma_i, i=1, \dots, D$ so that each dimension is unit free and has variance 1. This step is necessary (unlike the centering part), because recall that the PCA's goal is to maximize the variance. If a certain dimension in $\mathbf X$ is disproportionally large in magnitude, the variance PCA in turn try to maximize will  disproportionally be biased towards this dimension. For example, if dimension $i$ is measuring length in meter, its variance will be 10000 more by changing its unit to centimeters. In short, unit in data games the PCA system. Standardization prevents this. 

A nice feature that comes with standardization is it turns the covariance matrix $\mathbf S$ into correlation matrix $\mathbf C$: recall that $$Corr = \frac{\text{Cov}(X_i, X_j)}{\sqrt{\text{Var}(X_i) \text{Var}(X_j)}} = \text{Cov}(X_i, X_j)​.$$Since each $\mathbf x$ is unit variance. 

And we know that principal components are orthogonal, i.e., they are not correlated, we can interpret PCA as effectively *de-correlating* the original $D$ dimensions of the data matrix $\mathbf X$, which circles back to what we've discussed in [Dimension Reduction](#Dimension%20Reduction).  

## Step 2: Calculate the Covariance Matrix

## Step 3: Eigen-Decomposition 

## Step 4: Mapping data to subspace U


# PCA library

# Analyze PCA results

# Projecting back
We may also want to project the $\mathbf z_n$ back to $\mathbf {\tilde x_n} = \mathbf B \mathbf z_n=\mathbf B \mathbf B ^\top \mathbf x_n \in \mathbb R^D$ , which live in a lower dimensional subspace $U \subseteq \mathbb R^D, dim(U) = M$. For example, if $D = 3, M = 2$, then $\mathbf {\tilde x_n}$ and $\mathbf z_n$ are both projections of $\mathbf x_n$s: $\mathbf {\tilde x_n}$ are 3D vectors that lie in the 2D plane $U$, whereas $\mathbf z_n$ are 2D vectors. They are both projections, but one is viewed in the coordinate system of $\mathbb R^D$, the other one is viewed in terms of the subspace $U$, specifically its orthonormal basis $\mathbf b_1, \ldots, \mathbf b_M$. 

The reason we are using $\mathbf B$ to project back is because it minimizes the euclidean distance between $\mathbf {x_n}$ and $\mathbf {\tilde x_n}$, aka, the reconstruction error. 

We can think of PCA as a linear encoder $\mathbf B^\top$ that maximizes the variance and a linear decoder that minimizes the error.

# Limitations of PCA
- **Assumption of Linearity**. Recent success in Machine learning, specifically in deep learning, in some sense is solely depends on its ability to learn to model extremely complex non-linear functions. For example, if the original data points are in $\mathbb R^3$ , and we choose $M=2$, PCA will get us the optimal 2D plane that maximizes the projection variance and minimizes the reconstruction error. All good. But there must exists a "curly plane" or some much more complex structure that can account for more variance and has lower error. 
- **Variance as Information**: Variance is the bread and butter to PCA. What if the variance is "contaminated" by noise? 