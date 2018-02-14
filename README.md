# LinearWordEmbed
This document describes how to learn linear transformation between different word embeddings (e.g. CBOW and word2vec). For more details, see our paper: 

Bollegala, Hayashi, Kawarabayashi.
[Learning Linear Transformations between Counting-based and Prediction-based Word Embeddings.](http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0184544&type=printable)
PLoS ONE 12(9): e0184544, 2017. 

Unfortunately, the original code is dirty, so I decided to show the core recipe of our learning algorithm. 

## Recipe
Let `u_i` be the `m`-dimensional embedding vector and `v_i` be the `n`-dimensional embedding vector for word `i`. The core idea is to learn `C`, the `m` by `n` matrix that transforms `v_i` to `u_i` such that `u_i ~= Cv_i`. For this purpose, we define the objective function over `p` words as `\sum_{i=1}^p ||u_i - Cv_i||^2 = ||U-VC||^2_F`, where `U` and `V` are collections of embeddings over `p` words and `||.||_F` denotes the Frobenius norm. 

We use stochastic gradient descent (SGD) to learn `C`. For SGD, [vowpal wabbit (VW)](https://github.com/JohnLangford/vowpal_wabbit/wiki) is helpful, because it efficiently works for large scale data. 



Note that the problem is equivalent to `m`-variate linear regression. However, because VW cannot handle multidimensional output, we separate the problem as `m` scalar-output linear regression problems. For each prediction dimension `j=1,...,m`, we need to create a file in the [VW input format](https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format). 
In the VW format, each line corresponds to a training sample, and the entire file is something like this:
```
u_1j | 1:v_11 2:v_12 ... n:v_1n
u_2j | 1:v_21 2:v_22 ... n:v_2n
...
u_pj | 1:v_p1 2:v_p2 ... n:v_pn
```
 By running VW with the file for `j=1,...,m`, we can obtain `c_j` as the part of the transformation `C=[c_1;...;c_m]`.
