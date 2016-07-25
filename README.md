# DA-GTM #
Solving GTM with Deterministic Annealing

## Introduction ##
Generative Topographic Mapping (GTM) is an algorithm for data visualization through dimension reduction. Unlike PCA, which is a traditional visualization method based on linear algebra, GTM seeks a non-linear mapping. For its information theory-based background, GTM finds more separable map than PCA. The GTM problem is basically Gaussian mixture model problem and a standard method to solve this problem is Expectation-Maximization (EM) method.

We apply a novel optimization method, called Deterministic Annealing (DA), to solve the local optimum problem which the original GTM can suffer from.

## Building ##
Dependent Libraries are as follows
* MPI
* GNU Scientific Library (GSL)
* Parallel HDF5
* CMake