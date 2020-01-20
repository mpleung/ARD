"""Penalized estimator for recovering networks from ARD

This module implements the penalized regression estimator of Alidaee, Auerbach, and Leung (2020) for recovering networks from ARD. It builds on code from https://github.com/wetneb/tracenorm implementing Ji and Ye (2009).

Ji, S., & Ye, J. (2009, June). An accelerated gradient method for trace norm minimization. In Proceedings of the 26th annual international conference on machine learning (pp. 457-464). ACM.
"""

import numpy as np

def symmetrize(W):
    """Transforms a rectangular matrix into a symmetrized version.

    This function takes a rectangular matrix (one with more rows than columns) and performs the following transformations. First, zero out any negative entries. Second, extract the topmost square matrix - call it Wsym - with column dimension equal to that of the input W. Third, symmetrize Wsym by averaging the ij and ji entries together and replacing each entry with the average. Fourth, zero out the diagonals of Wsym. Fifth, replace the topmost square matrix of W with Wsym.

    Parameters
    ----------
    W
        A rectangular (more rows than columns) numpy matrix.

    Returns
    -------
    Wsym
        A numpy matrix with the same dimensions as W.
    """
    if W.shape[0] < W.shape[1]:
         raise ValueError('Input must be a rectangular matrix (more rows than columns).')

    Wsym = np.abs(W)/2 + W/2      # zero out negative entries
    Wsub = Wsym[:Wsym.shape[1],:] # extract topmost square
    Wsub = Wsub/2 + Wsub.T/2      # average off-diagonal pairs
    np.fill_diagonal(Wsub,0)      # zero out diagonals
    Wsym[:Wsym.shape[1],:] = Wsub
    return Wsym

def matrix_regression(Y, X, lmbd=-1, L=-1, symmetric=True, iterations=5000, etol=10e-5, verbose=False):
    """Estimates the distribution of a network from ARD.

    This function outputs argmin_W { 0.5||Y - XW||_F^2 + lambda||W||_* }, where ||.||_F is the Frobenius norm and ||.||_* is the nuclear norm. It implements a variation of Algorithm 2 of Ji and Ye (2009), an accelerated gradient descent method. Unlike their algorithm, we compute the Lipschitz constant L directly for additional speed-ups. Note that in our network application, W is the matrix of linking probabilities.

    Parameters
    ----------
    Y
        ARD, represented as a K x N_1 numpy matrix.
    X
        type data, represented as a K x N_2 matrix.
    lmbd : optional
        The penalty parameter.
    L : optional
        Lipschitz constant L in Ji and Ye (2009).
    symmetric : optional
        Boolean for whether you want modify the output of the accelerated gradient descent to ensure a non-negative output whose topmost matrix is symmetric with zero diagonals.  If True, we apply the symmetrize() function to the output of each gradient descent step.
    iterations : optional
        Maximum number of gradient descent iterations.
    etol : optional
        Error tolerance. Algorithm terminates when the mean absolute error between iterations is below etol.
    verbose : optional
        Boolean for whether you want the algorithm to report the error every 100 iterations.

    Returns
    -------
    W
        Matrix of linking probabilities, represented as a N_2 x N_1 numpy matrix.
    """
    # check the dimensions of Y and X
    if Y.shape[1] > X.shape[1]:
        raise ValueError('X must have at least as many columns as Y.')
    if X.shape[0] != X.shape[0]:
        raise ValueError('X and Y must have the same row dimension.')
    if Y.ndim != 2 or X.ndim != 2:
        raise ValueError('X and Y must be matrices.')

    # default penalty parameter
    if lmbd <= 0:
        lmbd = 2 * (np.sqrt(Y.shape[1]) + np.sqrt(X.shape[1]) + 1) * (np.sqrt(X.shape[1]) + np.sqrt(X.shape[0]))

    # initial guess for solution
    prev_W = symmetrize(np.random.rand(X.shape[1],Y.shape[1]))
    Z = prev_W

    # compute Lipschitz constant for optimizer
    if L == -1:
        U, s, V = np.linalg.svd(X.T.dot(X))
        L = s[0]

    iters = 0
    err = 1
    alpha = 1
    
    # Implements step 3 of Algorithm 2 of Ji and Ye (2009). Other steps are avoided because we already computed the Lipschitz constant.
    while iters < iterations and err > etol:
        W = gradient_step(Y, X, lmbd, L, Z) # first part of step 3
        prev_alpha = alpha
        alpha = (1 + np.sqrt(1 + 4*(prev_alpha**2)))/2  # second part of step 3, equation (18)
        Z = W + ((prev_alpha - 1)/alpha) * (W - prev_W) # third part of step 3, equation (19)
        
        err = np.abs(prev_W - W).mean() # measure error relative to previous step
        iters += 1
        prev_W = W # update

        if iters%100==0 and verbose: print('Iteration {}. Error {}'.format(iters,err))
        
    if verbose: print('Iteration {}. Error {}'.format(iters,err))
    if iters == iterations: print('Warning: max iterations hit.')
    
    if symmetric: W = symmetrize(W) # optionally impose constraints on graph
    return W

def gradient_step(Y, X, lmbd, L, current_W):
    """Gradient descent step of the accelerated gradient descent method. 

    This is the solution to equation (8) of Ji and Ye (2009).

    Parameters
    ----------
    Y
        ARD, represented as a K x N_1 numpy matrix.
    X
        type data, represented as a K x N_2 matrix.
    lmbd : optional
        The penalty parameter.
    L
        Lipschitz constant L in Ji and Ye (2009).
    current_W
        Current estimate of W in the gradient descent.

    Returns
    -------
    output
        Gradient update.
    """
    XT = X.T
    gradient = np.dot(XT.dot(X),current_W) - XT.dot(Y)
    C = current_W - (1/L) * gradient
    U, s, V = np.linalg.svd(C)

    s = s - lmbd/L
    final_s = np.maximum(s,0)
    U_dim = U.shape[1]
    V_dim = V.shape[0]
    S = np.zeros((U_dim, V_dim))
    S_rank = min(U_dim, V_dim)
    S[0:S_rank,0:S_rank] = np.diag(final_s)

    output = np.dot(U, np.dot(S, V))

    return output

