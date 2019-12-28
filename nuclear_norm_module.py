# Builds on code from https://github.com/wetneb/tracenorm implementing Ji and Ye (2009).
#
# Ji, S., & Ye, J. (2009, June). An accelerated gradient method for trace norm minimization. In Proceedings of the 26th annual international conference on machine learning (pp. 457-464). ACM.

import numpy as np

def symmetrize(W):
    """
    W = N_2 x N_1 matrix.

    Replaces top N_1 x N_1 submatrix of W with a symmetric version with zero diagonals.
    """
    Wsym = np.abs(W)/2 + W/2  # zero out negative entries
    Wsub = Wsym[:Wsym.shape[1],:]
    Wsub = Wsub/2 + Wsub.T/2  # average off-diagonal pairs
    np.fill_diagonal(Wsub,0)  # zero out diagonals
    Wsym[:Wsym.shape[1],:] = Wsub
    return Wsym

def matrix_regression(Y, X, lmbd, L=-1, symmetric=True, iterations=5000, etol=10e-5, verbose=False):
    """
    Outputs argmin_W { 0.5||Y - XW||_F^2 + lambda||W||_* }, where ||.||_F is the Frobenius norm and ||.||_* is the nuclear norm. Implements a variation of Algorithm 2 of Ji and Ye (2009), an accelerated gradient descent method. Unlike their algorithm, we compute the Lipschitz constant L directly for additional speed-ups.

    Note that in our application, W is the matrix of linking probabilities, what is denoted by P in monte_carlo.py.

    Y = K x N_1 matrix.
    X = K x N_2 matrix.
    lmbd = lambda, the penalty parameter.
    L = Lipschitz constant L in Ji and Ye (2009).
    symmetric = boolean for whether you want constrain the output to be a symmetric matrix with zero diagonals. If True, we apply the symmetrize() function to the output of each gradient descent step.
    iterations = maximum number of gradient descent iterations.
    etol = error tolerance. Algorithm terminates when the mean absolute error between iterations is below etol.
    verbose = boolean for whether you want the algorithm to report the error every 100 iterations.
    """
    # check the dimensions of Y and X
    if Y.shape[1] > X.shape[1]:
        raise ValueError('X must have at least as many columns as Y.')
    if Y.ndim != 2 or X.ndim != 2:
        raise ValueError('Y and X must be matrices.')

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
        if symmetric: W = symmetrize(W) # symmetrize gradient descent output
        prev_alpha = alpha
        alpha = (1 + np.sqrt(1 + 4*(prev_alpha**2)))/2  # second part of step 3, equation (18)
        Z = W + ((prev_alpha - 1)/alpha) * (W - prev_W) # third part of step 3, equation (19)
        
        err = np.abs(prev_W - W).mean() # measure error relative to previous step
        iters += 1
        prev_W = W # update

        if iters%100==0 and verbose: print('Iteration {}. Error {}'.format(iters,err))
        
    if verbose: print('Iteration {}. Error {}'.format(iters,err))
    if iters == iterations: print('Warning: max iterations hit.')

    return W

def gradient_step(Y, X, lmbd, L, current_W):
    """
    Gradient descent step of the accelerated gradient descent method. Solution to equation (8) of Ji and Ye (2009).

    current_W = current estimate of W in the gradient descent.
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

    return np.dot(U, np.dot(S, V))

