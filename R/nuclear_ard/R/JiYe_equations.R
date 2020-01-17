# This file implements various equations from 
# Ji & Ye (2009), implemented for the specific
# case of f(W) = ‖y - AX‖_2^2. This is the 
# relevant function for ARD implementation.
# Those interested in implementation for other
# objective functions should modify obj_func and
# obj_func.grad. All other functions should be identical.



#' Objective (Loss) function for ARD
#' 
#' \code{obj_func} calculates our objective function 
#' \deqn{F(X) = \Vert  y - AX \Vert _2^2 + \lambda \Vert  X \Vert _*}{F(X) = ‖y - AX‖_2^2 + \lambda ‖X‖_*}
#' but---as stated in \code{\link{obj_func.approx}} ---\eqn{\Vert \lambda \Vert_*}{\lambda ‖X‖_*} 
#' cancels out whenever needed, since the term is also in the approximated function it is being compared to.
#' As a result, it is not necessary to compute. Because of this, we effectively
#' are calculating \eqn{f} instead of \eqn{F}. 
#' 
#' This function should typically not be used directly. Instead, it should solely be called upon by other functions.
#' 
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#'  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
#' @param outputs A matrix object. This contains the ARD survey data 
#'  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre. 
#' @param W_k A matrix object. This is the current iteration's guess for the ideal N x M linear operator. 
#'  Users do not need to create this themselves. It is provided by \code{\link{accel_nuclear_gradient}} 
#'  via \code{\link{next_W_func}}. 
#' @return A scalar value denoting the value of our loss function. 
#' @export
#' @import Matrix

obj_func <- function(inputs, outputs, W_k) {
  # First, check that input parameters are valid.
  validation <- obj_func.testing(inputs, outputs, W_k)

  # We want to compute square of Frobenius norm of y - AX. Thus, we must first
  # compute y - AX.
  vec <- Matrix::Matrix(outputs - inputs %*% W_k)

  # Next, compute Frobenius norm and square it.
  # R has LAPACK implementation of this already via norm function.
  obj_func.value <- Matrix::norm(vec, type = 'f')^2
  return(obj_func.value)
}


#' Objective (Loss) function gradient for ARD
#' 
#' \code{obj_func.grad} calculates the gradient of our objective function 
#' \deqn{F(X) = \Vert  y - AX \Vert _2^2 + \lambda \Vert  X \Vert _*}{F(X) = ‖y - AX‖_2^2 + \lambda ‖X‖_*}
#' but---as stated in \code{\link{obj_func.approx}} ---\eqn{\Vert \lambda \Vert_*}{\lambda ‖X‖_*} 
#' cancels out whenever needed, since the term is also in the approximated function it is being compared to.
#' As a result, we equivalently compute ∇\eqn{f} and this function is used this way by XXX.
#' The formula for the gradient--- is \deqn{∇f(X) = X^TXW - X^Ty}{∇f(X) = (X^T)XW - (X^T)y.}
#' 
#' This function should typically not be used directly. Instead, it should solely be called upon by other functions.
#' 
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#'  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
#' @param outputs A matrix object. This contains the ARD survey data 
#'  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre. 
#' @param W_k A matrix object. This is the current iteration's guess for the ideal N x M linear operator. 
#'  Users do not need to create this themselves. It is provided by \code{\link{accel_nuclear_gradient}} 
#'  via \code{\link{next_W_func}}. 
#' @return A scalar value denoting the value of our loss function. 
#' @export
#' @import Matrix
obj_func.grad <- function(inputs, outputs, W_k) {
  # First, check that input parameters are valid.
  validation <- obj_func.testing(inputs, outputs, W_k)

  # To avoid long names, give compact matrix notation to variables
  X  <- inputs
  Xt <- t(inputs)
  Y  <- outputs

  # Compute relevant products
  XtX <- Xt %*% X
  XtY <- Xt %*% Y


  # Compute gradient of f(X). 
  gradient <- XtX %*% W_k- XtY
  return(gradient)
}

#' Approximated Loss function for ARD
#' 
#' \code{obj_func.approx} implements equation (8) from Ji & Ye (2009):
#' \deqn{P_\mu (W, W_{k - 1}) + \lambda \Vert W \Vert_*.}{P_{\mu}(W, W_{k-1}) + \lambda ‖W‖_*.}
#' But---as in \code{\link{obj_func}} ---\eqn{\Vert \lambda \Vert_*}{\lambda ‖W‖_*} 
#' cancels out whenever needed, since the term is also in the objective function it is being compared to.
#' As a result, it is not necessary to compute. As a result, we are only actually only returning \eqn{P_\mu (W, W_{k - 1})}
#' This is defined by equation (7) of Ji & Ye: \deqn{P_\mu (W, W_{k - 1}) = f(W_{k - 1}) + 〈W - W_{k-1}, ∇f(W_{k-1}〉+ (\mu/2) \Vert W - W_{k-1}\Vert _F^2.}{P_\mu (W, W_{k - 1}) = f(W_{k - 1}) + 〈W - W_{k-1}, ∇f(W_{k-1}〉 + (\mu/2) ‖ W - W_{k-1}‖_F^2.}
#' 
#' This function should typically not be used directly. Instead, it should solely be called upon by other functions.
#' 
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#'  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
#' @param outputs A matrix object. This contains the ARD survey data 
#'  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre. 
#' @param W_k A matrix object. This is the current iteration's guess for the ideal N x M linear operator. 
#'  Users do not need to create this themselves. It is provided by \code{\link{accel_nuclear_gradient}} 
#'  via \code{\link{next_W_func}}. 
#' @param W_kplus1 A matrix object. This is the next iteration's guess for the ideal N x M linear operator. 
#'  Users do not need to create this themselves. It is provided by \code{\link{accel_nuclear_gradient}} 
#'  via \code{\link{next_W_func}}. 
#' @param mu A scalar (numeric) value. It is provided by \code{\link{accel_nuclear_gradient}}. 
#' @return A scalar value denoting the value of our approximate loss function. 
#' @export
#' @import Matrix
obj_func.approx <- function(inputs, outputs, W_k, W_kplus1, mu) {
  # First, check that input parameters are valid.
  # Many checks are the same as for the regular objective function,
  # so we can re-use obj_func.testing.
  validation <- obj_func.testing(inputs, outputs, W_k)
  # But, we also have additional tests, due to extra parameters. 
  # These are included below.
  validation2 <- obj_func.testing(inputs, outputs, W_kplus1)
  if (class(mu) != 'numeric') {
      stop("Input variable 'mu' for obj_func.approx must be of class 'numeric'.
        Inputted object is of class ", class(mu))
  }

  # Equation (7) has three terms. Variable names will
  # refer to them by order.
  W.delta <- W_kplus1 - W_k
  f.grad  <- obj_func.grad(inputs, outputs, W_k)

  second_val.inner <- f.grad %*% Matrix::t(W.delta)
  
  first_val  <- obj_func(inputs, outputs, W_k)
  second_val <- matrix_trace(second_val.inner)
  third_val  <- (mu/2) * Matrix::norm(W.delta, type = 'f')^2
    
  P_mu <- first_val + second_val + third_val
  return(P_mu)
}

#' Iterator for loss function
#' 
#' \code{next_W_func} implements equation (15) from Ji & Ye (2009):
#' \deqn{W_k = p_L(W_{k - 1})} using the results from their Theorem 3.1,
#' which states that this can be found using singular value decomposition of \eqn{W_{k - 1}}.
#' 
#' This function should typically not be used directly. Instead, it should solely be called upon by other functions.
#' 
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#'  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
#' @param outputs A matrix object. This contains the ARD survey data 
#'  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre. 
#' @param W_k A matrix object. This is the current iteration's guess for the ideal N x M linear operator. 
#'  Users do not need to create this themselves. It is provided by \code{\link{accel_nuclear_gradient}} 
#'  via \code{\link{next_W_func}}. 
#' @param lambda A scalar (numeric) value. It is provided by \code{\link{accel_nuclear_gradient}}.
#' @param mu A scalar (numeric) value. It is provided by \code{\link{accel_nuclear_gradient}}.
#' @return A scalar value denoting the value of our approximate loss function. 
#' @export
#' @import Matrix
next_W_func <- function(inputs, outputs, lambda, mu, W_k) {
  # First, check that input parameters are valid.
  # Many checks are the same as for the regular objective function,
  # so we can re-use obj_func.testing.
  validation <- obj_func.testing(inputs, outputs, W_k)
  # But, we also have additional tests, due to extra parameters. 
  # These are included below.
  if (class(mu) != 'numeric') {
      stop("Input variable 'mu' for obj_func.approx must be of class 'numeric'.
        Inputted object is of class ", class(mu))
  } else if (class(lambda) != 'numeric') {
      stop("Input variable 'lambda' for obj_func.approx must be of class 'numeric'.
        Inputted object is of class ", class(lambda))
  }

  
  # First, introduce primitive variables for the function
  W <- W_k
  L <- mu #practically, our mu = L for this algorithm

  
  # Next, define C = W_{k-1} - (1/L)\grad f(W_{k - 1})
  # To do so, start by defining gradient of (1/2N)*‖y-Ax‖_2^2
  # which is (1/N)*(A^T A - A^T y) where, for us, W = A.
  # We use obj_func.grad since it is really gradient of f,
  # for reasons described in the latter function's comments.
  gradient <- obj_func.grad(inputs, outputs, W_k)
  
  C <- W - (1.0/L)*gradient
  
  # As per Theorem 3.1, gather the singular values from SVD of C
  C.svd             <- svd(C)
  C.singvalues      <- svd(C)$d
  C.singvalues.size <- length(C.singvalues)
  
  # Do soft thresholding
  C.singvalues.thresholded.mat <- cbind(C.singvalues - (lambda/(L))*rep(1.0, C.singvalues.size),
                                    rep(0.0, C.singvalues.size))
  C.singvalues.thresholded     <- apply(C.singvalues.thresholded.mat, 1 , max)
  
  # Get size of decomposed matrices, which will give us rank of C and let us create Sigma_lambda from Thm 3.1
  C.u  <- C.svd$u
  C.v  <- C.svd$v
  r    <- dim(C.u)[2]
  n    <- dim(C.v)[1]
  rank <- min(r, n)

  sigma_lambda                 <- Matrix::Matrix(as.numeric(0), nrow = r, ncol = n)
  sigma_lambda[1:rank, 1:rank] <- Matrix::Matrix(diag(C.singvalues.thresholded))

  # Return arg min value, based on Thm 3.1
  argmin <- C.u %*% sigma_lambda %*% t(C.v)

  return(argmin)  
}
