#' Accelerated gradient function for ARD
#'
#' \code{accel_nuclear_gradient} implements the Accelerated Gradient Algorithm (Algorithm 2)
#' from Ji & Ye (2009). This should be one of only two functions users interact with directly
#' from the \emph{ardlasso} package. (The other is \code{\link{matrix_regression}}, which is a wrapper
#' function for accel_nuclear_gradient, forcing the option Lipschitz == 'regression'.)
#' When symmetrize is set to \code{TRUE}, this function instead implements Algorithm 1 in
#' Alidaee, Auerbach, and Leung (2020). Note that Alidaee et al. (2019) have a slight change
#' in notation, wherein the tuple (N, M) below is equivalent to their (N_2, N_1).
#'
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#'  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
#' @param outputs A matrix object. This contains the ARD survey data
#'  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre.
#' @param lambda A scalar (numeric) value. This is an initial guess that will be iterated on. It can
#'  alternatively be defined as 'NW', which will set \eqn{\lambda = 2(\sqrt{N} + \sqrt{M})(\sqrt{N} + \sqrt{K})}.
#' @param Lipschitz A string. This determines how the Lipschitz constant is computed. 'JiYe' implements
#'  the iterative algorithm outlined in Ji and Ye (2009). 'regression' sets it to the analytically derived
#'  constant for the multivariate regression problem posed in Alidaee et al. (2020). Set to 'regression' by
#'  default.
#' @param iterations A scalar (integer) value. It is the number of iterations the user
#'  specifies should occur. Set by default to 5000.
#' @param etol A scaler. It is the error tolerance for the algorithm. The algorithm will terminate
#'  when either the maximum number of iterations has been met or the mean absolute error between iterations
#'  is below etol.
#' @param symmetrize A boolean value. This captures whether to implement the aforementioned modified
#'  gradient descent algorithm..
#' @return An N x M matrix estimate of network connections.
#' @export
#' @import Matrix
accel_nuclear_gradient <- function(inputs, outputs, lambda, Lipschitz = "regression", iterations = 5000, etol = 10e-05, gamma = 2.0, symmetrize = TRUE) {
  # This function implements Algorithm 2 from "An Accelerated Gradient
  # Method for Trace Norm Minimization" by Ji & Ye (2009)

  # Initialize scalar values
  alpha <- 1.0



  # Initialize W_0 = Z_1 \in R^{m x n}
  N <- dim(inputs)[2]
  M <- dim(outputs)[2]
  K <- dim(outputs)[1]

  Z <- matrix(runif(M * N, min = 0.0, max = 1.0), nrow = N, ncol = M)

  if (symmetrize == TRUE) {
    Z <- symmetrize(Z)
  }

  W <- Z

  if (Lipschitz == "regression") {
    # For the multivariate regression case, the Lipschitz constant can analytically be derived.
    # It is the square of the largest (first) singular value.
    L <- eigen(inputs %*% t(inputs))$values[1]
  } else if (Lipschitz == "JiYe") {
    # For objective functions where the Lipschitz constant cannot be derived analytically, we
    # implement the algorithm introduced by Ji and Ye (2009). Here, we initiate L to 1. Later,
    # within the for loop, each iteration will converge to the optimal L using Steps 1 and 2
    #  of their Algorithm 2.
    L <- 1
  } else {
    stop(Lipschitz, "is an invalid option for the parameter Lipschitz. Please select one of either 'regression' or 'JiYe'. See documentation for details.")
  }

  if (lambda == "NW") {
    lambda <- 2 * (sqrt(M) + sqrt(N) + 1) * (sqrt(N) + sqrt(K))
  }

  for (i in 1:iterations) {
    # print(paste0("Step ", i, " starting at: ", Sys.time(), ". L size ", L))


    # Steps 1 and 2 only needs to be conducted if the Lipschitz constant cannot be computed analytically.
    # It is computed analytically for the regression problem in Alidaee, Auerbach, and Leung (2019). This
    # functionality is only built in for the purpose of potential future generalization.
    if (Lipschitz == "JiYe") {
      JiYe_values <- compute_lipschitz(inputs, outputs, lambda, L.bar, Z, gamma)
      L <- JiYe_values$L_bar
      lambda <- JiYe_values$lambda
    }

    # Step 3: Update values before next iteration.
    value_iterator <- compute_iteration(inputs, outputs, lambda, L, Z, alpha, W, etol)
    W <- value_iterator$W
    alpha <- value_iterator$alpha
    Z <- value_iterator$Z
    flag <- value_iterator$flag

    if (flag) {
      break
    }
  }

  if (symmetrize == TRUE) W <- symmetrize(W)

  W <- pmax(W, matrix(0, nrow = N, ncol = M))

  return(as.matrix(W))
}


#' Matrix regression function for ARD
#'
#' \code{matrix_regression} is a wrapper for \code{\link{accel_nuclear_gradient}} for the specific
#' implementation of the multivariate regression discussed in Alidaee, Auerbach, and Leung (2020).
#' Consequently, options have been set to the optimal values of accelerated gradient descent
#' in the case of a multivariate regression.
#'
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#'  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
#' @param outputs A matrix object. This contains the ARD survey data
#'  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre.
#' @param iterations A scalar (integer) value. It is the number of iterations the user
#'  specifies should occur. Set by default to 5000.
#' @param etol A scaler. It is the error tolerance for the algorithm. The algorithm will terminate
#'  when either the maximum number of iterations has been met or the mean absolute error between iterations
#'  is below etol.
#' @return An N x M matrix estimate of network connections.
#' @export
#' @import Matrix
matrix_regression <- function(inputs, outputs, iterations = 5000, etol = 10e-05) {
  Lipschitz <- "regression"
  lambda <- "NW"
  symmetrize <- TRUE
  W <- accel_nuclear_gradient(inputs, outputs, lambda, Lipschitz, iterations, etol, gamma, symmetrize)
  return(W)
}
