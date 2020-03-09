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
accel_nuclear_gradient <- function(inputs, outputs, lambda, Lipschitz = 'regression', iterations = 5000, etol = 10e-05, Z_1 = 0, gamma = 2.0, symmetrize = TRUE) {
  # This function implements Algorithm 2 from "An Accelerated Gradient
  # Method for Trace Norm Minimization" by Ji & Ye (2009)
    
  # Initialize scalar values
  alpha   <- 1.0
  

  
  # Initialize W_0 = Z_1 \in R^{m x n}
  N <- dim(inputs)[2] 
  M <- dim(outputs)[2] 
  K <- dim(outputs)[1] 
  if (any(c('matrix', 'Matrix', 'dgCMatrix') %in% class(Z_1))) {
    if (dim(Z_1) == c(M, N)) {
      Z <- Z_1
    } else {
      Z <- matrix(runif(M * N, min = 0.0, max = 1.0), nrow = N, ncol = M)
    }
  } else {
    Z <- matrix(runif(M * N, min = 0.0, max = 1.0), nrow = N, ncol = M)
  } 
  if (symmetrize == TRUE) {
    Z <- symmetrize(Z)
  }
  W <- Z
  
  if (Lipschitz == 'regression') {
    # For the multivariate regression case, the Lipschitz constant can analytically be derived.
    # It is the square of the largest (first) singular value. 
    L <- eigen(inputs %*% t(inputs))$values[1]
  } else if (Lipschitz == 'JiYe') {
    # For objective functions where the Lipschitz constant cannot be derived analytically, we 
    # implement the algorithm introduced by Ji and Ye (2009). Here, we initiate L to 1. Later, 
    # within the for loop, each iteration will converge to the optimal L using Steps 1 and 2
    #  of their Algorithm 2.
    L <- 1
  } else {
    stop(Lipschitz, "is an invalid option for the parameter Lipschitz. Please select one of either 'regression' or 'JiYe'. See documentation for details.")
  }

  if (lambda == 'NW') {
    lambda <- 2 * (sqrt(M) + sqrt(N) + 1) * (sqrt(N) + sqrt(K))
  }
  
  for (i in 1:iterations) {
    #print(paste0("Step ", i, " starting at: ", Sys.time(), ". L size ", L))


    # Steps 1 and 2 only needs to be conducted if the Lipschitz constant cannot be computed analytically.
    # It is computed analytically for the regression problem in Alidaee, Auerbach, and Leung (2019). This
    # functionality is only built in for the purpose of potential future generalization.
    if (Lipschitz == 'JiYe') {

      # Update W
      pLZ <- next_W_func(inputs, outputs, lambda, L.bar, Z)

      
      # Step 2: While loop only activated if 
      # F(p_L(Z_{k - 1})) > Q(p_L(Z_{k-1}), Z_{k-1}).
      # Removing \lambda ||W||_* from both sides since cancels out, 
      # nuclear norm computation isn't super efficient, and don't
      # calculate F or Q elsewhere, so don't need generality.
      # Effectively, this clause says only keep iterating if actual
      # loss value is still greater than approximated loss, because
      # that means we can do better by iterating on the value of L.
      F.value <- obj_func(inputs, outputs, pLZ)
      Q.value <- obj_func.approx(inputs, outputs, Z, pLZ, L.bar)
      #print(paste0("Difference is ", F.value - Q.value, ": ", F.value, " vs ", Q.value))
      j <- 1
      while (F.value > Q.value) {
        #print(paste0("While step ", j, ". Difference is ", F.value - Q.value, ": ", F.value, " vs ", Q.value))
        # Update L bar, as stated in algorithm.
        L.bar <- gamma * L.bar
        
        # Recalculate values for while loop check.
        pLZ     <- next_W_func(inputs, outputs, lambda, L.bar, Z)
        F.value <- obj_func(inputs, outputs, pLZ)
        Q.value <- obj_func.approx(inputs, outputs, Z, pLZ, L.bar)
        j <- j + 1
      }
      #print("Done with while loop.")

      # Update L (technically part of step 3.)
      L           <- L.bar
    } 

    # Step 3: Update values before next iteration. 
    W_kmin1     <- W
    W           <- next_W_func(inputs, outputs, lambda, L, Z) 
    #if (symmetrize == TRUE) { # commenting this out for acceleration. Only symmetrizing at end of algorithm.
    #  W <- symmetrize(W)
    #}
    alpha_kmin1 <- alpha
    alpha       <- (1.0 + sqrt(1.0 + 4.0*alpha^2))/2.0
    Z           <- W + ((alpha_kmin1 - 1.0)/alpha)*(W - W_kmin1)

    error <- mean(abs(W_kmin1 - W))
    if (error < etol) {
      break
    }
    
  }
  
  if (symmetrize == TRUE) W <- symmetrize(W)
  
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
  Lipschitz  <- 'regression'
  lambda     <- 'NW'
  Z_1        <- 0
  symmetrize <- TRUE
  W <- accel_nuclear_gradient(inputs, outputs, lambda, Lipschitz, iterations, etol, Z_1, gamma, symmetrize)
  return(W)
}