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
#' @param lambda A scalar or 'NW' or 'CV'. This is the lambda value to use for the algorithm.
#'  'NW' will compute optimal using arguments based on Negahban and WAinwright (2011). 'CV' will compute the optimal lambda using cross-validation. Set by default to "NW".
#' @param iterations A scalar (integer) value. It is the number of iterations the user
#'  specifies should occur. Set by default to 5000.
#' @param etol A scaler. It is the error tolerance for the algorithm. The algorithm will terminate
#'  when either the maximum number of iterations has been met or the mean absolute error between iterations
#'  is below etol.
#' @param fixed_effects A boolean value. This captures whether to implement node-level fixed effects.
#' @param CV_grid A vector. This is the grid of lambda values to use for cross-validation. Set by default to seq(0.01, 10, by=0.01).
#' @param CV_folds A scalar (integer) value. This is the number of folds to use for cross-validation. Set by default to 5.
#' @return An N x M matrix estimate of network connections.
#' @export
#' @import Matrix
matrix_regression <- function(inputs, outputs, lambda = "NW", iterations = 5000, etol = 10e-05, fixed_effects = FALSE, CV_grid = NULL, CV_folds = 5) {
  Lipschitz <- "regression"
  symmetrize <- TRUE
  if (lambda == "CV") {
    CV <- TRUE
  } else {
    CV <- FALSE
  }
  gamma <- 2.0
  if (CV == FALSE) {
    W <- accel_nuclear_gradient(inputs, outputs, lambda, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects)
  } else {
    optimal_lambda <- cross_validation_wrapper(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
    W <- accel_nuclear_gradient(inputs, outputs, optimal_lambda, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects)
  }

  return(W)
}
