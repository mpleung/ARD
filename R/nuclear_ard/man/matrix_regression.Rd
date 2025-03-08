% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/modified_accelerated_descent.R
\name{matrix_regression}
\alias{matrix_regression}
\title{Matrix regression function for ARD}
\usage{
matrix_regression(
  inputs,
  outputs,
  lambda = "NW",
  iterations = 5000,
  etol = 1e-04,
  fixed_effects = FALSE,
  CV_grid = NULL,
  CV_folds = 5
)
}
\arguments{
\item{inputs}{A matrix object. This contains ARD census data in matrix form.
It should be of dimension K x N, where N = village size and K = number of ARD characteristics.}

\item{outputs}{A matrix object. This contains the ARD survey data
in matrix form with size K x M, where N > M = number of households receiving ARD questionairre.}

\item{lambda}{A scalar or 'NW' or 'CV'. This is the lambda value to use for the algorithm.
'NW' will compute optimal using arguments based on Negahban and WAinwright (2011). 'CV' will compute the optimal lambda using cross-validation. Set by default to "NW".}

\item{iterations}{A scalar (integer) value. It is the number of iterations the user
specifies should occur. Set by default to 5000.}

\item{etol}{A scaler. It is the error tolerance for the algorithm. The algorithm will terminate
when either the maximum number of iterations has been met or the mean absolute error between iterations
is below etol.}

\item{fixed_effects}{A boolean value. This captures whether to implement node-level fixed effects.}

\item{CV_grid}{A vector. This is the grid of lambda values to use for cross-validation. Set by default to seq(0.01, 10, by=0.01).}

\item{CV_folds}{A scalar (integer) value. This is the number of folds to use for cross-validation. Set by default to 5.}
}
\value{
An N x M matrix estimate of network connections.
}
\description{
\code{matrix_regression} is a wrapper for \code{\link{accel_nuclear_gradient}} for the specific
implementation of the multivariate regression discussed in Alidaee, Auerbach, and Leung (2020).
Consequently, options have been set to the optimal values of accelerated gradient descent
in the case of a multivariate regression.
}
