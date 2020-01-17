#' Input validation for objective function 
#' 
#' \code{\link{obj_func}} and \code{\link{obj_func.grad}} have common inputs
#' and therefore require the same validations. To avoid code redundancy, these
#' common checks have been placed in this function, which is then referenced.
#' 
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#'  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
#' @param outputs A matrix object. This contains the ARD survey data 
#'  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre. 
#' @param W_k A matrix object. This is the current iteration's guess for the ideal N x M linear operator. 
#'  Users do not need to create this themselves. It is provided via \code{\link{next_W_func}}. 
#' @return There is no returned value. This function solely raises errors in the event of invalid inputs. 
#' @export
#' @import Matrix
obj_func.testing <- function(inputs, outputs, W_k) {
  # Before proceeding, must check that mat meets requirements.
  if (!any(c("matrix", "Matrix", 'dgeMatrix') %in% class(inputs))) {
      stop("Input variable 'inputs' for obj_func must be of class 'matrix'.
        Inputted object is of class ", class(inputs))
  } else if (!any(c("matrix", "Matrix", 'dgeMatrix') %in% class(outputs))) {
      stop("Input variable 'outputs' for obj_func must be of class 'matrix'.
        Inputted object is of class ", class(outputs))
  } else if (!any(c("matrix", "Matrix", 'dgeMatrix') %in% class(W_k))) {
      stop("Input variable 'W_k' for obj_func must be of class 'matrix'.
        Inputted object is of class ", class(W_k))
  #} else if (ncol(inputs) != ncol(outputs)) {
    #  stop("Input variables 'inputs' and 'outputs' have different number of columns,
    #    implying inconsistent number of covariates in data: ", ncol(inputs), " (inputs) vs ",  ncol(outputs), " (outputs).")
  } else if (ncol(inputs) != nrow(W_k)) {
      stop("Number of columns for input variable 'inputs' does not match number of rows in 'W_k': ", 
        c(ncol(inputs), " vs ", nrow(W_k)), ". Please check construction of 'W_k' by accel_nuclear_gradient().")
  } else if (ncol(outputs) != ncol(W_k)) {
      stop("Number of columns for input variable 'outputs' does not match number of columns in 'W_k': ", 
        c(ncol(outputs), " vs ", ncol(W_k)), ". Please check construction of 'W_k' by accel_nuclear_gradient().")
  } else if (ncol(outputs) > ncol(inputs)) {
      stop("Input variable 'inputs' must have at least as many columns as input variable 'outputs'.")
  }
}