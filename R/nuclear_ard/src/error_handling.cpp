// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;


// ' Input validation for objective function 
// ' 
// ' \code{\link{obj_func}} and \code{\link{obj_func.grad}} have common inputs
// ' and therefore require the same validations. To avoid code redundancy, these
// ' common checks have been placed in this function, which is then referenced.
// ' 
// ' @param inputs A matrix object. This contains ARD census data in matrix form.
// '  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
// ' @param outputs A matrix object. This contains the ARD survey data 
// '  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre. 
// ' @param W_k A matrix object. This is the current iteration's guess for the ideal N x M linear operator. 
// '  Users do not need to create this themselves. It is provided via \code{\link{next_W_func}}. 
// ' @return There is no returned value. This function solely raises errors in the event of invalid inputs. 

bool obj_func_testing(arma::mat inputs, arma::mat outputs, arma::mat W_k) {
  // Before proceeding, must check that mat meets requirements.
    // 1. inputs and outputs must have same number of columns
    // 2. inputs and outputs must have same number of rows as W_k
    // 3. outputs must have fewer columns than inputs

    if(inputs.n_cols != outputs.n_cols) {
      Rcpp::stop("Input variables 'inputs' and 'outputs' have different number of columns, implying inconsistent number of covariates in data: ", inputs.n_cols, " (inputs) vs ",  outputs.n_cols, " (outputs).");
    } else if (inputs.n_cols != W_k.n_rows) {
      Rcpp::stop("Number of columns for input variable 'inputs' does not match number of rows in 'W_k': ", 
        inputs.n_cols, " vs ", W_k.n_rows, ". Please check construction of 'W_k' by accel_nuclear_gradient().");
    } else if (outputs.n_cols != W_k.n_cols) {
      Rcpp::stop("Number of columns for input variable 'outputs' does not match number of columns in 'W_k': ", 
        outputs.n_cols, " vs ", W_k.n_cols, ". Please check construction of 'W_k' by accel_nuclear_gradient().");
    } else if (outputs.n_cols > inputs.n_cols) {
      Rcpp::stop("Input variable 'inputs' must have at least as many columns as input variable 'outputs'.");
    } else {
      return true;
    }
}

