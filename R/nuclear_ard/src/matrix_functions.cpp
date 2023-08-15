// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "error_handling.h"

using namespace Rcpp;
using namespace arma;
using namespace std;


// This file contains general purpose functions on matrices. Specifically,
// the nuclear norm of a matrix and the trace of a matrix. It also contains
// a symmetrize function in order to implement the modified accelerated 
// gradient descent, proposed in Alidaee, Auerbach, and Leung (2019).

// ' Nuclear norm of a matrix
// ' 
// ' \code{nuclear_norm} calculates the nuclear norm 
// ' for a single matrix. This is the Schatten norm in
// ' the case of p = 1.
// ' 
// ' Because this is a nuclear norm, the function requires
// ' input to be a matrix to avoid incorrect implementation.
// ' 
// ' @seealso More information on the nuclear norm can be found at 
// '  \url{https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms}.
// ' 
// ' 
// ' @param matrix A matrix object. This can be a dcgMatrix from Matrix package.
// ' @return The nuclear norm of \code{mat}. This is a numeric, scalar value.
// ' @examples
// ' nuclear_norm(matrix(1, nrow=3, ncol=3))
// ' 
// ' A <- matrix(2, nrow=4, ncol=2)
// ' nuclear_norm(A)
// ' 
// ' B <- Matrix::Matrix(1, nrow=3, ncol=3)
// ' nuclear_norm(B)
// ' 
// ' \dontrun{
// ' nuclear_norm(c(A, B))
// ' nuclear_norm(c(1,2,3))
// ' }
// [[Rcpp::export]]
double nuclear_norm(const arma::mat& matrix) {
    arma::mat u;
    arma::vec s;
    arma::mat v;
    arma::svd(u, s, v, matrix);
    double nucnorm = arma::sum(s);

    return nucnorm;
}

// ' Symmetrize a matrix
// ' 
// ' \code{symmetrize} transforms an m x n matrix, where m > n, to be 
// ' symmetric in the top n x n submatrix, with zero diagonals.
// ' 
// ' 
// ' @param matrix A matrix object. This can be a dcgMatrix from Matrix package. Must be have square dimensions.
// ' @return The symmetric transfomation of \code{mat}. 
// ' @examples
// ' symmetrize(matrix(1, nrow=3, ncol=3))
// ' 
// ' A <- matrix(2, nrow=6, ncol=4)
// ' symmetrize(A)
// ' 
// ' \dontrun{
// ' A <- matrix(2, nrow=2, ncol=4)
// ' symmetrize(A)
// ' }
// [[Rcpp::export]]
arma::mat symmetrize(const arma::mat& matrix) {

  // Determine number of columns, which we will use for indexing.
  int N = matrix.n_cols;
  // First, zero out negative entries.
  arma::mat mat_symmetric = abs(matrix)/2.0 + matrix/2.0;
  // Next, isolate the top submatrix.
  arma::mat mat_submatrix = matrix.submat(0, 0, N-1, N-1);
  // Take average of the off-diagonal pairs in submatrix
  // in order to balance out directions.
  mat_submatrix = mat_submatrix/2.0 + mat_submatrix.t()/2.0;
  // Replace diagonals of submatrix with zeros.
  mat_submatrix.diag().zeros();
  // Insert submatrix into larger matrix to complete symmetrization.
  mat_symmetric.submat(0, 0, N-1, N-1) = mat_submatrix;
  
  return mat_symmetric;
}