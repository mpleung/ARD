#include <RcppArmadillo.h>


// This function solves the matrix regression problem:
//   Y = X * G,  where G is defined by:
//       G[i, j] = a[i] + a[j] for i != j, and G[i,i] = 0.
// The function returns the estimated vector 'a' of length N.
// The design matrix Z is built row-wise for each (k, j) observation:
//   For observation (k, j):
//     - For l == j: coefficient = (row sum of X[k,]) - X(k, j)
//     - For l != j: coefficient = X(k, l)
// We then solve the system using the normal equations.
// 
// [[Rcpp::export]]
arma::vec matrix_OLS(const arma::mat &X, const arma::mat &Y) {
  int K = X.n_rows;   // Number of rows in X (and Y)
  int N = X.n_cols;   // Number of columns in X (and Y), corresponds to the length of vector 'a'
  
  // Compute the row sums of X. For each k, S[k] = sum_{l=1}^{N} X(k,l)
  arma::vec S = arma::sum(X, 1);
  
  // Create the design matrix Z with (K*N) rows and N columns,
  // and a response vector y_vec of length K*N.
  arma::mat Z(K * N, N, arma::fill::zeros);
  arma::vec y_vec(K * N, arma::fill::zeros);
  
  int idx = 0;
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < N; ++j) {
      // For the (k,j) observation, fill the corresponding row in Z.
      for (int l = 0; l < N; ++l) {
        if (l == j) {
          Z(idx, l) = S(k) - X(k, j);
        } else {
          Z(idx, l) = X(k, l);
        }
      }
      // Set the corresponding entry in the response vector from Y.
      y_vec(idx) = Y(k, j);
      ++idx;
    }
  }
  
  // Solve the normal equations: a = (Z'Z)^{-1} * (Z'y_vec)
  arma::vec a_est = arma::solve(Z.t() * Z, Z.t() * y_vec);
  
  // Return the estimated a vector of length N.
  return a_est;
}