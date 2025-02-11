#include <RcppArmadillo.h>

// This function solves the generalized matrix regression problem:
//   Y = X * G, where:
//     • Y is a K x M matrix,
//     • X is a K x N matrix (with N > M),
//     • The matrix G (of fixed effects) is N x M and is defined by:
//           if i == j (and j appears as a column, i.e. j=1,…,M):  G[i,j] = 0
//           otherwise: G[i,j] = a[i] + a[j],
//     with the understanding that for j ∈ {1,...,M} the fixed effect used in the jth column 
//     is the same as that for the jth row.
// 
// For an observation corresponding to row k and column j of Y, the model implies:
//   Y(k,j) = ∑_{i=1}^N X(k,i) * G(i,j)
//          = ∑_{i ≠ j} X(k,i) * (a[i] + a[j])
//          = a[j]*[∑_{i=1}^N X(k,i) - X(k,j)] + ∑_{i ≠ j} X(k,i)*a[i].
// 
// Thus, for each observation (k,j) (with j now in {0, 1, ..., M-1}), we build a row of the
// design matrix Z (length N) as follows:
//   - Set the jth component to S[k] - X(k, j)  (with S[k] = ∑_{l=0}^{N-1} X(k,l))
//   - For l ≠ j set the coefficient to X(k, l)
// The global system is then solved by the normal equations.
//
// [[Rcpp::export]]
arma::vec matrix_OLS(const arma::mat &X, const arma::mat &Y) {
  int K = X.n_rows;    // number of observations
  int N = X.n_cols;    // number of fixed effects parameters (length of a)
  int M = Y.n_cols;    // now Y has M columns with M < N
  
  // Compute row sums for X: S(k) = ∑_{l=0}^{N-1} X(k, l)
  arma::vec S = arma::sum(X, 1);
  
  // Create the design matrix Z with (K*M) rows and N columns,
  // and a response vector y_vec of length K*M.
  arma::mat Z(K * M, N, arma::fill::zeros);
  arma::vec y_vec(K * M, arma::fill::zeros);
  
  int idx = 0;
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < M; ++j) {   // loop only over the M observed columns in Y
      for (int l = 0; l < N; ++l) {
        if (l == j) {
          // When the column index matches (for j in {0,...,M-1}),
          // use the special coefficient (row sum minus that entry)
          Z(idx, l) = S(k) - X(k, j);
        } else {
          Z(idx, l) = X(k, l);
        }
      }
      // The corresponding response value is from the (k,j) entry of Y.
      y_vec(idx) = Y(k, j);
      ++idx;
    }
  }
  
  // Solve the normal equations: a = (Z'Z)^{-1} (Z'y_vec)
  arma::vec a_est = arma::solve(Z.t() * Z, Z.t() * y_vec);
  
  // Return the estimated vector a of length N.
  return a_est;
}