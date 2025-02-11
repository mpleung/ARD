// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "Ji_Ye_eqs.h"
#include "matrix_regression.h"
using namespace Rcpp;
using namespace arma;
using namespace std;


// This file creates functions that act as wrappers for the inner 
// operations of loops in the function accel_nuclear_gradient().
// Specifically, it contains two functions. The first computationally
// calculates the Lipschitz constant L when analytical methods aren't
// available. The second function is a wrapper for updating the 
// various state variables in the loop until the next iteration. 

// ' Compute Lipschitz constant
// ' 
// ' \code{compute_lipschitz} calculates the lipschitz constant
// \ using the method of Ji and Ye (2009) when doing so analytically
// ' is not feasible. 
// ' 
// ' @param inputs A matrix object. This contains ARD census data in matrix form.
// '  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
// ' @param outputs A matrix object. This contains the ARD survey data
// ' in matrix form with size K x M, where N > M = number of households receiving ARD questionairre.
// ' @param lambda A scalar (numeric) value. This is an initial guess that will be iterated on. It can
// '  alternatively be defined as 'NW', which will set \eqn{\lambda = 2(\sqrt{N} + \sqrt{M})(\sqrt{N} + \sqrt{K})}.
// ' @param L_bar A scalar (numeric) value. This is the Lipschitz Guess for the current iteration.
// ' @param Z A matrix object of dimension N x M, where N = village size and M = number of households receiving ARD questionairre.
// ' @param gamma A scalar (numeric) value.
// '
// ' @return A list containing the new values for lambda and L_bar. Both are scalars.
// [[Rcpp::export]]

List compute_lipschitz(const arma::mat& inputs, const arma::mat& outputs, double lambda, double L_bar, const arma::mat& Z, const double gamma) {
    // Update W
    arma::mat plZ = next_W_func(inputs, outputs, lambda, L_bar, Z);

    // Step 2: While loop only activated if
    // F(p_L(Z_{k - 1})) > Q(p_L(Z_{k-1}), Z_{k-1}).
    // Removing \lambda ||W||_* from both sides since cancels out,
    // nuclear norm computation isn't super efficient, and don't
    // calculate F or Q elsewhere, so don't need generality.
    // Effectively, this clause says only keep iterating if actual
    // loss value is still greater than approximated loss, because
    // that means we can do better by iterating on the value of L.
    double F_value = obj_func(inputs, outputs, plZ);
    double Q_value = obj_func_approx(inputs, outputs, Z, plZ, L_bar);

    int j = 1;

    while (F_value > Q_value) {
        // Update L bar, as stated in algorithm.
        L_bar = gamma * L_bar;

        // Recalculate values for while loop check.
        plZ = next_W_func(inputs, outputs, lambda, L_bar, Z);
        F_value = obj_func(inputs, outputs, plZ);
        Q_value = obj_func_approx(inputs, outputs, Z, plZ, L_bar);
        j++;
    }

    return Rcpp::List::create(Rcpp::Named("lambda") = lambda, 
                            Rcpp::Named("L_bar") = L_bar);


}


// ' Compute iteration
// ' 
// ' \code{compute_gradient_iteration} calculates the values from a single
// \ iteration of the gradient descent algorithm in Ji and Ye (2009).
// ' 
// ' @param inputs A matrix object. This contains ARD census data in matrix form.
// '  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
// ' @param outputs A matrix object. This contains the ARD survey data
// ' in matrix form with size K x M, where N > M = number of households receiving ARD questionairre.
// ' @param lambda A scalar (numeric) value. This is an initial guess that will be iterated on. It can
// '  alternatively be defined as 'NW', which will set \eqn{\lambda = 2(\sqrt{N} + \sqrt{M})(\sqrt{N} + \sqrt{K})}.
// ' @param L_bar A scalar (numeric) value. This is the Lipschitz Guess for the current iteration.
// ' @param Z A matrix object of dimension N x M, where N = village size and M = number of households receiving ARD questionairre.
// ' @param alpha A scalar (numeric) value.
// ' @param W A matrix object. This is the current iteration's guess for the ideal N x M linear operator.
// ' @param etol A scalar (numeric) value. This is the error tolerance for the algorithm.
// '
// ' @return A list containing the new values for lambda and L_bar. Both are scalars.
// [[Rcpp::export]]

List compute_iteration(const arma::mat& inputs, const arma::mat& outputs,
 const double& lambda, const double& L_bar, arma::mat Z, double alpha, arma::mat W, double etol, bool fixed_effects_bool, arma::rowvec fixed_effects_vector_min1) {

    // Update values before next iteration.
    arma::mat W_kmin1 = W;

    // Get network size
    int N = W.n_rows;
    
    arma::rowvec fixed_effects_vector = arma::zeros<arma::rowvec>(N);

    // If fixed effects are wanted, we need to create updated output matrix for next_W_func
    
    if (fixed_effects_bool == TRUE) {
        arma::mat outputs_fixed_effects = outputs;
        // create matrix of fixed effects. 
        arma::mat fixed_effects_matrix = arma::repmat(fixed_effects_vector_min1, N, 1);

        fixed_effects_matrix = fixed_effects_matrix + fixed_effects_matrix.t();

        // make diagonal 0
        fixed_effects_matrix.diag().fill(0);
        // subtract fixed effects matrix from outputs
        arma::mat outputs_adjusted_by_fixed_effects = outputs - inputs * fixed_effects_matrix;



        // Get next W
        W = next_W_func(inputs, outputs_adjusted_by_fixed_effects, lambda, L_bar, Z);

        // Get new fixed effects vector
        arma::mat outputs_adjusted_by_W = outputs - inputs * W;
        fixed_effects_vector = matrix_OLS(inputs, outputs_adjusted_by_W).t();

    } else {
      W = next_W_func(inputs, outputs, lambda, L_bar, Z);
    }

    // if (symmetrize == TRUE) { // commenting this out for acceleration. Only symmetrizing at end of algorithm.

    // }




    double alpha_kmin1 = alpha;
    alpha = (1.0 + sqrt(1.0 + 4.0 * alpha * alpha)) / 2.0;
    Z = W + ((alpha_kmin1 - 1.0) / alpha) * (W - W_kmin1);


    double errorW = arma::mean(arma::mean(arma::abs(W_kmin1 - W)));
    double errorFE = arma::mean(arma::mean(arma::abs(fixed_effects_vector_min1 - fixed_effects_vector)));
    bool flag = FALSE;
    if (errorW < etol && errorFE < etol) {
      flag = TRUE;
    }

    return Rcpp::List::create(Rcpp::Named("W") = W,
                            Rcpp::Named("alpha") = alpha,
                            Rcpp::Named("Z") = Z,
                            Rcpp::Named("flag") = flag,
                            Rcpp::Named("fixed_effects_vector") = fixed_effects_vector);
 }