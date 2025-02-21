// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "Ji_Ye_eqs.h"
#include "matrix_regression.h"
#include "matrix_functions.h"

using namespace Rcpp;
using namespace arma;
using namespace std;



//' Accelerated gradient function for ARD
//'
//' @param inputs A matrix object. ARD census data (K x N).
//' @param outputs A matrix object. ARD survey data (K x M).
//' @param lambda A double or string. Initial lambda value or "NW".
//' @param Lipschitz String. Method for computing Lipschitz constant.
//' @param iterations Integer. Maximum number of iterations.
//' @param etol Double. Error tolerance.
//' @param gamma Double. Step size parameter.
//' @param symmetrize Boolean. Whether to symmetrize the output.
//' @param fixed_effects Boolean. Whether to use fixed effects.
//' @return An N x M matrix estimate of network connections.
//' @export
// [[Rcpp::export]]
arma::mat accel_nuclear_gradient_cpp(const arma::mat& inputs,
                                   const arma::mat& outputs,
                                   const SEXP& lambda_sexp,
                                   const std::string& Lipschitz = "regression",
                                   const int iterations = 5000,
                                   const double etol = 1e-5,
                                   const double gamma = 2.0,
                                   const bool symmetrize = true,
                                   const bool fixed_effects = false) {
    
    // Get dimensions
    int N = inputs.n_cols;  // village size
    int M = outputs.n_cols; // number of households
    int K = outputs.n_rows; // number of characteristics
    
    // Initialize lambda
    double lambda_val;
    if (TYPEOF(lambda_sexp) == STRSXP && Rcpp::as<std::string>(lambda_sexp) == "NW") {
        // If lambda is "NW", compute it
        lambda_val = 2.0 * (std::sqrt(M) + std::sqrt(N) + 1.0) * (std::sqrt(N) + std::sqrt(K));
    } else if (TYPEOF(lambda_sexp) == REALSXP) {
        lambda_val = as<double>(lambda_sexp);
    } else {
        throw std::runtime_error("Invalid lambda option. Use 'NW' or a numeric value.");
    }
    
    // Compute Lipschitz constant
    double L;
    if (Lipschitz == "regression") {
        // For regression case, use eigenvalue
        arma::mat XXt = inputs * inputs.t();
        arma::vec eigval = arma::eig_sym(XXt);
        L = eigval(eigval.n_elem - 1);  // largest eigenvalue
    } else if (Lipschitz == "JiYe") {
        L = 1.0;  // Initial value for JiYe method
    } else {
        throw std::runtime_error("Invalid Lipschitz option. Use 'regression' or 'JiYe'.");
    }

    // Initialize scalar values
    double alpha = 1.0;
    
    // Initialize matrices
    arma::mat Z = arma::randu(N, M);  // Random uniform initialization
    if (symmetrize) {
        Z = symmetrize(Z);
    }
    
    arma::mat W = Z;
    arma::rowvec fixed_effects_vector = arma::zeros<arma::rowvec>(N);
    
    // Main iteration loop
    for (int i = 0; i < iterations; i++) {
        // Update Lipschitz constant if using JiYe method
        if (Lipschitz == "JiYe") {
            Rcpp::List JiYe_values = compute_lipschitz(inputs, outputs, lambda_val, L, Z, gamma);
            L = as<double>(JiYe_values["L_bar"]);
            lambda_val = as<double>(JiYe_values["lambda"]);
        }
        
        // Update iteration values
        Rcpp::List value_iterator = compute_iteration(inputs, outputs, lambda_val, L, Z, alpha, W, etol, 
                                                    fixed_effects, fixed_effects_vector);
        
        W = as<arma::mat>(value_iterator["W"]);
        alpha = as<double>(value_iterator["alpha"]);
        Z = as<arma::mat>(value_iterator["Z"]);
        bool flag = as<bool>(value_iterator["flag"]);
        fixed_effects_vector = as<arma::rowvec>(value_iterator["fixed_effects_vector"]);
        
        if (flag) break;
    }
    
    // Handle fixed effects
    if (fixed_effects) {
        arma::mat fixed_effects_rows_mat = arma::repmat(fixed_effects_vector, N, 1);
        arma::mat fixed_effects_cols_mat = fixed_effects_rows_mat.t();
        arma::mat fixed_effects_matrix = fixed_effects_rows_mat + fixed_effects_cols_mat;
        fixed_effects_matrix.diag().zeros();
        W += fixed_effects_matrix;
    }
    
    // Final symmetrization if requested
    if (symmetrize) {
        W = symmetrize_matrix(W);
    }
    
    // Ensure non-negative values
    W = arma::max(W, arma::zeros(size(W)));
    
    return W;
} 