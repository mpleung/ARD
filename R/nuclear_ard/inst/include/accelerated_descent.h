#ifndef NUCLEARARD_ACCELERATED_DESCENT_H
#define NUCLEARARD_ACCELERATED_DESCENT_H

#include <RcppArmadillo.h>

arma::mat symmetrize_matrix(const arma::mat& W);

arma::mat accel_nuclear_gradient_cpp(const arma::mat& inputs,
                                   const arma::mat& outputs,
                                   const SEXP& lambda_sexp,
                                   const std::string& Lipschitz = "regression",
                                   const int iterations = 5000,
                                   const double etol = 1e-5,
                                   const double gamma = 2.0,
                                   const bool symmetrize = true,
                                   const bool fixed_effects = false);

#endif 