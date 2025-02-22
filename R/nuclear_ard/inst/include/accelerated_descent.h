#ifndef NUCLEARARD_ACCELERATED_DESCENT_H
#define NUCLEARARD_ACCELERATED_DESCENT_H

#include <RcppArmadillo.h>

arma::mat accel_nuclear_gradient_cpp(const arma::mat& inputs,
                                   const arma::mat& outputs,
                                   const double lambda_val,
                                   const std::string& Lipschitz = "regression",
                                   const int iterations = 5000,
                                   const double etol = 1e-5,
                                   const double gamma = 2.0,
                                   const bool symmetrized = true,
                                   const bool fixed_effects = false);

#endif 