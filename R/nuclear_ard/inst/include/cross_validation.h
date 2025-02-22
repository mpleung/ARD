#ifndef NUCLEARARD_CROSS_VALIDATION_H
#define NUCLEARARD_CROSS_VALIDATION_H

#include <RcppArmadillo.h>

double cross_validation(const arma::mat& inputs,
                                   const arma::mat& outputs,
                                   const SEXP& lambda_sexp,
                                   const std::string& Lipschitz = "regression",
                                   const int iterations = 5000,
                                   const double etol = 1e-5,
                                   const double gamma = 2.0,
                                   const bool symmetrize = true,
                                   const bool fixed_effects = false,
                                   const std::vector<double>& CV_grid = std::vector<double>(seq(0.01, 10, by=0.01)),
                                   const int CV_folds = 5);

#endif 