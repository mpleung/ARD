#ifndef NUCLEARARD_CROSS_VALID_H
#define NUCLEARARD_CROSS_VALID_H

#include <RcppArmadillo.h>
#include "accelerated_descent.h"
#include "Ji_Ye_eqs.h"
#include "matrix_regression.h"
#include "matrix_functions.h"
#include <algorithm>
#include <random>

double cross_validation_cpp(const arma::mat& inputs,
                          const arma::mat& outputs,
                          const std::string& Lipschitz = "regression",
                          const int iterations = 5000,
                          const double etol = 1e-5,
                          const double gamma = 2.0,
                          const bool symmetrized = true,
                          const bool fixed_effects = false,
                          const Rcpp::NumericVector CV_grid = Rcpp::NumericVector::create(),
                          const int CV_folds = 5);

#endif 