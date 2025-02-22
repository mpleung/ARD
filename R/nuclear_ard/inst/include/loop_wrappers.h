#ifndef NUCLEARARD_LOOP_WRAPPERS_H
#define NUCLEARARD_LOOP_WRAPPERS_H

#include <RcppArmadillo.h>
#include "Ji_Ye_eqs.h"
#include "matrix_regression.h"

Rcpp::List compute_lipschitz(const arma::mat& inputs, const arma::mat& outputs, double lambda, double L_bar, const arma::mat& Z, const double gamma);

Rcpp::List compute_iteration(const arma::mat& inputs, const arma::mat& outputs,
 const double& lambda, const double& L_bar, arma::mat Z, double alpha, arma::mat W, double etol, bool fixed_effects_bool, arma::rowvec fixed_effects_vector_min1);

#endif 