#ifndef NUCLEARARD_JIYE_H
#define NUCLEARARD_JIYE_H

#include <RcppArmadillo.h>
#include "error_handling.h"
#include "matrix_functions.h"

double obj_func(const arma::mat& inputs, const arma::mat& outputs, const arma::mat& W_k);

arma::mat obj_func_grad(const arma::mat& inputs, const arma::mat& outputs, const arma::mat& W_k);

double obj_func_approx(const arma::mat& inputs, const arma::mat& outputs, const arma::mat& W_k, const arma::mat& W_Kplus1, const double& mu);

arma::mat next_W_func(const arma::mat& inputs, const arma::mat& outputs, const double& lambda, const double& mu, const arma::mat& W_k);

#endif