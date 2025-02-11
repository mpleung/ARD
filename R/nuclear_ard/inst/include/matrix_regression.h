#ifndef NUCLEARARD_MATRIXREG_H
#define NUCLEARARD_MATRIXREG_H

#include <RcppArmadillo.h>


arma::vec matrix_OLS(const arma::mat &X, const arma::mat &Y);

#endif
