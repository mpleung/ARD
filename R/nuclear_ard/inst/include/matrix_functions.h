#ifndef NUCLEARARD_MATRIX_H
#define NUCLEARARD_MATRIX_H

#include <RcppArmadillo.h>
#include "error_handling.h"


double nuclear_norm(const arma::mat& matrix);

arma::mat symmetrize(const arma::mat& matrix);


#endif

