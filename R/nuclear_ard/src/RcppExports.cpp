// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// accel_nuclear_gradient_cpp
arma::mat accel_nuclear_gradient_cpp(const arma::mat& inputs, const arma::mat& outputs, const double lambda_val, const std::string& Lipschitz, const int iterations, const double etol, const double gamma, const bool symmetrized, const bool fixed_effects);
RcppExport SEXP _nuclearARD_accel_nuclear_gradient_cpp(SEXP inputsSEXP, SEXP outputsSEXP, SEXP lambda_valSEXP, SEXP LipschitzSEXP, SEXP iterationsSEXP, SEXP etolSEXP, SEXP gammaSEXP, SEXP symmetrizedSEXP, SEXP fixed_effectsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type inputs(inputsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type outputs(outputsSEXP);
    Rcpp::traits::input_parameter< const double >::type lambda_val(lambda_valSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type Lipschitz(LipschitzSEXP);
    Rcpp::traits::input_parameter< const int >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< const double >::type etol(etolSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< const bool >::type symmetrized(symmetrizedSEXP);
    Rcpp::traits::input_parameter< const bool >::type fixed_effects(fixed_effectsSEXP);
    rcpp_result_gen = Rcpp::wrap(accel_nuclear_gradient_cpp(inputs, outputs, lambda_val, Lipschitz, iterations, etol, gamma, symmetrized, fixed_effects));
    return rcpp_result_gen;
END_RCPP
}
// compute_lipschitz
List compute_lipschitz(const arma::mat& inputs, const arma::mat& outputs, double lambda, double L_bar, const arma::mat& Z, const double gamma);
RcppExport SEXP _nuclearARD_compute_lipschitz(SEXP inputsSEXP, SEXP outputsSEXP, SEXP lambdaSEXP, SEXP L_barSEXP, SEXP ZSEXP, SEXP gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type inputs(inputsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type outputs(outputsSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type L_bar(L_barSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_lipschitz(inputs, outputs, lambda, L_bar, Z, gamma));
    return rcpp_result_gen;
END_RCPP
}
// compute_iteration
List compute_iteration(const arma::mat& inputs, const arma::mat& outputs, const double& lambda, const double& L_bar, arma::mat Z, double alpha, arma::mat W, double etol, bool fixed_effects_bool, arma::rowvec fixed_effects_vector_min1);
RcppExport SEXP _nuclearARD_compute_iteration(SEXP inputsSEXP, SEXP outputsSEXP, SEXP lambdaSEXP, SEXP L_barSEXP, SEXP ZSEXP, SEXP alphaSEXP, SEXP WSEXP, SEXP etolSEXP, SEXP fixed_effects_boolSEXP, SEXP fixed_effects_vector_min1SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type inputs(inputsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type outputs(outputsSEXP);
    Rcpp::traits::input_parameter< const double& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const double& >::type L_bar(L_barSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type W(WSEXP);
    Rcpp::traits::input_parameter< double >::type etol(etolSEXP);
    Rcpp::traits::input_parameter< bool >::type fixed_effects_bool(fixed_effects_boolSEXP);
    Rcpp::traits::input_parameter< arma::rowvec >::type fixed_effects_vector_min1(fixed_effects_vector_min1SEXP);
    rcpp_result_gen = Rcpp::wrap(compute_iteration(inputs, outputs, lambda, L_bar, Z, alpha, W, etol, fixed_effects_bool, fixed_effects_vector_min1));
    return rcpp_result_gen;
END_RCPP
}
// nuclear_norm
double nuclear_norm(const arma::mat& matrix);
RcppExport SEXP _nuclearARD_nuclear_norm(SEXP matrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type matrix(matrixSEXP);
    rcpp_result_gen = Rcpp::wrap(nuclear_norm(matrix));
    return rcpp_result_gen;
END_RCPP
}
// symmetrize
arma::mat symmetrize(const arma::mat& matrix);
RcppExport SEXP _nuclearARD_symmetrize(SEXP matrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type matrix(matrixSEXP);
    rcpp_result_gen = Rcpp::wrap(symmetrize(matrix));
    return rcpp_result_gen;
END_RCPP
}
// matrix_OLS
arma::vec matrix_OLS(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP _nuclearARD_matrix_OLS(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(matrix_OLS(X, Y));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_nuclearARD_accel_nuclear_gradient_cpp", (DL_FUNC) &_nuclearARD_accel_nuclear_gradient_cpp, 9},
    {"_nuclearARD_compute_lipschitz", (DL_FUNC) &_nuclearARD_compute_lipschitz, 6},
    {"_nuclearARD_compute_iteration", (DL_FUNC) &_nuclearARD_compute_iteration, 10},
    {"_nuclearARD_nuclear_norm", (DL_FUNC) &_nuclearARD_nuclear_norm, 1},
    {"_nuclearARD_symmetrize", (DL_FUNC) &_nuclearARD_symmetrize, 1},
    {"_nuclearARD_matrix_OLS", (DL_FUNC) &_nuclearARD_matrix_OLS, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_nuclearARD(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
