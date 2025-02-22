// [[Rcpp::depends(RcppArmadillo)]]
#include <Rcpp.h>
#include <RcppArmadillo.h>
#include "Ji_Ye_eqs.h"
#include "matrix_regression.h"
#include "matrix_functions.h"
#include <algorithm>    // for std::shuffle and std::fill
#include <random>       // for std::random_device and std::mt19937


using namespace Rcpp;
using namespace arma;
using namespace std;


//' Cross validation function for ARD
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
//' @param CV_grid A vector. This is the grid of lambda values to use for cross-validation. Set by default to seq(0.01, 10, by=0.01).
//' @param CV_folds A scalar (integer) value. This is the number of folds to use for cross-validation. Set by default to 5.
//' @return A double. The optimal lambda value to use for network estimation. 
//' @export
// [[Rcpp::export]]
arma::mat cross_validation(const arma::mat& inputs, 
const arma::mat& outputs, 
const SEXP& lambda_sexp, 
const std::string& Lipschitz = "regression", 
const int iterations = 5000, 
const double etol = 1e-5, 
const double gamma = 2.0, 
const bool symmetrize = true, 
const bool fixed_effects = false, 
const std::vector<double>& CV_grid = std::vector<double>(seq(0.01, 10, by=0.01)), 
const int CV_folds = 5) {
    int K = inputs.n_rows;
    if (CV_folds > K) {
        stop("CV_folds must be less than the number of ARD traits.");
    }
    
    // Create fold assignments (equivalent to sample(rep(1:CV_folds, length.out = K)))
    IntegerVector sample_indices(K);
    for (int i = 0; i < K; i++) {
        sample_indices[i] = (i % CV_folds) + 1;
    }
    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sample_indices.begin(), sample_indices.end(), g);

    arma::mat CV_errors(CV_grid.size(), CV_folds);
    for (int fold = 0; fold < CV_folds; fold++) {
        IntegerVector train_indices = sample_indices[sample_indices != fold];
        IntegerVector test_indices = sample_indices[sample_indices == fold];

        arma::mat train_inputs = inputs.rows(train_indices);
        arma::mat train_outputs = outputs.rows(train_indices);
        arma::mat test_inputs = inputs.rows(test_indices);
        arma::mat test_outputs = outputs.rows(test_indices);

        for (int lambda_index = 0; lambda_index < CV_grid.size(); lambda_index++) {
            double lambda = CV_grid[lambda_index];
            arma::mat fit = accel_nuclear_gradient_cpp(train_inputs, train_outputs, lambda, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects);
            arma::mat predicted_valid = fit * test_inputs.t();
            double error_valid = arma::mean(arma::square(predicted_valid - test_outputs));
            CV_errors(lambda_index, fold) = error_valid;
        }
    }
    double best_lambda = CV_grid[arma::index_min(arma::mean(CV_errors, 1))];
    return best_lambda;
}