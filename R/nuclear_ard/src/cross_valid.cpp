// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "cross_valid.h"
#include "accelerated_descent.h"
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
//' @param Lipschitz String. Method for computing Lipschitz constant.
//' @param iterations Integer. Maximum number of iterations.
//' @param etol Double. Error tolerance.
//' @param gamma Double. Step size parameter.
//' @param symmetrized Boolean. Whether to symmetrize the output.
//' @param fixed_effects Boolean. Whether to use fixed effects.
//' @param CV_grid NumericVector. Grid of lambda values to use for cross-validation. If not provided, defaults to a sequence from 0.01 to 10 with step size 0.01.
//' @param CV_folds Integer. Number of folds to use for cross-validation. Defaults to 5.
//' @return A double. The optimal lambda value to use for network estimation. 
//' @export
// [[Rcpp::export]]
double cross_validation_cpp(const arma::mat& inputs, 
    const arma::mat& outputs, 
    const std::string& Lipschitz,
    const int iterations,
    const double etol,
    const double gamma,
    const bool symmetrized,
    const bool fixed_effects,
    const Rcpp::NumericVector CV_grid,
    const int CV_folds) {

    // Convert NumericVector to std::vector<double> for internal use
    std::vector<double> lambda_grid;
    if (CV_grid.length() == 0) {
        lambda_grid.reserve(1000);
        for(int i = 1; i <= 1000; ++i) {
            lambda_grid.push_back(i * 0.01);  // Creates sequence from 0.01 to 10
        }
    } else {
        lambda_grid.assign(CV_grid.begin(), CV_grid.end());
    }

    int K = inputs.n_rows;
    if (CV_folds > K) {
        stop("CV_folds must be less than the number of ARD traits.");
    }
    
    // Create fold assignments
    IntegerVector fold_indices(K);
    for (int i = 0; i < K; i++) {
        fold_indices[i] = (i % CV_folds) + 1;
    }
    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(fold_indices.begin(), fold_indices.end(), g);

    arma::mat CV_errors(lambda_grid.size(), CV_folds);
    for (int fold = 0; fold < CV_folds; fold++) {
        // Create train/test indices
        std::vector<int> train_idx, test_idx;
        for(int i = 0; i < K; i++) {
            if(fold_indices[i] == fold + 1) {
                test_idx.push_back(i);
            } else {
                train_idx.push_back(i);
            }
        }
        
        // Convert indices to uvec using arma::uvec constructor
        arma::uvec trainInd(train_idx.size());
        arma::uvec testInd(test_idx.size());
        for(size_t i = 0; i < train_idx.size(); ++i) trainInd[i] = train_idx[i];
        for(size_t i = 0; i < test_idx.size(); ++i) testInd[i] = test_idx[i];

        arma::mat train_inputs = inputs.rows(trainInd);
        arma::mat train_outputs = outputs.rows(trainInd);
        arma::mat test_inputs = inputs.rows(testInd);
        arma::mat test_outputs = outputs.rows(testInd);


        for (size_t lambda_index = 0; lambda_index < lambda_grid.size(); lambda_index++) {
            // Create a new SEXP for the current lambda value
            double lambda_val = lambda_grid[lambda_index];
            
            // Fit model on training data
            arma::mat fit = accel_nuclear_gradient_cpp(train_inputs, train_outputs, lambda_val, Lipschitz, iterations, etol, gamma, symmetrized, fixed_effects);
            
            arma::mat predicted_valid = fit * test_inputs.t();

            arma::vec errors = vectorise(predicted_valid.t() - test_outputs);
            double mse = arma::mean(arma::square(errors));
            CV_errors(lambda_index, fold) = mse;
        }
    }
    
    arma::vec mean_errors = arma::mean(CV_errors, 1);
    uword best_index = mean_errors.index_min();
    return lambda_grid[best_index];
}