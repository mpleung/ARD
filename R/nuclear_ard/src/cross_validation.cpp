// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "Ji_Ye_eqs.h"
#include "matrix_regression.h"
#include "accelerated_descent.h"
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
double cross_validation(const arma::mat& inputs, 
    const arma::mat& outputs, 
    const SEXP& lambda_sexp, 
    const std::string& Lipschitz = "regression", 
    const int iterations = 5000, 
    const double etol = 1e-5, 
    const double gamma = 2.0, 
    const bool symmetrize = true, 
    const bool fixed_effects = false, 
    const std::vector<double>& CV_grid = std::vector<double>(), 
    const int CV_folds = 5) {

    std::vector<double> lambda_grid = CV_grid;
    // If CV_grid was not provided, create default sequence
    if (lambda_grid.empty()) {
        lambda_grid.reserve(1000);
        for(int i = 1; i <= 1000; ++i) {
            lambda_grid.push_back(i * 0.01);  // Creates sequence from 0.01 to 10
        }
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
        
        arma::uvec trainInd = arma::conv_to<arma::uvec>::from(arma::vec(train_idx));
        arma::uvec testInd = arma::conv_to<arma::uvec>::from(arma::vec(test_idx));

        arma::mat train_inputs = inputs.rows(trainInd);
        arma::mat train_outputs = outputs.rows(trainInd);
        arma::mat test_inputs = inputs.rows(testInd);
        arma::mat test_outputs = outputs.rows(testInd);

        for (size_t lambda_index = 0; lambda_index < lambda_grid.size(); lambda_index++) {
            // Create a new SEXP for the current lambda value
            SEXP lambda_val = Rcpp::wrap(lambda_grid[lambda_index]);
            
            arma::mat fit = accel_nuclear_gradient_cpp(train_inputs, train_outputs, lambda_val, 
                                                     Lipschitz, iterations, etol, gamma, 
                                                     symmetrize, fixed_effects);
            
            arma::mat predicted_valid = fit * test_inputs.t();
            arma::vec errors = vectorise(predicted_valid - test_outputs);
            double mse = arma::mean(arma::square(errors));
            CV_errors(lambda_index, fold) = mse;
        }
    }
    
    arma::vec mean_errors = arma::mean(CV_errors, 1);
    uword best_index = mean_errors.index_min();
    return lambda_grid[best_index];
}