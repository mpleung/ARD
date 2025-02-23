#' Cross Validation for ARD (R Implementation)
#'
#' \code{cross_validation} implements the cross validation algorithm outlined in Alidaee et al. (2020)
#' for selecting the optimal lambda value in ARD network estimation.
#'
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#' @param outputs A matrix object. This contains the ARD survey data in matrix form.
#' @param Lipschitz A string. Method for computing the Lipschitz constant ("regression" or "JiYe").
#' @param iterations An integer. Maximum number of iterations.
#' @param etol A double. Error tolerance for convergence.
#' @param gamma A double. Step size parameter.
#' @param symmetrized A boolean. Whether to symmetrize the output.
#' @param fixed_effects A boolean. Whether to use fixed effects.
#' @param CV_grid A NumericVector. Grid of lambda values to use for cross-validation.
#' @param CV_folds An integer. Number of folds to use for cross-validation.
#' @return A double. The optimal lambda value to use for network estimation.
#' @export
cross_validation <- function(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrized, fixed_effects, CV_grid, CV_folds) {
    # Ensure CV_grid is numeric
    lambda_grid <- as.numeric(CV_grid)

    # Detect number of cores available
    num_cores <- detectCores() - 1

    # Number of observations
    K <- nrow(inputs)

    all_pairs <- combn(K, 2)

    # Create fold assignments and shuffle them
    fold_indices <- all_pairs[, sample(ncol(all_pairs), CV_folds)]

    # Initialize matrix to store cross validation errors (rows: lambda values, cols: folds)
    CV_errors <- matrix(NA, nrow = length(lambda_grid), ncol = CV_folds)

    # Loop over each fold
    for (fold in 1:CV_folds) {
        cat(sprintf("\rProcessing fold %d of %d", fold, CV_folds))
        flush.console()

        # Identify training and testing indices based on fold
        test_idx <- fold_indices[, fold]
        train_idx <- setdiff(1:K, test_idx)

        train_inputs <- inputs[train_idx, , drop = FALSE]
        train_outputs <- outputs[train_idx, , drop = FALSE]
        test_inputs <- inputs[test_idx, , drop = FALSE]
        test_outputs <- outputs[test_idx, , drop = FALSE]

        plan(multisession, workers = num_cores)

        # Parallelize the loop over lambda values
        fold_errors <- future_map(1:length(lambda_grid), function(l) {
            lambda_val <- lambda_grid[l]

            # Fit the model on training data. Replace accel_nuclear_gradient with your R implementation.
            fit <- accel_nuclear_gradient(train_inputs, train_outputs, lambda_val, Lipschitz, iterations, etol, gamma, symmetrized, fixed_effects)

            # Predict on test data. If fit is a matrix, then prediction is computed as described.
            # In C++: predicted_valid = t(fit %*% t(test_inputs)).
            predicted_valid <- t(fit %*% t(test_inputs))

            # Compute mean squared error between predictions and true test outputs
            mse <- mean((predicted_valid - test_outputs)^2)

            return(mse)
        })
        plan(sequential)

        # Store the errors for this fold
        CV_errors[, fold] <- unlist(fold_errors)
    }
    cat("\n")

    # Compute mean error for each lambda value across folds
    mean_errors <- rowMeans(CV_errors)
    best_index <- which.min(mean_errors)
    var_errors <- var(mean_errors)
    min_error <- min(mean_errors)

    return(list(best_lambda = lambda_grid[best_index], min_error = min_error, var_errors = var_errors))
}


#' Cross Validation for ARD (R Implementation)
#'
#' \code{cross_validation} implements the cross validation algorithm outlined in Alidaee et al. (2020)
#' for selecting the optimal lambda value in ARD network estimation.
#'
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#' @param outputs A matrix object. This contains the ARD survey data in matrix form.
#' @param Lipschitz A string. Method for computing the Lipschitz constant ("regression" or "JiYe").
#' @param iterations An integer. Maximum number of iterations.
#' @param etol A double. Error tolerance for convergence.
#' @param gamma A double. Step size parameter.
#' @param symmetrized A boolean. Whether to symmetrize the output.
#' @param fixed_effects A boolean. Whether to use fixed effects.
#' @param CV_grid A NumericVector. Grid of lambda values to use for cross-validation.
#' @param CV_folds An integer. Number of folds to use for cross-validation.
#' @return A double. The optimal lambda value to use for network estimation.
#' @export
cross_validation_mse <- function(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrized, fixed_effects, lambda, CV_folds) {
    # Ensure CV_grid is numeric

    # Detect number of cores available
    num_cores <- detectCores() - 1

    # Number of observations
    K <- nrow(inputs)

    # Create fold assignments and shuffle them
    all_pairs <- combn(K, 2)

    # Create fold assignments and shuffle them
    fold_indices <- all_pairs[, sample(ncol(all_pairs), CV_folds)]

    plan(multisession, workers = num_cores)
    cv_errors <- future_map(1:CV_folds, function(fold) {
        # Identify training and testing indices based on fold
        test_idx <- fold_indices[, fold]
        train_idx <- setdiff(1:K, test_idx)

        train_inputs <- inputs[train_idx, , drop = FALSE]
        train_outputs <- outputs[train_idx, , drop = FALSE]
        test_inputs <- inputs[test_idx, , drop = FALSE]
        test_outputs <- outputs[test_idx, , drop = FALSE]



        # Fit the model on training data. Replace accel_nuclear_gradient with your R implementation.
        fit <- accel_nuclear_gradient(train_inputs, train_outputs, lambda, Lipschitz, iterations, etol, gamma, symmetrized, fixed_effects)

        # Predict on test data. If fit is a matrix, then prediction is computed as described.
        # In C++: predicted_valid = t(fit %*% t(test_inputs)).
        predicted_valid <- t(fit %*% t(test_inputs))

        # Compute mean squared error between predictions and true test outputs
        mse <- mean((predicted_valid - test_outputs)^2)

        return(mse)
    })
    plan(sequential)

    # Compute mean error for each lambda value across folds
    mean_error <- mean(unlist(cv_errors))


    return(mean_error)
}
