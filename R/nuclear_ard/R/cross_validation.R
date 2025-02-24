#' Cross Validation for ARD (R Implementation)
#'
#' \code{cross_validation_combined} implements the cross validation algorithm outlined in Alidaee et al. (2020)
#' for selecting the optimal lambda value in ARD network estimation or computing mean squared error using cross validation.
#'
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#' @param outputs A matrix object. This contains the ARD survey data in matrix form.
#' @param Lipschitz A string. Method for computing the Lipschitz constant ("regression" or "JiYe").
#' @param iterations An integer. Maximum number of iterations.
#' @param etol A double. Error tolerance for convergence.
#' @param gamma A double. Step size parameter.
#' @param symmetrized A boolean. Whether to symmetrize the output.
#' @param fixed_effects A boolean. Whether to use fixed effects.
#' @param CV_folds An integer. Number of folds to use for cross-validation.
#' @param method A string. Determines the mode of cross validation.
#'        Use "grid" to perform cross validation over a grid of lambda values or "mse" to compute the
#'        cross validation mean squared error for a single lambda value.
#' @param lambda A double. A specific lambda value on which to compute the cross validation MSE.
#'        Required if method = "mse".
#' @param CV_grid A NumericVector. Grid of lambda values to use for cross-validation.
#'        Required if method = "grid".
#'
#' @return If method = "grid", returns a list containing:
#'         \describe{
#'           \item{best_lambda}{The lambda value corresponding to the minimum cross validation error.}
#'           \item{min_error}{The minimum cross validation mean squared error.}
#'           \item{var_errors}{The variance of the cross validation errors across folds.}
#'         }
#'         If method = "mse", returns a double representing the cross validation mean squared error for the provided lambda.
#' @export
cross_validation_combined <- function(inputs, outputs, Lipschitz, iterations, etol, gamma,
                                      symmetrized, fixed_effects, CV_folds,
                                      method = c("grid", "mse"), lambda = NULL, CV_grid = NULL) {
    method <- match.arg(method)
    num_cores <- parallel::detectCores() - 1
    K <- nrow(inputs)
    all_pairs <- combn(K, 2)
    fold_indices <- all_pairs[, sample(ncol(all_pairs), CV_folds)]

    # Precompute the splits so we don't subset repeatedly later on.
    fold_splits <- lapply(1:CV_folds, function(fold) {
        test_idx <- fold_indices[, fold]
        train_idx <- setdiff(1:K, test_idx)
        list(
            train_inputs = inputs[train_idx, , drop = FALSE],
            train_outputs = outputs[train_idx, , drop = FALSE],
            test_inputs = inputs[test_idx, , drop = FALSE],
            test_outputs = outputs[test_idx, , drop = FALSE]
        )
    })

    # Helper function to compute the MSE for a given lambda value on a split
    compute_fold_mse <- function(lambda_val, split) {
        fit <- accel_nuclear_gradient_wrapper(
            split$train_inputs, split$train_outputs, lambda_val,
            Lipschitz, iterations, etol, gamma,
            symmetrized, fixed_effects
        )
        predicted_valid <- t(fit %*% t(split$test_inputs))
        mean((predicted_valid - split$test_outputs)^2)
    }

    if (method == "grid") {
        if (is.null(CV_grid)) stop("CV_grid must be provided when method='grid'")
        lambda_grid <- as.numeric(CV_grid)
        CV_errors <- matrix(NA, nrow = length(lambda_grid), ncol = CV_folds)

        for (fold in 1:CV_folds) {
            cat(sprintf("\rProcessing fold %d of %d", fold, CV_folds))
            flush.console()
            split <- fold_splits[[fold]]

            future::plan(future::multisession, workers = num_cores)
            fold_errors <- furrr::future_map(lambda_grid, function(lam) {
                compute_fold_mse(lam, split)
            })
            future::plan(future::sequential)

            CV_errors[, fold] <- unlist(fold_errors)
        }
        cat("\n")
        mean_errors <- rowMeans(CV_errors)
        best_index <- which.min(mean_errors)
        var_errors <- var(mean_errors)
        min_error <- min(mean_errors)

        return(list(
            best_lambda = lambda_grid[best_index],
            min_error = min_error,
            var_errors = var_errors
        ))
    } else if (method == "mse") {
        if (is.null(lambda)) stop("A lambda value must be provided when method='mse'")

        future::plan(future::multisession, workers = num_cores)
        cv_errors <- furrr::future_map(fold_splits, function(split) {
            compute_fold_mse(lambda, split)
        })
        future::plan(future::sequential)

        mean_error <- mean(unlist(cv_errors))
        return(mean_error)
    }
}

# Example usage:
# result <- cross_validation_combined(inputs, outputs, "regression", 100, 1e-6, 0.01, TRUE, FALSE, 5,
#                                       method = "grid", CV_grid = seq(0.01, 1, length.out = 10))
# mse_result <- cross_validation_combined(inputs, outputs, "regression", 100, 1e-6, 0.01, TRUE, FALSE, 5,
#                                         method = "mse", lambda = 0.1)
