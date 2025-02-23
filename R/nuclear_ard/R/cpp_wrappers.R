#' Accelerated gradient function for ARD (C++ wrapper)
#'
#' \code{accel_nuclear_gradient} implements the Accelerated Gradient Algorithm (Algorithm 2)
#' from Ji & Ye (2009). This should be one of only two functions users interact with directly
#' from the \emph{ardlasso} package. (The other is \code{\link{matrix_regression}}, which is a wrapper
#' function for accel_nuclear_gradient, forcing the option Lipschitz == 'regression'.)
#' When symmetrize is set to \code{TRUE}, this function instead implements Algorithm 1 in
#' Alidaee, Auerbach, and Leung (2020). Note that Alidaee et al. (2019) have a slight change
#' in notation, wherein the tuple (N, M) below is equivalent to their (N_2, N_1).
#'
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#'  It should be of dimension K x N, where N = village size and K = number of ARD characteristics.
#' @param outputs A matrix object. This contains the ARD survey data
#'  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre.
#' @param lambda A scalar (numeric) value. This is an initial guess that will be iterated on. It can
#'  alternatively be defined as 'NW', which will set \eqn{\lambda = 2(\sqrt{N} + \sqrt{M})(\sqrt{N} + \sqrt{K})}.
#' @param Lipschitz A string. This determines how the Lipschitz constant is computed. 'JiYe' implements
#'  the iterative algorithm outlined in Ji and Ye (2009). 'regression' sets it to the analytically derived
#'  constant for the multivariate regression problem posed in Alidaee et al. (2020). Set to 'regression' by
#'  default.
#' @param iterations A scalar (integer) value. It is the number of iterations the user
#'  specifies should occur. Set by default to 5000.
#' @param etol A scaler. It is the error tolerance for the algorithm. The algorithm will terminate
#'  when either the maximum number of iterations has been met or the mean absolute error between iterations
#'  is below etol.
#' @param symmetrize A boolean value. This captures whether to implement the aforementioned modified
#'  gradient descent algorithm..
#' @param fixed_effects A boolean value. This captures whether to implement node-level fixed effects.
#' @return An N x M matrix estimate of network connections.
#' @export
#' @import Matrix
accel_nuclear_gradient_wrapper <- function(inputs, outputs, lambda, Lipschitz = "regression", iterations = 5000, etol = 10e-05, gamma = 2.0, symmetrize = TRUE, fixed_effects = FALSE) {
    # This function implements Algorithm 2 from "An Accelerated Gradient
    # Method for Trace Norm Minimization" by Ji & Ye (2009)

    # Make sure inputs are valid.
    if (is.logical(fixed_effects) == FALSE) {
        stop("fixed_effects must be a boolean value. See documentation for details.")
    }
    if (is.logical(symmetrize) == FALSE) {
        stop("symmetrize must be a boolean value. See documentation for details.")
    }
    if (is.numeric(iterations) == FALSE) {
        stop("iterations must be a numeric value. See documentation for details.")
    }
    if (is.numeric(lambda) == FALSE && lambda != "NW") {
        stop("lambda must be either 'NW' or a numeric value. See documentation for details.")
    }
    if (is.numeric(etol) == FALSE) {
        stop("etol must be a numeric value. See documentation for details.")
    }
    if (is.numeric(gamma) == FALSE) {
        stop("gamma must be a numeric value. See documentation for details.")
    }
    if (Lipschitz == "regression") {
        # For the multivariate regression case, the Lipschitz constant can analytically be derived.
        # It is the square of the largest (first) singular value.
        L <- eigen(inputs %*% t(inputs))$values[1]
    } else if (Lipschitz == "JiYe") {
        # For objective functions where the Lipschitz constant cannot be derived analytically, we
        # implement the algorithm introduced by Ji and Ye (2009). Here, we initiate L to 1. Later,
        # within the for loop, each iteration will converge to the optimal L using Steps 1 and 2
        #  of their Algorithm 2.
        L <- 1
    } else {
        stop(Lipschitz, "is an invalid option for the parameter Lipschitz. Please select one of either 'regression' or 'JiYe'. See documentation for details.")
    }
    if (lambda == "NW") {
        lambda <- 2 * (sqrt(M) + sqrt(N) + 1) * (sqrt(N) + sqrt(K))
    }
    estimate <- accel_nuclear_gradient_cpp(inputs, outputs, lambda, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects)
    return(estimate)
}

#' Cross validation function for ARD (C++ wrapper)
#'
#' \code{cross_validation} implements the cross validation algorithm outlined in Alidaee et al. (2020).
#'
#' @param inputs A matrix object. This contains ARD census data in matrix form.
#' @param outputs A matrix object. This contains the ARD survey data
#'  in matrix form with size K x M, where N > M = number of households receiving ARD questionairre.
#' @param Lipschitz String. Method for computing Lipschitz constant.
#' @param iterations Integer. Maximum number of iterations.
#' @param etol Double. Error tolerance.
#' @param gamma Double. Step size parameter.
#' @param symmetrized Boolean. Whether to symmetrize the output.
#' @param fixed_effects Boolean. Whether to use fixed effects.
#' @param CV_grid NumericVector. Grid of lambda values to use for cross-validation. If not provided, a dynamic grid search is conducted.
#' @param CV_folds Integer. Number of folds to use for cross-validation. Defaults to 5.
#' @return A double. The optimal lambda value to use for network estimation.
#' @export
#' @import Matrix
# cross_validation_wrapper <- function(inputs, outputs, Lipschitz = "regression", iterations = 5000, etol = 10e-05, gamma = 2.0, symmetrize = TRUE, fixed_effects = FALSE, CV_grid = NULL, CV_folds = 5) {
#     if (CV_folds >= dim(inputs)[1]) {
#         stop("CV_folds must be less than the number of ARD traits.")
#     }
#     # If CV_grid is not provided, conduct a dynamic grid search.
#     if (is.null(CV_grid)) {
#         dynamic <- TRUE
#         grid_size <- 20
#         CV_grid <- seq(0, 200, grid_size)
#         min_value <- 0.00001
#         CV_grid[1] <- min_value
#         print(paste0("First estimate grid: ", CV_grid[1], " to ", CV_grid[length(CV_grid)]))
#         estimate <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
#         print(paste0("Estimate: ", estimate$best_lambda, " Var Error: ", estimate$var_errors, " Min Error: ", estimate$min_error))
#         # Rerun with new CV_grid based around the estimate.
#         margin <- grid_size
#         grid_size <- 5
#         min_estimate <- max(min_value, floor(estimate$best_lambda) - margin)
#         max_estimate <- ceiling(estimate$best_lambda) + margin
#         CV_grid <- seq(min_estimate, max_estimate, grid_size)
#         print(paste0("Second estimate grid: ", CV_grid[1], " to ", CV_grid[length(CV_grid)]))
#         estimate <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
#         print(paste0("Estimate: ", estimate$best_lambda, " Var Error: ", estimate$var_errors, " Min Error: ", estimate$min_error))
#         # Rerun with new CV_grid based around the estimate.
#         margin <- grid_size
#         grid_size <- 1
#         min_estimate <- max(min_value, floor(estimate$best_lambda) - margin)
#         max_estimate <- ceiling(estimate$best_lambda) + margin
#         CV_grid <- seq(min_estimate, max_estimate, grid_size)
#         print(paste0("Third estimate grid: ", CV_grid[1], " to ", CV_grid[length(CV_grid)]))
#         estimate <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
#         print(paste0("Estimate: ", estimate$best_lambda, " Var Error: ", estimate$var_errors, " Min Error: ", estimate$min_error))
#         # Rerun with new CV_grid based around the estimate.
#         margin <- grid_size
#         grid_size <- 0.5
#         min_estimate <- max(min_value, round(estimate$best_lambda - margin, digits = 1))
#         max_estimate <- round(estimate$best_lambda + margin, digits = 1)
#         CV_grid <- seq(min_estimate, max_estimate, grid_size)
#         print(paste0("Fourth estimate grid: ", CV_grid[1], " to ", CV_grid[length(CV_grid)]))
#         estimate <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
#         print(paste0("Estimate: ", estimate$best_lambda, " Var Error: ", estimate$var_errors, " Min Error: ", estimate$min_error))
#         # Rerun with new CV_grid based around the estimate.
#         margin <- grid_size
#         grid_size <- 0.1
#         min_estimate <- max(min_value, round(estimate$best_lambda - margin, digits = 1))
#         max_estimate <- round(estimate$best_lambda + margin, digits = 1)
#         CV_grid <- seq(min_estimate, max_estimate, grid_size)
#         print(paste0("Fifth estimate grid: ", CV_grid[1], " to ", CV_grid[length(CV_grid)]))
#         estimate <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
#         print(paste0("Estimate: ", estimate$best_lambda, " Var Error: ", estimate$var_errors, " Min Error: ", estimate$min_error))
#         # Rerun with new CV_grid based around the estimate.
#         margin <- grid_size
#         grid_size <- 0.05
#         min_estimate <- max(min_value, round(estimate$best_lambda - margin, digits = 2))
#         max_estimate <- round(estimate$best_lambda + margin, digits = 2)
#         CV_grid <- seq(min_estimate, max_estimate, grid_size)
#         print(paste0("Sixth estimate grid: ", CV_grid[1], " to ", CV_grid[length(CV_grid)]))
#         estimate <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
#         print(paste0("Estimate: ", estimate$best_lambda, " Var Error: ", estimate$var_errors, " Min Error: ", estimate$min_error))
#         # Rerun with new CV_grid based around the estimate.
#         margin <- grid_size
#         grid_size <- 0.01
#         min_estimate <- max(min_value, round(estimate$best_lambda - margin, digits = 2))
#         max_estimate <- round(estimate$best_lambda + margin, digits = 2)
#         CV_grid <- seq(min_estimate, max_estimate, grid_size)
#         print(paste0("Final estimate grid: ", CV_grid[1], " to ", CV_grid[length(CV_grid)]))
#         estimate <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
#         print(paste0("Estimate: ", estimate$best_lambda, " Var Error: ", estimate$var_errors, " Min Error: ", estimate$min_error))
#         return(estimate$best_lambda)

#         CV_grid <- seq(0.01, 0.31, 0.01)
#         print(paste0("Benchmark estimate grid: ", CV_grid[1], " to ", CV_grid[length(CV_grid)]))
#         estimate <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
#         print(paste0("Estimate: ", estimate$best_lambda, " Var Error: ", estimate$var_errors, " Min Error: ", estimate$min_error))
#     } else {
#         dynamic <- FALSE
#         estimate <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
#     }
#     return(estimate$best_lambda)
# }
cross_validation_wrapper <- function(inputs, outputs, Lipschitz = "regression", iterations = 5000, etol = 10e-05, gamma = 2.0, symmetrize = TRUE, fixed_effects = FALSE, CV_grid = NULL, CV_folds = 5) {
    N <- dim(inputs)[2]
    K <- dim(inputs)[1]
    M <- dim(outputs)[2]
    if (CV_folds >= K) {
        stop("CV_folds must be less than the number of ARD traits.")
    }
    if (is.null(CV_grid)) {
        # objective <- function(lambda) {
        #     return(cross_validation_mse(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, lambda, CV_folds))
        # }

        NW_recommended_lambda <- 2 * (sqrt(N) + sqrt(M) + 1) * (sqrt(N) + sqrt(K))

        objective <- function(lambda) {
            mse <- cross_validation_mse(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, lambda, CV_folds)
            return(list(Score = -mse, Pred = NA))
        }


        # opt_result <- optimize(objective,
        #     lower = 0.01,
        #     upper = 1000,
        #     maximum = FALSE,
        #     tol = 0.1
        # )
        # # Extract the optimal parameter and its associated MSE
        # best_param <- opt_result$minimum
        # best_mse <- opt_result$objective

        OPT_Res <- BayesianOptimization(objective,
            bounds = list(lambda = c(0.01, NW_recommended_lambda)),
            init_points = 10, n_iter = 10,
            acq = "ucb", kappa = 10, eps = 10,
            verbose = TRUE
        )
        best_param <- OPT_Res$Best_Par
        best_mse <- -OPT_Res$Best_Value


        cat("Optimal parameter:", best_param, "\n")
        cat("Associated average MSE:", best_mse, "\n")
    } else {
        output <- cross_validation(inputs, outputs, Lipschitz, iterations, etol, gamma, symmetrize, fixed_effects, CV_grid, CV_folds)
        print(paste0("Optimal lambda: ", output$best_lambda, " Min Error: ", output$min_error))
        best_param <- output$best_lambda
    }
    return(best_param)
}
