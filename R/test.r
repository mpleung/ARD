library(nuclearARD)
make_basis <- function(k, N, size) replace(numeric(N), (1 + (k - 1) * size):(k * size), 1)

generate_HRM <- function(N) {
    group_size <- 20
    num_groups <- N / group_size
    p <- 0.9
    q <- 0.1
    bases <- lapply(1:num_groups, function(x) make_basis(x, N, group_size)) |> (\(.) rep(., group_size))()
    bases_mat <- do.call(rbind, bases)
    bases_mat_order <- lapply(1:num_groups, function(x) x + 0:(group_size - 1) * num_groups)
    bases_mat <- bases_mat[unlist(bases_mat_order), ]
    bases_mat[bases_mat == 1] <- p
    bases_mat[bases_mat == 0] <- q
    diag(bases_mat) <- 0
    DGP_mat <- bases_mat
    HRM_mats <- Adj_Matrix_Construction(DGP_mat)
    return(HRM_mats)
}


generate_RDP <- function(N) {
    positions <- sqrt(runif(N))
    P_RDP <- positions * t(replicate(N, positions))
    RDP_mats <- Adj_Matrix_Construction(P_RDP)
    return(RDP_mats)
}


Adj_Matrix_Construction <- function(P_mat) {
    N <- nrow(P_mat)
    diag(P_mat) <- 0
    U <- matrix(runif(N * N), nrow = N, ncol = N)
    U <- t(U) / 2 + U / 2
    A_mat <- (U < P_mat) * 1.0
    return(list(P_mat = P_mat, A_mat = A_mat))
}

N <- 300
K <- 6

net <- generate_RDP(N)


types <- matrix(rbinom(K * N, 1, 0.5), nrow = K, ncol = N)

ARD_mats <- types %*% net$A_mat


matrix_reg <- matrix_regression(
    inputs = types,
    outputs = ARD_mats,
    fixed_effects = TRUE,
    CV = TRUE,
    CV_folds = 5
)

matrix_reg <- matrix_regression(
    inputs = types,
    outputs = ARD_mats,
    fixed_effects = TRUE,
    CV = TRUE,
    CV_grid = seq(0.01, 10, by = 0.1),
    CV_folds = 5
)
