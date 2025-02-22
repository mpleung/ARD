library(nuclearARD)

# Create some test data
N <- 100 # village size
K <- 6 # number of traits
M <- N # number of households

# Generate random inputs (traits)
types <- matrix(rbinom(K * N, 1, 0.5), nrow = K, ncol = N)

# Generate random outputs (ARD data)
ARD_mats <- matrix(rbinom(K * M, 1, 0.5), nrow = K, ncol = M)

# Try the matrix regression with cross validation
matrix_reg <- matrix_regression(
    inputs = types,
    outputs = ARD_mats,
    fixed_effects = TRUE,
    CV = TRUE,
    CV_folds = 5
)
