install.packages('nuclearARD_0.1.1.tar.gz', repos=NULL, type='source') #Only needs to be run very first time
library(nuclearARD)
set.seed(0) # set seed

N1 <- 100
N2 <- 200
K <- as.integer(round(N1^0.4))

# simulate network
positions <- sqrt(runif(N2)) 
M <- positions %*% t(positions) # n x n matrix of link probabilities
diag(M) <- 0 # zero out diagonal entries to ensure no self links
U <- matrix(runif(N2^2), nrow=N2, ncol=N2)
U <- t(U)/2 + U/2 # make matrix symmetric to have an undirected network
G <- (U < M)[,1:N1] # simulated network submatrix


types <- matrix(rbinom(K*N2, size=1, prob=0.5), nrow=K, ncol=N2)
ARDs <- types %*% G


write.csv(t(ARDs), 'ARD_data.csv')
write.csv(t(types), 'type_data.csv')


# load CSVs as numpy matrices

ARDs <- t(as.matrix(read.csv('ARD_data.csv')))
types <- t(as.matrix(read.csv('type_data.csv')))


# store dimensions
K <- nrow(ARDs)
N1 <- ncol(ARDs)
N2 <- ncol(types)


lmbd <- 2 * (sqrt(N1) + sqrt(N2) + 1) * (sqrt(N2) + sqrt(K))
M_hat <- accel_nuclear_gradient(types, ARDs, lmbd)
#or just run
M_hat <- matrix_regression(types, ARDs)


print(M_hat[1:5, 1:5])


write.csv(as.matrix(M_hat), 'estimated_network.csv')


U <- matrix(runif(N2*N1), nrow=N2, ncol=N1) # draw uniform random variables
diag(U) <- 0 # zero out the diagonal entries (if no self links)


U_sub <- U[1:N1, ] # extract upper N1 x N1 submatrix
U_sub <- t(U_sub)/2 + U_sub # symmetrize the submatrix
U[1:N1, ] <- U_sub # replace the upper N1 x N1 submatrix of
                    #     the original matrix U with U_sub



G <- 1*(U < M_hat)


print(G[1:10, 1:10])
