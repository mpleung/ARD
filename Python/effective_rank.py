# Coded for Python 3.

# Produces Table 1, which gives the effective ranks of M^* under three models of network formation.

import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform

Ns = [50,100,200,300,400,500] # sample sizes
B = 500

eff_ranks = np.zeros((3,len(Ns)))

for i,n in enumerate(Ns):
    np.random.seed(n)

    for b in range(B):
        # latent space model
        alpha = np.random.normal(0,1,n)
        positions = np.random.exponential(size=(n,2))
        latent_index = alpha + alpha[:,None] - squareform(pdist(positions))
        P_LSM = np.exp(latent_index) / (1 + np.exp(latent_index)) # n x n matrix of link probabilities
        np.fill_diagonal(P_LSM,0) # zero diagonals

        U, s, V = np.linalg.svd(P_LSM)
        eff_ranks[0,i] += s.sum() / np.sqrt((P_LSM**2).sum())

        # random dot product graph
        positions = np.sqrt(np.random.uniform(0,1,n))
        P_RDP = positions * positions[:,None] # n x n matrix of link probabilities
        np.fill_diagonal(P_RDP,0) # zero diagonals

        U, s, V = np.linalg.svd(P_RDP)
        eff_ranks[1,i] += s.sum() / np.sqrt((P_RDP**2).sum())

        # stochastic block model
        P_SBM = np.ones((n,n))*0.3 # n x n matrix of link probabilities
        bs = int(n/5)
        for r in range(5): P_SBM[(r*bs):((r+1)*bs),(r*bs):((r+1)*bs)] = 0.7
        np.fill_diagonal(P_SBM,0) # zero diagonals

        U, s, V = np.linalg.svd(P_SBM)
        eff_ranks[2,i] += s.sum() / np.sqrt((P_SBM**2).sum())

print('\caption{Effective Ranks}')
table = pd.DataFrame(eff_ranks/B)
table.columns = Ns
table.index.rename('$n$', inplace=True)
table.index = ['LSM', 'RDP', 'SBM']
print(table.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False))

