import numpy as np
from scipy.stats import norm
from numpy import linalg as LA  # for computing eignvalues

# Define parameters
mu = np.array([0.5, 0.2, 0.3])
weights = np.array([0.5, 0.2, 0.3])
rho = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
sigma_mat = np.array([[0.3, 0, 0], [0, 0.2, 0], [0, 0, 0.15]])
cov = sigma_mat.dot(rho).dot(sigma_mat)
sigma = np.sqrt(weights.dot(cov).dot(weights.T))
alpha = 0.95
T = np.size(weights)
marginal_VaR = np.array([0.0, 0.0, 0.0])
marginal_ES = np.array([0.0, 0.0, 0.0])

#  Compute Marginal VaR and Marginal ES
for i in range(T):
    marginal_VaR[i] = -mu[i] + norm.ppf(alpha)*(np.dot(cov, np.transpose(weights)))[i]/sigma
    marginal_ES[i] = -mu[i] + norm.pdf(norm.ppf(alpha))*(np.dot(cov, np.transpose(weights)))[i]/((1-alpha)*sigma)
print('Marginal VaR')
print(marginal_VaR)
print('Marginal ES')
print(marginal_ES)

#  Compute eigenvalues of matrix rho
eigenvalues = LA.eigvals(rho)
print('Eigenvalues')
print(eigenvalues)





