import numpy as np
from cvxpy import *
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy import linalg as LA  # for computing eignvalues

#  First, We set n as number of stocks, mu as the vector of expected returns
#  rho as correlation coefficient matrix, cov as the covariance matrix
n = 3  # Number of stocks
mu = np.array([[0.3], [0.2], [0.1]])
sigmas = np.array([[0.2, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.1]])
rho = np.array([[1, -0.3, -0.5], [-0.3, 1.0, -0.6], [-0.5, -0.6, 1]])
cov = np.dot(np.dot(sigmas, rho), sigmas)

#  Set Variables for the Optimization Problem
w = Variable(n)  # weights
ret = mu.T*w  # returns
risk = quad_form(w, cov)  # Here Risk means variance

#  Compute 200 Opportunity Sets
SAMPLES = 400  # Number of Possible Portfolios
returns = np.zeros(SAMPLES)  # array for returns
sigmas = np.zeros(SAMPLES)  # array for sigmas
weights = np.zeros([SAMPLES, n])
for i in range(SAMPLES):
    ss = np.abs(np.random.randn(3))
    total = np.sum(ss)
    for j in range(n):
        weights[i][j] = ss[j]/total
    returns[i] = np.dot(weights[i].T, mu)
    sigmas[i] = np.sqrt(np.dot(np.dot(weights[i], cov), weights[i].transpose()))

#  Compute the curve
SAMPLES = 30  # Number of Opportunity Sets on the curve
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
for i in range(SAMPLES):
    delta = (0.3 - 0.1)/SAMPLES
    prob = Problem(Minimize(risk), [sum(w) == 1, ret == 0.1 + i * delta, w >= 0])
    prob.solve()
    risk_data[i] = np.sqrt(risk.value)
    ret_data[i] = ret.value

#  Compute the minimum variance portfolio
prob = Problem(Minimize(risk), [sum(w) == 1, w >= 0])
prob.solve()
MVP_sigma = np.sqrt(risk.value)
MVP_ret = ret.value
print('The Sigma of Minimum Variance Portfolio is \n', MVP_sigma)
print('The return of Minimum Variance Portfolio is \n', MVP_ret[0])

#  Set the label names
plt.xlabel('Standard deviation')
plt.ylabel('Return')

#  Plot the curve
curve = plt.plot(risk_data, ret_data, 'g-', label='Curve')
MVP = plt.plot(MVP_sigma, MVP_ret, '*-', label='Minimum Variance Portfolio')
Opportunity_Set = plt.scatter(sigmas, returns, label='Opportunity Set')

#  Plot the Individual Stocks' Points
for i in range(n):
    plt.plot(sqrt(cov[i, i]).value, mu[i], 'o--', label='stock '+str(i))

plt.legend(loc='upper right')
plt.show()


#  Compute the 95 % VaR and 95 % ES for each of the combination of weights
VaRs = -returns + norm.ppf(0.95)*sigmas
ESs = -returns + sigmas*norm.pdf(norm.ppf(0.95))/(1-0.95)
smallest_VaR = np.argmin(VaRs)
smallest_ES = np.argmin(ESs)
print('The Allocation for the smallest VaR is \n', weights[smallest_VaR])
print('The Allocation for the smallest ES is \n', weights[smallest_ES])

#  Determine whether the correlation coefficient matrix is positive definite or not
eigenvalues = LA.eigvals(rho)
print('Eigenvalue\n')
print(eigenvalues)
T = np.size(eigenvalues)
for i in range(T):
    if eigenvalues[i] <0:
        print('It is not a Positive Definite Matrix\n')
        break
    elif i == np.size(eigenvalues) and eigenvalues[i] > 0:
        print('It is a Positive Definite Matrix\n')







