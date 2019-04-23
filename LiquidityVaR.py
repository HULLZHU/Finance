import numpy as np
from scipy.stats import norm

#  (a）
mu = 0.01
sigma = 0.03
spread = 0.0035
spread_sigma = 0.015
standard_VaR = (- mu + norm.ppf(0.99)*sigma)*16
liquidity_adjuster = (0.5 * (spread + norm.ppf(0.99)*spread_sigma))*16
LaVaR = standard_VaR + liquidity_adjuster
print('LVaR at 99% Confidence Level is ')
print(LaVaR)

#  (b)
amt = 40
mu = 0.03
spread = 0.0055
standard_VaR = 40*(- mu + norm.ppf(0.95)*sigma)
liquidity_adjuster = 40 * 0.5 * spread
LaVaR = standard_VaR + liquidity_adjuster
print('Before the change of Spread, the LVaR is ')
print(LaVaR)
spread = 0.0255
liquidity_adjuster = 40 * 0.5 * spread
LaVaR = standard_VaR + liquidity_adjuster
print('After the change of Spread, the LVaR is ')
print(LaVaR)