import numpy as np
from scipy.stats import norm


def expected_improvement(mu, sigma, best_f):
    improvement = mu - best_f 
    with np.errstate(divide='warn'):
        Z = np.divide(improvement, sigma, out=np.zeros_like(improvement), where=sigma > 0)
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0  # Handle cases where sigma is zero
    return ei
