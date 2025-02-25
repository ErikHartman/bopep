import numpy as np
from scipy.stats import norm

class AcquisitionFunction:
    def __init__(self):
        self.best_so_far = 0.0
        pass

    def compute_acquisition(self, means, stds, acquisition_function):
        if acquisition_function == "expected_improvement":
            return self.expected_improvement(means, stds)
        elif acquisition_function == "upper_confidence_bound":
            return self.upper_confidence_bound(means, stds)
        elif acquisition_function == "probability_of_improvement": 
            return self.probability_of_improvement(means, stds)
        elif acquisition_function == "standard_deviation":
            return stds
        elif acquisition_function == "mean":
            return means
        else:
            raise ValueError(f"Acquisition function {acquisition_function} not recognized.")

    def expected_improvement(self, means, stds):
        """
        Compute the Expected Improvement (EI) acquisition function.

        Parameters
        means : array-like
            Predicted mean values from the surrogate model.
        stds : array-like
            Predicted standard deviation/uncertainty values from the surrogate model.
        best_so_far : float
            The current best observed function value.

        Returns
        array-like
            Expected improvement values at the input locations.
        """
        improvement = means - self.best_so_far
        with np.errstate(divide="warn"):
            Z = np.divide(
                improvement, stds, out=np.zeros_like(improvement), where=stds > 0
            )
        ei = improvement * norm.cdf(Z) + stds * norm.pdf(Z)
        ei[stds == 0.0] = 0.0
        self.best_so_far = max(self.best_so_far, np.max(means))
        return ei

    def upper_confidence_bound(self, means, stds, kappa=1.96):
        """
        Compute the Upper Confidence Bound (UCB) acquisition function.

        Parameters
        means : array-like
            Predicted mean values from the surrogate model.
        stds : array-like
            Predicted standard deviation/uncertainty values from the surrogate model.
        kappa : float
            The exploration-exploitation trade-off parameter.

        Returns
        array-like
            UCB values at the input locations.
        """
        return means + kappa * stds
    
    def probability_of_improvement(self, means, stds):
        """
        Compute the Probability of Improvement (PI) acquisition function.

        Parameters
        means : array-like
            Predicted mean values from the surrogate model.
        stds : array-like
            Predicted standard deviation/uncertainty values from the surrogate model.
        best_so_far : float
            The current best observed function value.

        Returns
        array-like
            PI values at the input locations.
        """
        improvement = means - self.best_so_far
        with np.errstate(divide="warn"):
            Z = np.divide(
                improvement, stds, out=np.zeros_like(improvement), where=stds > 0
            )
        pi = norm.cdf(Z)
        pi[stds == 0.0] = 0.0
        return pi
    
