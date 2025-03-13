import numpy as np
from scipy.stats import norm

class AcquisitionFunction:
    def __init__(self):
        self.best_so_far_ei = 0.0
        self.best_so_far_pi = 0.0

    def compute_acquisition(self, predictions, acquisition_function):
        """
        Args:
            peptides (dict): {peptide: embedding} -- used for consistent ordering if needed
            predictions (dict): {peptide: (mean, std)}
            acquisition_function (str): which method to compute
        Returns:
            dict {peptide: acquisition_value}
        """
        peptides = list(predictions.keys())
        means = np.array([predictions[p][0] for p in peptides])
        stds = np.array([predictions[p][1] for p in peptides])
        if acquisition_function == "expected_improvement":
            return self.expected_improvement(peptides, means, stds)
        elif acquisition_function == "upper_confidence_bound":
            return self.upper_confidence_bound(peptides, means, stds)
        elif acquisition_function == "probability_of_improvement":
            return self.probability_of_improvement(peptides, means, stds)
        elif acquisition_function == "standard_deviation":
            return self.standard_deviation(peptides, stds)
        elif acquisition_function == "mean":
            return self.mean(peptides, means)
        else:
            raise ValueError(f"Acquisition function '{acquisition_function}' not recognized.")

    def expected_improvement(self, peptides, means, stds):
        """
        Compute the Expected Improvement (EI) for each peptide.
        """
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        improvement = means - self.best_so_far_ei
        with np.errstate(divide="ignore"):
            Z = np.divide(improvement, stds, out=np.zeros_like(improvement), where=(stds > 1e-12))

        ei = improvement * norm.cdf(Z) + stds * norm.pdf(Z)
        ei[stds == 0.0] = 0.0

        current_best_pred = np.max(means)
        if current_best_pred > self.best_so_far_ei:
            self.best_so_far_ei = current_best_pred

        return {
            peptide: float(value)
            for peptide, value in zip(peptides, ei)
        }

    def upper_confidence_bound(self, peptides, means, stds, kappa=1.96):
        """
        Compute the UCB for each peptide.

        UCB(c) = mean + kappa * std
        """
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        ucb = means + kappa * stds

        return {
            peptide: float(value)
            for peptide, value in zip(peptides, ucb)
        }

    def probability_of_improvement(self, peptides, means, stds):
        """
        Compute the Probability of Improvement (PI) for each peptide.

        PI(c) = Phi((mean - best_so_far) / std)
        """
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        improvement = means - self.best_so_far_pi
        with np.errstate(divide="ignore"):
            Z = np.divide(improvement, stds, out=np.zeros_like(improvement), where=(stds > 1e-12))
        pi = norm.cdf(Z)
        pi[stds == 0.0] = 0.0

        current_best_pred = np.max(means)
        if current_best_pred > self.best_so_far_pi:
            self.best_so_far_pi = current_best_pred

        return {
            peptide: float(value)
            for peptide, value in zip(peptides, pi)
        }

    def standard_deviation(self, peptides, stds):
        """
        Standard deviation as an acquisition function. Pure exploration.
        """
        return {
            peptide: float(std)
            for peptide, std in zip(peptides, stds)
        }

    def mean(self, peptides, means):
        """
        Mean as an acquisition function (pure exploitation).
        """
        return {
            peptide: float(mean)
            for peptide, mean in zip(peptides, means)
        }
