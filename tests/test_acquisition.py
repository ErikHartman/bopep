import pytest
import numpy as np
from bopep.bayes.acquisition import AcquisitionFunction


class TestAcquisitionFunction:
    """Test suite for the ParEGO-based acquisition functions."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.acq = AcquisitionFunction(rng_seed=42)
        
        # Single-objective test data
        self.sequences_single = ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"]
        self.means_single = np.array([1.0, 2.0, 3.0])
        self.stds_single = np.array([0.1, 0.2, 0.3])
        
        self.predictions_single = {
            "PEPTIDE1": (1.0, 0.1),
            "PEPTIDE2": (2.0, 0.2), 
            "PEPTIDE3": (3.0, 0.3)
        }
        
        # Multi-objective test data
        self.predictions_multi = {
            "PEPTIDE1": {"obj1": (1.0, 0.1), "obj2": (2.0, 0.2)},
            "PEPTIDE2": {"obj1": (1.5, 0.1), "obj2": (1.5, 0.2)},
            "PEPTIDE3": {"obj1": (2.0, 0.1), "obj2": (1.0, 0.2)}
        }
        
        self.objective_directions = {"obj1": "maximize", "obj2": "minimize"}


class TestSingleObjectiveFunctions(TestAcquisitionFunction):
    """Tests for single-objective acquisition functions."""
    
    def test_expected_improvement_maximize(self):
        """Test expected improvement with maximize=True."""
        result = self.acq.expected_improvement(
            self.sequences_single, self.means_single, self.stds_single, maximize=True
        )
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert len(result) == len(self.sequences_single)
        assert all(p in result for p in self.sequences_single)
        assert all(isinstance(v, float) for v in result.values())
        
        # EI values should be non-negative
        assert all(v >= 0.0 for v in result.values())
        
        # For maximization, higher means should generally have higher EI
        # (though this can vary due to uncertainty)
        assert result["PEPTIDE3"] >= 0.0  # Highest mean should have some EI
    
    def test_expected_improvement_minimize(self):
        """Test expected improvement with maximize=False."""
        result = self.acq.expected_improvement(
            self.sequences_single, self.means_single, self.stds_single, maximize=False
        )
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert len(result) == len(self.sequences_single)
        assert all(isinstance(v, float) for v in result.values())
        
        # EI values should be non-negative
        assert all(v >= 0.0 for v in result.values())
        
        # For minimization, lower means should generally have higher EI
        assert result["PEPTIDE1"] >= 0.0  # Lowest mean should have some EI
    
    def test_upper_confidence_bound_maximize(self):
        """Test upper confidence bound with maximize=True."""
        kappa = 1.96
        result = self.acq.upper_confidence_bound(
            self.sequences_single, self.means_single, self.stds_single, 
            kappa=kappa, maximize=True
        )
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert len(result) == len(self.sequences_single)
        
        # For maximize=True, UCB should be mean + kappa * std
        for i, sequence in enumerate(self.sequences_single):
            expected = self.means_single[i] + kappa * self.stds_single[i]
            assert abs(result[sequence] - expected) < 1e-10
    
    def test_upper_confidence_bound_minimize(self):
        """Test upper confidence bound with maximize=False."""
        kappa = 1.96
        result = self.acq.upper_confidence_bound(
            self.sequences_single, self.means_single, self.stds_single,
            kappa=kappa, maximize=False
        )
        
        # For maximize=False, UCB should be mean - kappa * std (lower confidence bound)
        for i, sequence in enumerate(self.sequences_single):
            expected = self.means_single[i] - kappa * self.stds_single[i]
            assert abs(result[sequence] - expected) < 1e-10
    
    def test_probability_of_improvement_maximize(self):
        """Test probability of improvement with maximize=True."""
        result = self.acq.probability_of_improvement(
            self.sequences_single, self.means_single, self.stds_single, maximize=True
        )
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert len(result) == len(self.sequences_single)
        
        # PI values should be between 0 and 1
        assert all(0.0 <= v <= 1.0 for v in result.values())
    
    def test_probability_of_improvement_minimize(self):
        """Test probability of improvement with maximize=False.""" 
        result = self.acq.probability_of_improvement(
            self.sequences_single, self.means_single, self.stds_single, maximize=False
        )
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert len(result) == len(self.sequences_single)
        
        # PI values should be between 0 and 1
        assert all(0.0 <= v <= 1.0 for v in result.values())
    
    def test_standard_deviation(self):
        """Test standard deviation function."""
        result = self.acq.standard_deviation(self.sequences_single, self.stds_single)
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert len(result) == len(self.sequences_single)
        
        # Values should match input standard deviations
        for i, sequence in enumerate(self.sequences_single):
            assert abs(result[sequence] - self.stds_single[i]) < 1e-10
    
    def test_mean(self):
        """Test mean function."""
        result = self.acq.mean(self.sequences_single, self.means_single)
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert len(result) == len(self.sequences_single)
        
        # Values should match input means
        for i, sequence in enumerate(self.sequences_single):
            assert abs(result[sequence] - self.means_single[i]) < 1e-10


class TestMultiObjectiveFunctions(TestAcquisitionFunction):
    """Tests for multi-objective ParEGO acquisition functions."""
    
    def test_parego_chebyshev_ei(self):
        """Test ParEGO Chebyshev expected improvement."""
        result = self.acq.parego_chebyshev_ei(
            self.predictions_multi, 
            objective_directions=self.objective_directions
        )
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert len(result) == len(self.predictions_multi)
        assert all(p in result for p in self.predictions_multi.keys())
        assert all(isinstance(v, float) for v in result.values())
        
        # EI values should be non-negative
        assert all(v >= 0.0 for v in result.values())
    
    def test_parego_chebyshev_ucb(self):
        """Test ParEGO Chebyshev upper confidence bound."""
        result = self.acq.parego_chebyshev_ucb(
            self.predictions_multi,
            objective_directions=self.objective_directions
        )
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert len(result) == len(self.predictions_multi)
        assert all(isinstance(v, float) for v in result.values())
    
    def test_parego_reference_point_tracking(self):
        """Test that ParEGO properly tracks ideal and nadir reference points."""
        # Run ParEGO EI to initialize reference points
        self.acq.parego_chebyshev_ei(
            self.predictions_multi,
            objective_directions=self.objective_directions
        )
        
        # Check that reference points were set
        assert self.acq._ideal is not None
        assert self.acq._nadir is not None
        assert len(self.acq._ideal) == 2  # Two objectives
        assert len(self.acq._nadir) == 2
    
    def test_parego_with_different_objective_orders(self):
        """Test ParEGO with different objective ordering."""
        # Test with explicit objective order
        result1 = self.acq.parego_chebyshev_ei(
            self.predictions_multi,
            objective_directions=self.objective_directions,
            objective_order=["obj1", "obj2"]
        )
        
        result2 = self.acq.parego_chebyshev_ei(
            self.predictions_multi,
            objective_directions=self.objective_directions,
            objective_order=["obj2", "obj1"]
        )
        
        # Results should be valid dictionaries (ordering may affect values)
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert len(result1) == len(result2) == len(self.predictions_multi)


class TestComputeAcquisitionAPI(TestAcquisitionFunction):
    """Tests for the main compute_acquisition API."""
    
    def test_single_objective_api(self):
        """Test compute_acquisition API for single-objective functions."""
        # Test expected improvement
        result_ei = self.acq.compute_acquisition(
            self.predictions_single, "expected_improvement", maximize=True
        )
        assert isinstance(result_ei, dict)
        assert len(result_ei) == len(self.predictions_single)
        
        # Test UCB
        result_ucb = self.acq.compute_acquisition(
            self.predictions_single, "upper_confidence_bound", 
            maximize=True, kappa=1.96
        )
        assert isinstance(result_ucb, dict)
        assert len(result_ucb) == len(self.predictions_single)
        
        # Test probability of improvement
        result_pi = self.acq.compute_acquisition(
            self.predictions_single, "probability_of_improvement", maximize=True
        )
        assert isinstance(result_pi, dict)
        assert len(result_pi) == len(self.predictions_single)
        
        # Test standard deviation
        result_std = self.acq.compute_acquisition(
            self.predictions_single, "standard_deviation"
        )
        assert isinstance(result_std, dict)
        
        # Test mean
        result_mean = self.acq.compute_acquisition(
            self.predictions_single, "mean"
        )
        assert isinstance(result_mean, dict)
    
    def test_multi_objective_api(self):
        """Test compute_acquisition API for multi-objective functions.""" 
        # Test ParEGO EI
        result_ei = self.acq.compute_acquisition(
            self.predictions_multi, "parego_chebyshev_ei",
            objective_directions=self.objective_directions
        )
        assert isinstance(result_ei, dict)
        assert len(result_ei) == len(self.predictions_multi)
        
        # Test ParEGO UCB
        result_ucb = self.acq.compute_acquisition(
            self.predictions_multi, "parego_chebyshev_ucb",
            objective_directions=self.objective_directions
        )
        assert isinstance(result_ucb, dict)
        assert len(result_ucb) == len(self.predictions_multi)
    
    def test_maximize_minimize_api(self):
        """Test that maximize parameter works correctly through the API."""
        # Test maximize=True
        result_max = self.acq.compute_acquisition(
            self.predictions_single, "expected_improvement", maximize=True
        )
        
        # Test maximize=False
        result_min = self.acq.compute_acquisition(
            self.predictions_single, "expected_improvement", maximize=False
        )
        
        # Both should be valid results
        assert isinstance(result_max, dict)
        assert isinstance(result_min, dict)
        assert len(result_max) == len(result_min) == len(self.predictions_single)
        
        # Results should generally be different
        assert result_max != result_min


class TestEdgeCasesAndErrors(TestAcquisitionFunction):
    """Tests for edge cases and error conditions."""
    
    def test_invalid_acquisition_function(self):
        """Test error handling for invalid acquisition function names."""
        with pytest.raises(ValueError, match="not recognized"):
            self.acq.compute_acquisition(
                self.predictions_single, "invalid_function"
            )
    
    def test_zero_standard_deviation(self):
        """Test handling of zero standard deviations."""
        zero_stds = np.array([0.0, 0.0, 0.0])
        
        # Should not crash and should return valid results
        result_ei = self.acq.expected_improvement(
            self.sequences_single, self.means_single, zero_stds
        )
        assert isinstance(result_ei, dict)
        assert all(v >= 0.0 for v in result_ei.values())
        
        result_pi = self.acq.probability_of_improvement(
            self.sequences_single, self.means_single, zero_stds
        )
        assert isinstance(result_pi, dict)
        assert all(0.0 <= v <= 1.0 for v in result_pi.values())
    
    def test_small_standard_deviation(self):
        """Test handling of very small standard deviations."""
        small_stds = np.array([1e-15, 1e-15, 1e-15])
        
        # Should not crash and should return valid results
        result_ei = self.acq.expected_improvement(
            self.sequences_single, self.means_single, small_stds
        )
        assert isinstance(result_ei, dict)
        assert all(v >= 0.0 for v in result_ei.values())

    
    def test_single_sequence(self):
        """Test with single sequence input."""
        single_predictions = {"PEPTIDE1": (1.0, 0.1)}
        
        result = self.acq.compute_acquisition(
            single_predictions, "expected_improvement"
        )
        assert isinstance(result, dict)
        assert len(result) == 1
        assert "PEPTIDE1" in result
    
    def test_negative_kappa_ucb(self):
        """Test UCB with negative kappa value."""
        result = self.acq.upper_confidence_bound(
            self.sequences_single, self.means_single, self.stds_single,
            kappa=-1.0, maximize=True
        )
        
        # Should still work and give mean - |kappa| * std for maximize=True
        assert isinstance(result, dict)
        for i, sequence in enumerate(self.sequences_single):
            expected = self.means_single[i] - 1.0 * self.stds_single[i]
            assert abs(result[sequence] - expected) < 1e-10
    
    def test_multi_objective_missing_directions(self):
        """Test multi-objective functions without objective_directions."""
        # This should either work with defaults or raise a clear error
        try:
            result = self.acq.parego_chebyshev_ei(self.predictions_multi)
            # If it works, result should be valid
            assert isinstance(result, dict)
        except (ValueError, KeyError, TypeError):
            # If it raises an error, that's also acceptable behavior
            pass
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        extreme_means = np.array([1e10, -1e10, 0.0])
        extreme_stds = np.array([1e-10, 1e10, 1e-5])
        
        # Should not crash with extreme values
        result_ei = self.acq.expected_improvement(
            self.sequences_single, extreme_means, extreme_stds
        )
        assert isinstance(result_ei, dict)
        assert all(not np.isnan(v) for v in result_ei.values())
        assert all(not np.isinf(v) for v in result_ei.values())
    
    def test_best_so_far_tracking(self):
        """Test that best_so_far values are properly tracked."""
        initial_best_ei = self.acq.best_so_far_ei
        initial_best_pi = self.acq.best_so_far_pi
        
        # Run EI to update best_so_far_ei
        self.acq.expected_improvement(
            self.sequences_single, self.means_single, self.stds_single
        )
        
        # Run PI to update best_so_far_pi
        self.acq.probability_of_improvement(
            self.sequences_single, self.means_single, self.stds_single
        )
        
        # Best values should have been updated
        # (exact values depend on maximize parameter and current implementation)
        assert isinstance(self.acq.best_so_far_ei, float)
        assert isinstance(self.acq.best_so_far_pi, float)