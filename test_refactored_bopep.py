#!/usr/bin/env python3

"""
Test script to validate the refactored BoPep class works correctly
with the new SurrogateModelManager.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import numpy as np
from unittest.mock import patch

def test_bopep_instantiation():
    """Test that BoPep can be instantiated with SurrogateModelManager."""
    
    # Import first, then patch the validation function
    from bopep.search.optimization import BoPep
    
    # Mock the dependency validation to avoid colabfold issues
    with patch('bopep.search.optimization._validate_dependencies'):
        
        try:
            # Test basic instantiation
            bopep = BoPep(
                surrogate_model_kwargs={
                    'model_type': 'MonteCarloDropout',
                    'network_type': 'mlp',
                    'hidden_layers': [128, 64],
                    'dropout_rate': 0.1
                },
                docker_kwargs={
                    'models': ['alphafold'],
                    'output_dir': './test_docker_output'
                },
                log_dir='./test_output'
            )
            
            print("✓ BoPep instantiation successful!")
            
            # Check that the surrogate manager is properly initialized
            assert hasattr(bopep, 'surrogate_manager'), "Missing surrogate_manager attribute"
            assert bopep.surrogate_manager is not None, "surrogate_manager is None"
            
            print(f"✓ SurrogateModelManager type: {type(bopep.surrogate_manager).__name__}")
            print(f"✓ Model kwargs: {bopep.surrogate_manager.surrogate_model_kwargs}")
            
            # Check that other components are initialized
            assert hasattr(bopep, 'scorer'), "Missing scorer attribute"
            assert hasattr(bopep, 'docker'), "Missing docker attribute"
            assert hasattr(bopep, 'selector'), "Missing selector attribute"
            assert hasattr(bopep, 'scores_to_objective'), "Missing scores_to_objective attribute"
            
            print("✓ All BoPep components properly initialized")
            
            # Test that the manager has the required methods
            assert hasattr(bopep.surrogate_manager, 'predict'), "Missing predict method"
            assert hasattr(bopep.surrogate_manager, 'train_with_validation_split'), "Missing train_with_validation_split method"
            assert hasattr(bopep.surrogate_manager, 'optimize_hyperparameters'), "Missing optimize_hyperparameters method"
            
            print("✓ SurrogateModelManager has all required methods")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during BoPep instantiation: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_bopep_manager_integration():
    """Test that BoPep integrates properly with SurrogateModelManager methods."""
    
    from bopep.search.optimization import BoPep
    
    with patch('bopep.search.optimization._validate_dependencies'):
        
        try:
            bopep = BoPep(
                surrogate_model_kwargs={
                    'model_type': 'MonteCarloDropout',
                    'network_type': 'mlp',
                    'hidden_layers': [64, 32],
                    'dropout_rate': 0.1
                },
                docker_kwargs={
                    'models': ['alphafold'],
                    'output_dir': './test_docker_output'
                },
                log_dir='./test_output'
            )
            
            # Test that we can access manager methods
            manager = bopep.surrogate_manager
            
            # Check model configuration
            assert manager.surrogate_model_kwargs['model_type'] == 'MonteCarloDropout'
            assert manager.surrogate_model_kwargs['network_type'] == 'mlp'
            assert manager.surrogate_model_kwargs['hidden_layers'] == [64, 32]
            
            print("✓ SurrogateModelManager configuration correctly stored")
            
            # Test device handling (should use GPU if available)
            manager.set_device()
            print(f"✓ Device set to: {manager.device}")
            
            # Test model class resolution
            model_class = manager._get_model_class()
            print(f"✓ Model class resolved: {model_class.__name__}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during manager integration test: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_backward_compatibility():
    """Test that key BoPep functionality still works after refactoring."""
    
    from bopep.search.optimization import BoPep
    
    with patch('bopep.search.optimization._validate_dependencies'):
        
        try:
            bopep = BoPep(
                surrogate_model_kwargs={
                    'model_type': 'MonteCarloDropout',
                    'network_type': 'mlp'
                },
                docker_kwargs={
                    'models': ['alphafold'],
                    'output_dir': './test_docker_output'
                }
            )
            
            # Check that the run method still exists and has the correct signature
            assert hasattr(bopep, 'run'), "Missing run method"
            
            # Check that the run method has the expected parameters
            import inspect
            run_params = list(inspect.signature(bopep.run).parameters.keys())
            expected_params = ['schedule', 'batch_size', 'target_structure_path', 'embeddings']
            
            for param in expected_params:
                assert param in run_params, f"Missing expected parameter: {param}"
            
            print("✓ BoPep.run method has correct signature")
            
            # Check that key methods/attributes are no longer present (removed during refactoring)
            removed_methods = ['_optimize_hyperparameters', '_initialize_model']
            for method in removed_methods:
                assert not hasattr(bopep, method), f"Method {method} should have been removed"
            
            print("✓ Obsolete methods successfully removed")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during backward compatibility test: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("Testing refactored BoPep class with SurrogateModelManager...")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Instantiation Test", test_bopep_instantiation),
        ("Manager Integration Test", test_bopep_manager_integration),
        ("Backward Compatibility Test", test_backward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("-" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! BoPep refactoring successful!")
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)
