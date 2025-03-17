from typing import Dict, Literal, Any, Optional

from bopep.surrogate_model.base_models import (
    BaseNetwork,
    MLPNetwork,
    BiLSTMNetwork,
    BiGRUNetwork,
)
import torch


class NetworkFactory:
    """
    Factory class for creating different types of neural networks with appropriate parameters.
    Provides validation and standardization of parameters across network types.
    """

    @staticmethod
    def get_network(
        network_type: Literal["mlp", "bilstm", "bigru"],
        input_dim: int,
        output_dim: int = 1,
        **kwargs,
    ) -> BaseNetwork:
        """
        Create and return a neural network of the specified type with appropriate parameters.

        Args:
            network_type: Type of network to create ("mlp", "bilstm", "bigru")
            input_dim: Input dimension for the network
            output_dim: Output dimension for the network (default: 1)
            **kwargs: Additional parameters specific to the network type

        Returns:
            Instantiated neural network
        """
        # Default parameters
        network_params = {
            "dropout_rate": kwargs.get("dropout_rate", 0.0),
            "input_dim": input_dim,
            "output_dim": output_dim,
        }

        device = kwargs.get("device", torch.device("cpu"))

        if network_type == "mlp":
            # Handle MLP specific parameters
            hidden_dims = kwargs.get("hidden_dims", [128, 64])
            network_params["hidden_dims"] = hidden_dims
            model = MLPNetwork(**network_params)

        elif network_type in ["bilstm"]:
            # Handle BiLSTM specific parameters
            hidden_dim = kwargs.get("hidden_dim", kwargs.get("lstm_hidden_dim", 128))
            num_layers = kwargs.get("num_layers", kwargs.get("lstm_layers", 1))

            # Ensure hidden_dim is an int
            if hidden_dim is None:
                hidden_dim = 128
                print(f"Warning: hidden_dim was None, defaulting to {hidden_dim}")

            network_params["hidden_dim"] = hidden_dim
            network_params["num_layers"] = num_layers
            model = BiLSTMNetwork(**network_params)

        elif network_type in ["bigru"]:
            # Handle BiGRU specific parameters
            hidden_dim = kwargs.get("hidden_dim", kwargs.get("gru_hidden_dim", 128))
            num_layers = kwargs.get("num_layers", kwargs.get("gru_layers", 1))

            # Ensure hidden_dim is an int
            if hidden_dim is None:
                hidden_dim = 128
                print(f"Warning: hidden_dim was None, defaulting to {hidden_dim}")

            network_params["hidden_dim"] = hidden_dim
            network_params["num_layers"] = num_layers
            model = BiGRUNetwork(**network_params)

        else:
            raise ValueError(f"Unsupported network_type: {network_type}")

        model.to(device)
        return model

    @staticmethod
    def validate_params(network_type: str, params: Dict[str, Any]) -> None:
        """
        Validate that required parameters for a network type are present.

        Args:
            network_type: Type of network ("mlp", "bilstm", "bigru")
            params: Dictionary of parameters to validate

        Raises:
            ValueError: If required parameters are missing
        """

        required_params = {
            "mlp": ["input_dim", "hidden_dims"],
            "bilstm": ["input_dim", "hidden_dim", "num_layers"],
            "bigru": ["input_dim", "hidden_dim", "num_layers"],
        }

        if network_type not in required_params:
            raise ValueError(f"Unknown network type: {network_type}")

        missing = [p for p in required_params[network_type] if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {network_type}: {missing}"
            )
