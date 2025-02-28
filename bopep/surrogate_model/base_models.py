import torch
from typing import List, Optional
import torch.nn.utils.rnn as rnn_utils


class BaseNetwork(torch.nn.Module):
    """Base class for all network architectures."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        raise NotImplementedError("Subclasses must implement forward.")


class MLPNetwork(BaseNetwork):
    """Simple feed-forward neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.0,
        output_dim: int = 1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        # lengths ignored for MLP
        return self.network(x)


class BiLSTMNetwork(BaseNetwork):
    """Bidirectional LSTM network for sequence data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float = 0.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        # BiLSTM output dimension is 2*hidden_dim
        self.fc = torch.nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        x shape: [batch_size, seq_length, input_dim]
        lengths: list of actual sequence lengths for each item in the batch (if available).
        """
        if lengths is not None:
            # Use pack_padded_sequence to ignore padded timesteps
            x_packed = rnn_utils.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.lstm(x_packed)
            # Unpack to get (N, L_max, 2*hidden_dim) if needed
            lstm_out, _ = rnn_utils.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
        else:
            # If no lengths provided, just run LSTM on the full padded sequence
            lstm_out, _ = self.lstm(x)

        # Simplest approach: we rely on the last time step *in the padded sequence*. Might not be best option! TODO
        final_hidden = lstm_out[:, -1, :]  # shape (N, 2*hidden_dim)

        final_hidden = self.dropout(final_hidden)
        output = self.fc(final_hidden)  # shape (N, output_dim)
        return output


class BiGRUNetwork(BaseNetwork):
    """
    Bidirectional GRU network for sequence data.
    Compared to LSTM, GRU has fewer parameters and is faster to train.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float = 0.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        # BiGRU output dimension is 2*hidden_dim (forward + backward)
        self.fc = torch.nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_length, input_dim]
        gru_out, _ = self.gru(x)
        # Use the final hidden state from both directions
        final_hidden = gru_out[:, -1, :]
        final_hidden = self.dropout(final_hidden)
        output = self.fc(final_hidden)
        return output
