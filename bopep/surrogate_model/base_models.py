import torch
from typing import List, Optional, Tuple
import torch.nn.utils.rnn as rnn_utils
import math


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


class SelfAttention(torch.nn.Module):
    """
    Self-attention mechanism to focus on important parts of the sequence.
    """
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = torch.nn.Linear(hidden_dim, attention_dim)
        self.query = torch.nn.Linear(attention_dim, 1)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        
    def forward(self, sequences: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention over sequence positions.
        
        Args:
            sequences: [batch_size, seq_length, hidden_dim]
            mask: [batch_size, seq_length] - binary mask (1 for valid positions, 0 for padding)
            
        Returns:
            context_vector: [batch_size, hidden_dim] - weighted sum of sequence vectors
            attention_weights: [batch_size, seq_length] - attention distribution
        """
        # Project to attention space
        projection = torch.tanh(self.projection(sequences))  # [batch_size, seq_length, attention_dim]
        
        # Calculate attention scores
        energy = self.query(projection).squeeze(-1)  # [batch_size, seq_length]
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask is on the same device as energy
            mask = mask.to(energy.device)
            energy = energy.masked_fill(mask == 0, -1e10)
            
        # Normalize attention scores
        attention_weights = torch.nn.functional.softmax(energy, dim=-1)  # [batch_size, seq_length]
        
        # Apply attention to get context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), sequences
        ).squeeze(1)  # [batch_size, hidden_dim]
        
        # Apply layer normalization
        context_vector = self.layer_norm(context_vector)
        
        return context_vector, attention_weights


class PositionalEncoding(torch.nn.Module):
    """
    Adds positional information to the sequence embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (won't be updated during backprop)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to the input embeddings.
        
        Args:
            x: [batch_size, seq_length, embedding_dim]
            
        Returns:
            x + positional_encoding: [batch_size, seq_length, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]


class BiLSTMNetwork(BaseNetwork):
    """
    Enhanced Bidirectional LSTM network for sequence data with attention mechanism,
    layer normalization, and improved pooling strategies.
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
        
        # Add positional encoding
        self.positional_encoding = PositionalEncoding(input_dim)
        
        # Bidirectional LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        
        # Layer normalization for better training stability
        self.layer_norm = torch.nn.LayerNorm(hidden_dim * 2)
        
        # Self-attention mechanism
        self.attention = SelfAttention(hidden_dim * 2)
        
        # Output layer
        self.fc1 = torch.nn.Linear(hidden_dim * 4, hidden_dim)  # 4*hidden_dim from concatenating attention + max pooling
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        x shape: [batch_size, seq_length, input_dim]
        lengths: list of actual sequence lengths for each item in the batch (if available)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Add positional encodings for better sequence position awareness
        x = self.positional_encoding(x)
        
        # Process sequences (either packed or as is)
        if lengths is not None:
            # Create sequence mask for attention - make sure it's on the same device as x
            max_len = x.size(1)
            mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < torch.tensor(lengths, device=device).unsqueeze(1)
            
            # Use pack_padded_sequence to ignore padded timesteps
            x_packed = rnn_utils.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.lstm(x_packed)
            # Unpack to get (N, L_max, 2*hidden_dim)
            lstm_out, _ = rnn_utils.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
        else:
            # If no lengths provided, create a mask assuming all positions are valid
            mask = torch.ones(batch_size, x.size(1), device=device).bool()
            lstm_out, _ = self.lstm(x)
        
        # Apply layer normalization for stability
        normalized_lstm_out = self.layer_norm(lstm_out)
        
        # Get attention-weighted context vector
        context_vector, _ = self.attention(normalized_lstm_out, mask)  # [batch_size, hidden_dim*2]
        
        # Max pooling across sequence dimension
        max_pooled, _ = torch.max(normalized_lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # Concatenate different feature views
        combined = torch.cat([context_vector, max_pooled], dim=-1)  # [batch_size, hidden_dim*4]
        
        # Project to output
        hidden = self.fc1(combined)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)  # [batch_size, output_dim]
        
        return output


class BiGRUNetwork(BaseNetwork):
    """
    Enhanced Bidirectional GRU network with attention and improved feature extraction.
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
        
        # Add positional encoding
        self.positional_encoding = PositionalEncoding(input_dim)
        
        # Bidirectional GRU
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        
        # Layer normalization
        self.layer_norm = torch.nn.LayerNorm(hidden_dim * 2)
        
        # Self-attention
        self.attention = SelfAttention(hidden_dim * 2)
        
        # Output layers
        self.fc1 = torch.nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.activation = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
    
    def forward(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        x shape: [batch_size, seq_length, input_dim]
        lengths: list of actual sequence lengths for each item in batch
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Add positional encodings
        x = self.positional_encoding(x)
        
        # Process sequences
        if lengths is not None:
            # Create mask for attention - on the same device as x
            max_len = x.size(1)
            mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < torch.tensor(lengths, device=device).unsqueeze(1)
            
            # Pack sequences
            x_packed = rnn_utils.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            gru_out_packed, _ = self.gru(x_packed)
            # Unpack
            gru_out, _ = rnn_utils.pad_packed_sequence(
                gru_out_packed, batch_first=True
            )
        else:
            mask = torch.ones(batch_size, x.size(1), device=device).bool()
            gru_out, _ = self.gru(x)
        
        # Apply layer normalization
        normalized_gru_out = self.layer_norm(gru_out)
        
        # Get attention-weighted representation
        context_vector, _ = self.attention(normalized_gru_out, mask)  # [batch_size, hidden_dim*2]
        
        # Max pooling for another view of the sequence
        max_pooled, _ = torch.max(normalized_gru_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # Combine features
        combined = torch.cat([context_vector, max_pooled], dim=-1)  # [batch_size, hidden_dim*4]
        
        # Output projection
        hidden = self.fc1(combined)
        hidden = self.activation(hidden)
        hidden = self.dropout1(hidden)
        output = self.fc2(hidden)
        output = self.dropout2(output)
        
        return output
