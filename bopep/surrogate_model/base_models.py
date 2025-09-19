import torch
from typing import List, Optional, Tuple
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import math

class BaseNetwork(nn.Module):
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
        n_objectives: int = 1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        # For multi-objective, final output dimension is output_dim * n_objectives
        final_output_dim = output_dim * n_objectives
        layers.append(nn.Linear(prev_dim, final_output_dim))
        self.network = nn.Sequential(*layers)
        self.n_objectives = n_objectives
        self.output_dim = output_dim

    def forward(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        # lengths ignored for MLP
        output = self.network(x)
        
        # Reshape output for multi-objective case
        if self.n_objectives > 1:
            batch_size = output.shape[0]
            # Reshape from [batch_size, output_dim * n_objectives] to [batch_size, n_objectives, output_dim]
            output = output.view(batch_size, self.n_objectives, self.output_dim)
        
        return output


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
        
        # Apply mask
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
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        

        pe[:,0::2] = torch.sin(position * div_term)
        odd_len = pe[:,1::2].size(1)
        pe[:,1::2] = torch.cos(position * div_term[:odd_len])
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


class RNNetwork(nn.Module):
    """
    Unified BiLSTM/BiGRU + SelfAttention + max-pooling + MLP head.
    Exactly matches your original BiLSTMNetwork and BiGRUNetwork logic.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float = 0.0,
        output_dim: int = 1,
        architecture: str = "gru",   # "gru" or "lstm"
        n_objectives: int = 1,
    ):
        super().__init__()
        self.architecture = architecture
        self.n_objectives = n_objectives
        self.output_dim = output_dim
        
        # positional encoding (same as before)
        self.positional_encoding = PositionalEncoding(input_dim)
        
        # choose RNN class
        rnn_cls = nn.GRU if architecture=="gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers>1 else 0.0,
        )
        
        # self‐attention over the 2*hidden_dim
        self.attention = SelfAttention(hidden_dim*2)
        
        # head sizes
        in_dim = hidden_dim * 6  # [attn(2H) + maxpool(2H) + meanpool(2H)]
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # For multi-objective, final output dimension is output_dim * n_objectives
        final_output_dim = output_dim * n_objectives
        self.fc2 = nn.Linear(hidden_dim, final_output_dim)

    def forward(
        self,
        x: torch.Tensor,               # [B, T, input_dim]
        lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        B, T, _ = x.size()
        device = x.device

        # add positional encodings
        x = self.positional_encoding(x)  # [B, T, input_dim]

        # compute valid token mask
        if lengths is not None:
            if isinstance(lengths, torch.Tensor): #Sometimes lengths is a tensor and sometimes a list
                lt = lengths.clone().detach().to(device)
            else:
                lt = torch.tensor(lengths, device=device, dtype=torch.long)
            mask = torch.arange(T, device=device)[None, :] < lt[:, None]  # [B, T]
            pack_x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.rnn(pack_x)
            h, _ = rnn_utils.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            mask = torch.ones(B, T, device=device, dtype=torch.bool)
            h, _ = self.rnn(x)  # [B, T, 2*hidden_dim]

        # attention (drop layer-norm here)
        ctx, _ = self.attention(h, mask)  # [B, 2H]

        # masked max pooling
        masked_h = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))  # [B, T, 2H]
        max_pooled, _ = masked_h.max(dim=1)                           # [B, 2H]

        # masked mean pooling
        masked_h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        sum_pooled = masked_h.sum(dim=1)                              # [B, 2H]
        lengths_tensor = mask.sum(dim=1, keepdim=True)                # [B, 1]
        mean_pooled = sum_pooled / lengths_tensor.clamp(min=1)        # [B, 2H]

        # concatenate context + max + mean pooling
        feats = torch.cat([ctx, max_pooled, mean_pooled], dim=-1)     # [B, 6H]

        # fc1 -> activation -> dropout -> fc2
        h1 = self.fc1(feats)
        h1 = self.act(h1)
        h1 = self.dropout(h1)
        out = self.fc2(h1) 

        # Reshape output for multi-objective case
        if self.n_objectives > 1:
            batch_size = out.shape[0]
            # Reshape from [batch_size, output_dim * n_objectives] to [batch_size, n_objectives, output_dim]
            out = out.view(batch_size, self.n_objectives, self.output_dim)

        return out