import torch
from typing import List, Optional, Tuple
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.nn as nn

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
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        # lengths ignored for MLP
        return self.network(x)

class RNNetwork(BaseNetwork):
    """
    Recurrent neural network for peptide binding prediction.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_rate: float = 0.01,
        output_dim: int = 1,
        max_len: int = 100,
        architecture: str = "gru",
    ):
        super().__init__()
        self.architecture = architecture
        
        # Input normalization (optional - depends on your VAE output)
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Recurrent layer
        if architecture == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,  # Use VAE latent dim directly
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
            )
        else:  # lstm
            self.rnn = nn.LSTM(
                input_size=input_dim,  # Use VAE latent dim directly
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
            )
        
        # Layer norm on RNN output
        self.rnn_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Enhanced attention with relative positions
        self.attention = RelativeAttention(
            hidden_dim=hidden_dim * 2,
            attention_dim=hidden_dim,
            max_len=max_len,
            dropout=dropout_rate,
        )
        
        # Feature dropout
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Output MLP with residual connection
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[List[int]] = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        device = x.device
        
        # Project and normalize input (VAE latent space)
        x = self.input_norm(x)
        
        # Create mask and pack sequences
        if lengths is not None:
            lengths_t = torch.tensor(lengths, device=device)
            mask = torch.arange(T, device=device)[None, :] < lengths_t[:, None]
            
            # Pack for efficient RNN processing
            x_packed = rnn_utils.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            if self.architecture == "gru":
                out_packed, _ = self.rnn(x_packed)
            else:
                out_packed, _ = self.rnn(x_packed)
            
            # Unpack
            h, _ = rnn_utils.pad_packed_sequence(out_packed, batch_first=True)
        else:
            mask = torch.ones(B, T, device=device, dtype=torch.bool)
            if self.architecture == "gru":
                h, _ = self.rnn(x)
            else:
                h, _ = self.rnn(x)
        
        # Normalize RNN output
        h = self.rnn_norm(h)  # [B, T, 2H]
        
        # Get attention context
        ctx, attn_weights = self.attention(h, mask)  # [B, 2H]
        
        # Multiple pooling strategies
        max_pooled, _ = h.max(dim=1)  # [B, 2H]
        
        # Proper mean pooling (accounting for padding)
        if lengths is not None:
            sum_pooled = h.sum(dim=1)  # [B, 2H]
            mean_pooled = sum_pooled / lengths_t[:, None].float()
        else:
            mean_pooled = h.mean(dim=1)  # [B, 2H]
        
        # Combine all features
        features = torch.cat([ctx, max_pooled, mean_pooled], dim=-1)  # [B, 6H]
        features = self.feature_dropout(features)
        
        # Final prediction
        output = self.output_mlp(features)
        
        return output


class RelativeAttention(nn.Module):
    """
    Self-attention with relative positional encoding for peptide sequences.
    """
    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = 64,
        max_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.max_len = max_len
        
        # Content-based attention
        self.content_proj = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.query = nn.Linear(attention_dim, 1, bias=False)
        
        # Relative positional embeddings
        self.rel_pos_emb = nn.Embedding(2 * max_len - 1, attention_dim)
        
        # Attention dropout
        self.attention_dropout = nn.Dropout(dropout)
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = sequences.size()
        device = sequences.device
        
        # Content-based scores
        content_proj = torch.tanh(self.content_proj(sequences))  # [B, T, A]
        content_scores = self.query(content_proj).squeeze(-1)  # [B, T]
        
        # Relative positional bias
        positions = torch.arange(T, device=device)
        relative_positions = positions[None, :] - positions[:, None]  # [T, T]
        relative_positions = relative_positions + (self.max_len - 1)  # Shift to positive
        relative_positions = torch.clamp(relative_positions, 0, 2 * self.max_len - 2)
        
        rel_embeddings = self.rel_pos_emb(relative_positions)  # [T, T, A]
        
        # Compute relative bias: sum over attention dimension
        rel_bias = torch.einsum('bta,tua->bt', content_proj, rel_embeddings)
        
        # Combine content and positional information
        attention_scores = content_scores + rel_bias
        
        # Apply mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention
        context = torch.bmm(attention_weights.unsqueeze(1), sequences).squeeze(1)
        
        # Layer normalization
        context = self.layer_norm(context)
        
        return context, attention_weights
