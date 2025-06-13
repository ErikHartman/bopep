import torch
from typing import List, Optional, Tuple
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
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


class SelfAttention(nn.Module):
    """
    Self-attention mechanism to focus on important parts of the sequence.
    """
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, attention_dim)
        self.query      = nn.Linear(attention_dim, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        sequences: torch.Tensor,               # [B, T, hidden_dim]
        mask: Optional[torch.Tensor] = None,   # [B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) project into attention space
        proj   = torch.tanh(self.projection(sequences))      # [B, T, A]
        # 2) score each timestep
        energy = self.query(proj).squeeze(-1)                # [B, T]

        # 3) mask out padding
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # 4) normalize
        weights = F.softmax(energy, dim=-1)                  # [B, T]
        
        # 5) weighted sum of original sequence vectors
        context = torch.bmm(weights.unsqueeze(1), sequences) \
                         .squeeze(1)                        # [B, hidden_dim]
        
        # 6) layer-norm for stability
        context = self.layer_norm(context)                   # [B, hidden_dim]
        
        return context, weights


class PositionalEncoding(nn.Module):
    """
    Adds sine/cosine positional encodings to token embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
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
    ):
        super().__init__()
        self.architecture = architecture
        
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
        
        # layer‐norm on RNN outputs
        self.layer_norm = nn.LayerNorm(hidden_dim*2)
        
        # self‐attention over the 2*hidden_dim
        self.attention = SelfAttention(hidden_dim*2)
        
        # head sizes
        in_dim = hidden_dim * 4  # [attn(2H) + maxpool(2H)]
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        
        # replicate original dropout/fc2 logic exactly
        if architecture == "gru":
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2       = nn.Linear(hidden_dim, output_dim)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:  # lstm
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2     = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,               # [B, T, input_dim]
        lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        B, T, _ = x.size()
        device  = x.device
        
        # add positional encodings
        x = self.positional_encoding(x)  # [B, T, input_dim]
        
        # optionally pack padded sequences
        if lengths is not None:
            lt     = torch.tensor(lengths, device=device)
            mask   = torch.arange(T, device=device)[None,:] < lt[:,None]  # [B,T]
            pack_x = rnn_utils.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            out_p, _ = self.rnn(pack_x)
            h, _     = rnn_utils.pad_packed_sequence(out_p, batch_first=True)
        else:
            mask = torch.ones(B, T, device=device, dtype=torch.bool)
            h, _ = self.rnn(x)  # [B, T, 2*hidden_dim]
        
        # layer-norm
        h = self.layer_norm(h)  # [B, T, 2H]
        
        # attention + max-pooling
        ctx, _      = self.attention(h, mask)     # [B, 2H]
        max_pooled, _ = h.max(dim=1)              # [B, 2H]
        
        # concat features
        feats = torch.cat([ctx, max_pooled], dim=-1)  # [B, 4H]
        
        # fc1 -> activation -> dropout(s) -> fc2 (exactly as original)
        h1 = self.fc1(feats)
        h1 = self.act(h1)
        
        if self.architecture=="gru":
            h1 = self.dropout1(h1)
            out = self.fc2(h1)
            out = self.dropout2(out)
        else:
            h1 = self.dropout(h1)
            out = self.fc2(h1)
        
        return out