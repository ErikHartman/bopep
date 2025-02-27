import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

class PeptideDataset(Dataset):
    def __init__(self, embeddings_dict):
        self.peptides = list(embeddings_dict.keys())
        self.embeddings = list(embeddings_dict.values())
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32)

def collate_fn(batch):
    """Improved collate function with sorted sequences for better packing"""
    # Sort batch by descending sequence length
    batch.sort(key=lambda x: x.shape[0], reverse=True)
    lengths = [x.shape[0] for x in batch]
    max_length = lengths[0]
    
    # Pad sequences
    padded_batch = torch.nn.utils.rnn.pad_sequence(
        batch, 
        batch_first=True, 
        padding_value=0.0
    )
    return padded_batch, torch.tensor(lengths)

class PeptideAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super(PeptideAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            bidirectional=True,
            batch_first=True
        )
        self.encoder_ln = nn.LayerNorm(hidden_dim*2)
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_dim*2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim,
            input_dim,
            bidirectional=True,
            batch_first=True
        )
        self.decoder_ln = nn.LayerNorm(input_dim*2)
        self.output_fc = nn.Sequential(
            nn.Linear(input_dim*2, input_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for LSTM parameters"""
        for name, param in self.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, lengths):
        # Encoder
        packed = pack_padded_sequence(
            x, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Encoder processing
        packed_out, (hidden, cell) = self.encoder_lstm(packed)
        encoded, _ = pad_packed_sequence(packed_out, batch_first=True)
        encoded = self.encoder_ln(encoded)
        latent = self.encoder_fc(encoded)
        
        # Decoder processing
        decoded = self.decoder_fc(latent)
        decoded = nn.functional.relu(decoded)
        
        # Process through decoder LSTM
        decoded_out, _ = self.decoder_lstm(decoded)
        decoded_out = self.decoder_ln(decoded_out)
        reconstructed = self.output_fc(decoded_out)
        
        return reconstructed, latent