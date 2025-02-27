import torch
import torch.nn as nn
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
    # Find the maximum length in the batch
    max_length = max([item.shape[0] for item in batch])
    
    # Pad all tensors to the maximum length
    padded_batch = [torch.nn.functional.pad(item, (0, 0, 0, max_length - item.shape[0])) for item in batch]
    lengths = [item.shape[0] for item in batch]
    
    return torch.stack(padded_batch), torch.tensor(lengths)

class PeptideAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(PeptideAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_rnn1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.encoder_rnn2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.encoder_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, latent_dim)
        self.encoder_relu = nn.ReLU()
        
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.decoder_relu = nn.ReLU()
        self.decoder_rnn1 = nn.LSTM(hidden_dim * 2, hidden_dim * 2, batch_first=True, bidirectional=True)
        self.decoder_rnn2 = nn.LSTM(hidden_dim * 4, input_dim, batch_first=True, bidirectional=True)
        self.final_fc = nn.Linear(input_dim * 2, input_dim)
        
    def forward(self, x, lengths):
        # Encode
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_h1, _ = self.encoder_rnn1(packed_x)
        packed_h2, _ = self.encoder_rnn2(packed_h1)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_h2, batch_first=True)
        
        latent = self.encoder_relu(self.encoder_fc1(h))
        latent = self.encoder_fc2(latent)
        
        # Decode
        decoded_h = self.decoder_relu(self.decoder_fc1(latent))
        decoded_h = self.decoder_fc2(decoded_h)
        decoded_x1, _ = self.decoder_rnn1(decoded_h)
        decoded_x2, _ = self.decoder_rnn2(decoded_x1)
        decoded_x = self.final_fc(decoded_x2)
        
        return decoded_x, latent