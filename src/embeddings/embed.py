import os
from tqdm import tqdm
import torch
import esm


def embed(peptide_sequences, model_path="esm2_t33_650M_UR50D.pt"):
    
    # Check if the model file exists
    if os.path.exists(model_path):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        print(f"Loaded model from local path: {model_path}")
    else:
        # Download the model if it does not exist locally
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # Save the model for future use
        torch.save(model.state_dict(), model_path)
        print(f"Downloaded and saved model to: {model_path}")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
 
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    embeddings = {}
    sequence_data = [(f"seq_{i}", seq) for i, seq in enumerate(peptide_sequences)]

    batch_size = 64
    for i in tqdm(range(0, len(sequence_data), batch_size), desc="Generating embeddings"):
        batch_sequences = sequence_data[i : i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_sequences)
        
        with torch.no_grad():
            results = model(
                batch_tokens, repr_layers=[33], return_contacts=False
            )  # layer 33 is the final layer in the model
        
        token_embeddings = results["representations"][33]
        
        for j, (label, seq) in enumerate(batch_sequences):
            seq_len = len(seq)
            embedding = token_embeddings[j, 1 : seq_len + 1].mean(0).cpu().numpy()
            embeddings[seq] = embedding

    return embeddings

