import os
from tqdm import tqdm
import torch
import esm


def embed_esm(
    peptide_sequences, model_path="esm2_t33_650M_UR50D.pt", average: bool = True
):
    # Check if the model file exists locally
    if model_path:
        model_path = os.path.abspath(model_path)
        if os.path.exists(model_path):
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    else:
        # Download the model if it does not exist locally
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("ESM moved to GPU.")
    else:
        print("GPU not available. ESM remains on CPU.")

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Set the model to evaluation mode

    embeddings = {}
    sequence_data = [(f"seq_{i}", seq) for i, seq in enumerate(peptide_sequences)]

    batch_size = 64
    for i in tqdm(
        range(0, len(sequence_data), batch_size), desc="Generating embeddings"
    ):
        batch_sequences = sequence_data[i : i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_sequences)

        # Move input tokens to GPU if available
        batch_tokens = batch_tokens.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate embeddings with no gradient tracking
        with torch.no_grad():
            results = model(
                batch_tokens, repr_layers=[33], return_contacts=False
            )  # Layer 33 is the final layer

        # Retrieve the embeddings from the model output
        token_embeddings = results["representations"][33]

        # Process and store embeddings for each sequence in the batch
        for j, (label, seq) in enumerate(batch_sequences):
            seq_len = len(seq)
            # Average embedding over sequence length and move to CPU for storage
            if average:
                embedding = token_embeddings[j, 1 : seq_len + 1].mean(0).cpu().numpy()
            else:
                embedding = token_embeddings[j, 1 : seq_len + 1].cpu().numpy()
            embeddings[seq] = embedding
    print("ESM embedding dim: ", embeddings[peptide_sequences[0]].shape)
    return embeddings
