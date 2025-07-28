from tqdm import tqdm
import torch
import esm
from typing import Optional


def embed_esm(
    peptide_sequences,
    model :torch.nn.Module,
    alphabet: esm.Alphabet,
    average: bool = True,
    batch_size: int = 128,
    device: Optional[str] = None,
):
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Set the model to evaluation mode

    embeddings = {}
    sequence_data = [(f"seq_{i}", seq) for i, seq in enumerate(peptide_sequences)]

    for i in tqdm(
        range(0, len(sequence_data), batch_size), desc="Generating ESM embeddings"
    ):
        batch_sequences = sequence_data[i : i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_sequences)

        # Move input tokens to the same device as the model
        batch_tokens = batch_tokens.to(device)

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

    print("ESM embedding dim: ", embedding.shape)
    return embeddings
