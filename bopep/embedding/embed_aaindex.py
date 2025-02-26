import csv
import numpy as np
from tqdm import tqdm
import importlib.resources as pkg_resources

def get_aaindex_path():
    return pkg_resources.files("bopep.embedding").joinpath("aaindex1.csv")

def embed_aaindex(peptide_sequences):
    """
    Generate peptide embeddings using aaindex properties.
    
    Each row in the aaindex CSV (except for the header) is treated as one property.
    For each peptide, the embedding vector is constructed by averaging the property
    values (one per amino acid) for each aaindex property.
    """

    aaindex_properties = []
    aaindex_file = get_aaindex_path()

    with aaindex_file.open("r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            description = row["Description"]
            mapping = {}
            if 'NA' in row.values():
                continue
            for aa, value in row.items():
                if aa == "Description":
                    continue
                mapping[aa] = float(value)
            aaindex_properties.append((description, mapping))
    
    embeddings = {}

    for peptide in tqdm(peptide_sequences, desc="Generating aaindex embeddings"):
        embedding = []
        for description, mapping in aaindex_properties:
            values = [mapping.get(letter, 0.0) for letter in peptide]
            avg_value = sum(values) / len(values) if values else 0.0
            embedding.append(avg_value)
        embeddings[peptide] = np.array(embedding)
    
    return embeddings


if __name__ == "__main__":
    peptides = ["ACDE", "WXYZ", "MNOPQR"]
    embeddings = embed_aaindex(peptides)
    for peptide, emb in embeddings.items():
        print(f"Peptide: {peptide}, Embedding: {emb}")
