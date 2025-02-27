import csv
import numpy as np
from tqdm import tqdm
import importlib.resources as pkg_resources

import csv
import numpy as np
from tqdm import tqdm
import importlib.resources as pkg_resources


def get_aaindex_path():
    return pkg_resources.files("bopep.embedding").joinpath("aaindex1.csv")


def embed_aaindex(peptide_sequences, average: bool = True):
    """
    Generate peptide embeddings using aaindex properties.

    - When `average=True`: Returns an embedding vector of shape `(num_properties,)` for each peptide.
    - When `average=False`: Returns an embedding of shape `(sequence_length, num_properties)`,
      where each amino acid gets its own vector.
    """

    aaindex_properties = []
    aaindex_file = get_aaindex_path()

    with aaindex_file.open("r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            description = row["Description"]
            mapping = {}
            if "NA" in row.values():
                continue
            for aa, value in row.items():
                if aa == "Description":
                    continue
                mapping[aa] = float(value)
            aaindex_properties.append((description, mapping))

    num_properties = len(aaindex_properties)  # Number of AAIndex properties
    embeddings = {}

    for peptide in tqdm(peptide_sequences, desc="Generating aaindex embeddings"):
        seq_length = len(peptide)
        embedding_matrix = np.zeros((seq_length, num_properties))

        for prop_idx, (description, mapping) in enumerate(aaindex_properties):
            values = [mapping.get(letter, 0.0) for letter in peptide]
            embedding_matrix[:, prop_idx] = values  # Assign each AA property to its row

        if average:
            embeddings[peptide] = embedding_matrix.mean(
                axis=0
            )  # Shape: (num_properties,)
        else:
            embeddings[peptide] = (
                embedding_matrix  # Shape: (sequence_length, num_properties)
            )

    print("AAIndex embedding dim:", embeddings[peptide_sequences[0]].shape)
    return embeddings
