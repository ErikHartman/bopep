from bopep.embedding.embed_esm import embed
from bopep.embedding.utils import filter_peptides   


class Embedder:
    def __init__(self):
        pass

    def embed_esm(self, peptides):
        peptides = filter_peptides(peptides)
        embeddings = embed(peptides)
        return embeddings