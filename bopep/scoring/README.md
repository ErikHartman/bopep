# Scoring module

The scoring module should hold a class/set of functions that score protein-peptide complexes.

It should contain several scoring functions, and allow the user to select among these for their Bayesian optimziation runs.

```py
scorer = Scorer()
scorer.score(complex_pdb, loss_fxn = "evobind", protein_chain = "A", peptide_chain="B")
```
