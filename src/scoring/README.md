# Scoring module

The scoring module should hold a class/set of functions that score protein-peptide complexes.

It should contain several scoring functions, and allow the user to select among these for their Bayesian optimziation runs.

I imagine something like:

```py
scorer = Scorer()
scorer.score(complex_pdb, loss_fxn = "evobind", protein_chain = "A", peptide_chain="B")
```

## List of known scoring functions

1. EvoBinds loss function 
2. GDockScore (requires cloning a git repo) https://gitlab.com/mcfeemat/gdockscore
3. Ours previously used (linear combination of Rosetta scores and ipTM)
4. Raw ipTM


Peptide-protein databases
BioLip
PDBBind
Propedia