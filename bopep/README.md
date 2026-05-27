# Modules

This directory contains the core `bopep` package. The modules are organized by responsibility so the same building blocks can be reused across search, design, docking, and scoring workflows.

## Package overview

- `search`: High-level search workflows, including `PeptidomeSearch` and `ProteomeSearch`, checkpointing, candidate selection, and search utilities.
- `bayes`: Bayesian optimization utilities, mainly acquisition functions used to prioritize which sequences to evaluate next.
- `surrogate_model`: Train and manage predictive models that estimate objective values from sequence embeddings.
- `embedding`: Convert sequences into numeric representations, including ESM- and AAIndex-based embeddings and dimensionality reduction helpers.
- `docking`: Interfaces for structure-conditioned docking with supported backends such as AlphaFold and Boltz.
- `folding`: Monomer folding utilities for unconditional or sequence-level structure prediction workflows.
- `scoring`: Score docked complexes or monomer structures using confidence metrics, Rosetta-based terms, geometry checks, and objective aggregation helpers.
- `diffusion`: Generative design components, including the BoRF pipeline and helpers around diffusion and MPNN/FastRelax-based refinement.
- `genetic_algorithm`: Evolutionary search components used by BoGA, including sequence generation and mutation logic.
- `structure`: Shared parsers and caching utilities for working with PDB/CIF structures and extracting chain- or residue-level information.
- `config`: Configuration loading and default YAML presets for the main workflows.
- `logging`: Logging helpers used to track optimization runs and output artifacts.

## Common entry points

The top-level package re-exports several commonly used classes and functions, including:

- `PeptidomeSearch` and `ProteomeSearch` for Bayesian optimization over sequence collections.
- `Docker` and `AlphaFoldMonomer` for structure generation workflows.
- `ComplexScorer` and `MonomerScorer` for evaluating predicted structures.
- `Embedder` for sequence featurization.
- `BoRF` for diffusion-driven design workflows.
- `StructureParser` and `parse_structure` for structural data access.

For usage examples, see the repository-level documentation and the examples in `/examples`.
