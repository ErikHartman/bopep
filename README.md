# Bayesian Optimization for identifying endogenous peptide binders

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErikHartman/bopep/blob/main/bopep.ipynb)

This repository contains the code for `bopep`, a tool for identifying peptide binders to proteins from large scale peptidomic data.

## Running bopep in Google Colab
I have created a [notebook in Google Colab](https://colab.research.google.com/github/ErikHartman/bopep/blob/main/bopep.ipynb) which allows you to run the `bopep` workflow without installing anything locally.

For the non-premium users of Google Colab, you are limited to one T4 GPU. For large datasets, the pipeline will take some time.

Advanced users can adopt this repository for local usage. If there is enough demand I will work towards a local implementation.

## Credits
Credits go out to those who have created great packages and tools such as LocalColabFold, PyRosetta, torch, optuna and other modules.

Cite these modules:
