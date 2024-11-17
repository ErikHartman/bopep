# BoPep: Bayesian Optimization for identifying endogenous peptide binders

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErikHartman/bopep/blob/main/bopep.ipynb)

This repository contains the code for `BoPep`, a tool for identifying peptide binders to proteins from large scale peptidomic data.

## Running BoPep in Google Colab
I have created a [notebook in Google Colab](https://colab.research.google.com/github/ErikHartman/BoPep/blob/main/BoPep.ipynb) which allows you to run the `BoPep` workflow without installing anything locally.

For the non-premium users of Google Colab, you are limited to one T4 GPU. For large datasets, the pipeline will take some time.

Advanced users can adopt this repository for local usage. If there is enough demand I will work towards a local implementation.

## Credits
Credits go out to those who have created great packages and tools such as LocalColabFold, PyRosetta, torch, optuna and other modules.

Cite these modules:
```
@article{Mirdita2022,
  title = {ColabFold: making protein folding accessible to all},
  volume = {19},
  ISSN = {1548-7105},
  url = {http://dx.doi.org/10.1038/s41592-022-01488-1},
  DOI = {10.1038/s41592-022-01488-1},
  number = {6},
  journal = {Nature Methods},
  publisher = {Springer Science and Business Media LLC},
  author = {Mirdita,  Milot and Sch\"{u}tze,  Konstantin and Moriwaki,  Yoshitaka and Heo,  Lim and Ovchinnikov,  Sergey and Steinegger,  Martin},
  year = {2022},
  month = may,
  pages = {679–682}
}

@article{Evans2021,
  title = {Protein complex prediction with AlphaFold-Multimer},
  url = {http://dx.doi.org/10.1101/2021.10.04.463034},
  DOI = {10.1101/2021.10.04.463034},
  publisher = {Cold Spring Harbor Laboratory},
  author = {Evans,  Richard and O’Neill,  Michael and Pritzel,  Alexander and Antropova,  Natasha and Senior,  Andrew and Green,  Tim and Žídek,  Augustin and Bates,  Russ and Blackwell,  Sam and Yim,  Jason and Ronneberger,  Olaf and Bodenstein,  Sebastian and Zielinski,  Michal and Bridgland,  Alex and Potapenko,  Anna and Cowie,  Andrew and Tunyasuvunakool,  Kathryn and Jain,  Rishub and Clancy,  Ellen and Kohli,  Pushmeet and Jumper,  John and Hassabis,  Demis},
  year = {2021},
  month = oct 
}

@article{Chaudhury2010,
  title = {PyRosetta: a script-based interface for implementing molecular modeling algorithms using Rosetta},
  volume = {26},
  ISSN = {1367-4803},
  url = {http://dx.doi.org/10.1093/bioinformatics/btq007},
  DOI = {10.1093/bioinformatics/btq007},
  number = {5},
  journal = {Bioinformatics},
  publisher = {Oxford University Press (OUP)},
  author = {Chaudhury,  Sidhartha and Lyskov,  Sergey and Gray,  Jeffrey J.},
  year = {2010},
  month = jan,
  pages = {689–691}
}

@misc{akiba2019optunanextgenerationhyperparameteroptimization,
      title={Optuna: A Next-generation Hyperparameter Optimization Framework}, 
      author={Takuya Akiba and Shotaro Sano and Toshihiko Yanase and Takeru Ohta and Masanori Koyama},
      year={2019},
      eprint={1907.10902},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1907.10902}, 
}

```
