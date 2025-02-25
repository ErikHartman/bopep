# ⛏️ BoPep: Mining for peptide binders in large-scale data with Bayesian Optimization ⛏️

This repository contains the code for `BoPep`, a tool for identifying peptide binders to proteins from a large set of candidate peptides.

## Installation for running locally

To run `bopep` locally, you will need to clone this repository, install **LocalColabFold** and **PyRosetta** as well as other dependencies using pip. Follow the steps below to set up your environment:

### Step 1: Clone the Repository

First, clone the repository to your local machine (not available on pip yet):

```bash
git clone https://github.com/ErikHartman/bopep.git
cd bopep
```

### Step 2: Set Up a Virtual Environment

It’s recommended to set up a virtual environment to keep dependencies isolated:

```bash
python -m venv bopep_env # Or python3
source bopep_env/bin/activate
```

### Step 3: Install Dependencies

1. **Install LocalColabFold**: LocalColabFold is a fantastic package that allows you to run ColabFold locally. Follow the installation procedure [here](https://github.com/YoshitakaMo/localcolabfold) to install it.

Remember to export the `PATH` variable and make sure `colabfold_batch` is callable by running:
```bash
# For bash or zsh
# e.g. export PATH="/home/moriwaki/Desktop/localcolabfold/colabfold-conda/bin:$PATH"
export PATH="/path/to/your/localcolabfold/colabfold-conda/bin:$PATH"

colabfold_batch --help
```
This should work if you follow the instructions in the LocalColabFold git repo.


2. **Install PyRosetta**: PyRosetta is freely available for academic users. Any commercial usage requires the purchasing of a license.

   - Go to the [PyRosetta download page](https://www.pyrosetta.org/downloads) and read up on the terms for the license.
   - Install PyRosetta in your environment using the [pyrosetta-installer](https://pypi.org/project/pyrosetta-installer/) with pip: 
    ```bash
   pip install pyrosetta-installer
   ```
   - Then run:
   ```bash
   python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta(skip_if_installed=True)'
   ```


3. **Install Remaining Dependencies**:
   Finally, install any additional dependencies required for `bopep`:

   ```bash
   pip install -r requirements.txt
   ```


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
