# Bayesian Optimization for identifying endogenous peptide binders

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErikHartman/bopep/blob/main/bopep.ipynb)

This repository contains the code for `bopep`, a tool for identifying peptide binders to proteins from large scale peptidomic data.

## Running in Google Colab
I have created a [notebook in Google Colab](https://colab.research.google.com/github/ErikHartman/bopep/blob/main/bopep.ipynb) which allows you to run the `bopep` workflow without installing anything locally.

For the non-premium users of Google Colab, you are limited to one T4 GPU. For large datasets, the pipeline will take some time.

## Installation for running locally

To run `bopep` locally, you will need to clone this repository, install **LocalColabFold** and **PyRosetta** as well as other dependencies using pip. Follow the steps below to set up your environment:

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/ErikHartman/bopep.git
cd bopep
```

### Step 2: Set Up a Virtual Environment

It’s recommended to set up a virtual environment to keep dependencies isolated:

```bash
python -m venv bopep_env # Or python3
source bopep_env/bin/activate  # On Windows, use `bopep_env\Scripts\activate`
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
   python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
   ```


3. **Install Remaining Dependencies**:
   Finally, install any additional dependencies required for `bopep`:

   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Running the Optimization Process

After the setup is complete, you can run the Bayesian optimization process as described below. Make sure the virtual environment is activated.

ADD CODE EXAMPLES.

## Credits
Credits go out to those who have created great packages and tools such as LocalColabFold, PyRosetta, torch, optuna and other modules.

Cite these modules:
