# Synthesis Module

## Features

- **RFDiffusion**: Designs peptide backbones
- **ProteinMPNN & FastRelax**: Run sequence design on structural scaffolds and repeatedly relax and redesign 

## Installation Requirements

Before using this module, ensure you have:

1. **ProteinMPNN** installed and configured, can be installed from [ProteinMPNN GitHub](https://github.com/dauparas/ProteinMPNN)
2. **PyRosetta**  installed and configured, can be installed from [PyRosetta](https://www.pyrosetta.org/)
3. **RFDiffusion** design PDB files as input, can be installed from [RFdiffusion GitHub](https://github.com/RosettaCommons/RFdiffusion)
4. **Required Python packages**: pandas, biopython, python-dotenv

## Basic Usage

See examples file for a complete usage example. Each class in this module has its own usage example defined as a main function at the bottom of the file.