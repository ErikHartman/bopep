from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bopep",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "pyrosetta": [
            "pyrosetta @ https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python313.ubuntu.wheel/pyrosetta-2025.25+release.a0cefad01b-cp313-cp313-linux_x86_64.whl#sha256=0ec84ac865336ca3266eb8a6e8b1c5908dbce661db21f173bbed36007e3f8408",
            "pyrosetta-installer==0.1.2"
        ],
    },
    author="Erik Hartman",
    author_email="erik.hartman@hotmail.com",
    description="Bayesian Optimization for peptide docking",
    package_data={
        "bopep.embedding": ["aaindex1.csv"],
    },
)