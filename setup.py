from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bopep",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    author="Erik Hartman",
    author_email="erik.hartman@hotmail.com",
    description="Bayesian Optimization for peptide docking",
)