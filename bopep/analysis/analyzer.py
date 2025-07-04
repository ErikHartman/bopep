import matplotlib.pyplot as plt
import pandas as pd
import os

class Analyzer:
    """
    Class to analyze BoPep results.

    Takes output directory as input and provides methods to visualize and analyze the results.

    Types of analysis include:
    - plot of objective over iterations
    - plot of r2 over iterations


    """
    def __init__(self):
        pass

    def plot_objective(self, objective_dataframe: pd.DataFrame):
        pass

    def plot_r2(self, model_dataframe: pd.DataFrame):
        pass

    def plot_hyperparameters(self, hyperparameter_dataframe: pd.DataFrame):
        pass

    def plot_umaps(self, embeddings: dict, objective_dataframe: pd.DataFrame):
        pass