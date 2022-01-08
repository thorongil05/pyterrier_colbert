from typing import List
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum

class PruningMeasure(Enum):
    PERCENTAGE = 0
    INDEX_REDUCTION = 1
    AVERAGE_REDUCTION = 2

class StaticPruningVisualization:

    def __init__(self, pruning_measure : PruningMeasure = PruningMeasure.INDEX_REDUCTION, colors : List[str] = None):
        if colors:
            self.colors = colors
        else:
            self.colors = ['green', 'red', 'blue', 'orange', 'grey', 'brown']
        self.names = []
        self.dataframes : List(pd.DataFrame) = []
        self.font_size = 18
        self.fig_size = (10,8)
        self.pruning_measure : PruningMeasure = pruning_measure
        if pruning_measure == PruningMeasure.INDEX_REDUCTION:
            self.inf_x_limit = 1
            self.sup_x_limit = 10 # configurable
        if pruning_measure == PruningMeasure.PERCENTAGE:
            self.inf_x_limit = 0
            self.sup_x_limit = 100
        if pruning_measure == PruningMeasure.AVERAGE_REDUCTION:
            self.inf_x_limit = 0
            self.sup_x_limit = 1

    def set_data(self, dataframes : List[pd.DataFrame], names : List[str]):
        self.names = names
        self.dataframes = dataframes

    def plot(self, x_value='reduction', y_value : str = 'nDCG@10', label_column='short-name', marker='o'):
        assert len(self.dataframes) == len(self.names) and len(self.dataframes) != 0, 'names and dataframes must be of the same length'
        colors = self.colors[:len(self.dataframes)]
        plt.ioff()
        fig, ax = plt.subplots(figsize=self.fig_size)
        fig.suptitle(y_value + ' - comparison', fontsize=30)
        ax.set_xlim([self.inf_x_limit, self.sup_x_limit])
        for dataframe, name, color in zip(self.dataframes, self.names, colors):
            dataframe = dataframe.sort_values(by=x_value)
            ax.plot(dataframe[x_value], dataframe[y_value], marker=marker, label=name, color=color)
            if label_column and label_column in dataframe.columns: 
                self._plot_labels(ax, dataframe, label_column, x_value, y_value)
        ax.set_ylim([0, 1])
        ax.set_xlabel(x_value, fontsize=self.font_size)
        ax.set_ylabel(y_value, fontsize=self.font_size)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return fig, ax


    def _plot_labels(self, ax, dataframe: pd.DataFrame, label_column, x_value, y_value):
        for i, text in enumerate(dataframe[label_column]):
            if dataframe.iloc[i][x_value] <= self.sup_x_limit:
                ax.text(dataframe.iloc[i][x_value] + .01, dataframe.iloc[i][y_value] + .01, text, fontsize=9)