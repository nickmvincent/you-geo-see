import pandas as pd
from plotters_jan17_2by4 import plot_importance
import matplotlib.pyplot as plt


def main():
    importance_df = pd.read_csv('importance_df.csv')
    plot_importance(importance_df)
    plt.show()

main()