import pandas as pd
from plotters import plot_importance
import matplotlib.pyplot as plt


def main():
    importance_df = pd.read_csv('importance_df.csv')
    plot_importance(importance_df)
    plt.tight_layout()
    plt.show()

main()