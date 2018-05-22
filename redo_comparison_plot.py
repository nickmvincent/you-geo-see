import pandas as pd
from plotters import plot_comparison
import matplotlib.pyplot as plt


comparison_df = pd.read_csv('comparison_df.csv')
plot_comparison(comparison_df)
plt.show()
