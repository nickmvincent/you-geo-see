import pandas as pd
from plotters import plot_importance
import matplotlib.pyplot as plt



importance_df = pd.read_csv('SAVEimportance_df.csv')
plot_importance(importance_df)
plt.show()
