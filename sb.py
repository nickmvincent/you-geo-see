#%%
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.collections import PatchCollection

# # Create list for all the error patches
# boxes = []

# # Loop over data points; create box from errors at each point
# rect = Rectangle(xy=(0.75, 0.75), width=0.5, height=0.5)
# boxes.append(rect)

# # Create patch collection with specified colour/alpha
# pc = PatchCollection(boxes, facecolor='w', alpha=0.8,
#                         edgecolor=None)

# plt.scatter([0,1,2], [0,1,2])
# ax = plt.gca()
# ax.add_collection(pc)

df = pd.read_csv('importance_df.csv')
df.head()

#%%
full_appears = df[
    (df.metric == 'domain_appears') & (df.subset == 'full')
]
len(full_appears)

#%%
set(full_appears.domain.value_counts())

#%%
top3_appears = df[
    (df.metric == 'domain_appears') & (df.subset == 'top_three')
]
len(top3_appears)

#%%
set(top3_appears.domain.value_counts())

#%%
