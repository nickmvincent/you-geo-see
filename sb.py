#%%
import pandas as pd
import matplotlib.pyplot as plt


#%%
comp_df = pd.read_csv('comparison_df.csv')

#%%
# what's the mean value of difference in all Wikipedia tests
x = comp_df[
    (comp_df.column.str.contains('wikipedia')) & 
    (comp_df.category != 'all') & 
    (comp_df.column.str.contains('results_full'))   
]
x
x.mean()

#%%

df = pd.read_csv('importance_df.csv')
df.head()

#%%
full_appears = df[
    (df.metric == 'domain_maps') & (df.subset == 'full')
]
len(full_appears)

#%%
set(full_appears.domain.value_counts())

#%%

#%%
top3_appears = df[
    (df.metric == 'domain_appears') & (df.subset == 'top_three')
]
len(top3_appears)

#%%
set(top3_appears.domain.value_counts())

#%%
df.link_type.value_counts()
len(df)

#%%
