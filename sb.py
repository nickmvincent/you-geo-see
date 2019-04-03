#%%
import pandas as pd
import matplotlib.pyplot as plt


#%%
comp_df = pd.read_csv('all_tests_together.csv')

#%%
comp_df[(comp_df.column.str.contains('yelp'))]
#%%


# what's the mean value of difference in all Wikipedia tests
x = comp_df[
    (comp_df.column.str.contains('wikipedia')) & 
    (comp_df.category != 'all') & 
    (comp_df.column.str.contains('results_full'))   
]
x
x.add_inc.describe()

#%%
ugc_mask = (comp_df.column.str.contains('wikipedia'))

for phrase in [
    'wikipedia', 'yelp', 'tripadvisor', 'facebook',
    'twitter', 'instagram', 'youtube', 'linkedin',
]:
    ugc_mask = ugc_mask | (comp_df.column.str.contains(phrase))

#%%
ugc = comp_df[ugc_mask]
ugc[ugc.fisher_pval < 0.05]
len(ugc[ugc.fisher_pval < 0.05])

#%%
comp_df[ugc_mask].add_inc.describe()
# comp_df[ugc_mask].category.value_counts()
# comp_df[ugc_mask]

#%%
comp_df[ugc_mask & (comp_df.category == 'popular')].comparison.value_counts()


#%%
y = comp_df[
    (comp_df.column.str.contains('wikipedia')) & 
    (comp_df.category != 'all') & 
    (comp_df.column.str.contains('results_full'))   
]
y.add_inc.describe()

#%%
y = comp_df[
    (comp_df.column.str.contains('wikipedia')) & 
    (comp_df.category != 'all') & 
    (comp_df.column.str.contains('results_top_three'))   
]
y.add_inc.describe()

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
