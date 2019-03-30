import pandas as pd
import numpy as np

from collections import defaultdict

def main():
    df = pd.read_csv('importance_df.csv')
    df = df[df.link_type == 'results_and_knowledge_panel']
    df = df.fillna(0)
    print(df.head())

    cat_to_rate = defaultdict(list)
    for category in list(df['category'].drop_duplicates()):
        print('category', category)
        filt_cat = df[df.category == category]
        for domain in list(filt_cat['domain'].drop_duplicates()):
                filt_domain = filt_cat[filt_cat.domain == domain]
                for subset in list(filt_domain.subset.drop_duplicates()):
                    filt_subset = filt_domain[filt_domain['subset'] == subset]
                    for metric in list(filt_subset.metric.drop_duplicates()):
                        filt_metric = filt_subset[filt_subset.metric == metric]
                        if metric == 'domain_rank' or metric == 'domain_count':
                            filt_metric = filt_metric[filt_metric['val'] != 0]
                        val = np.mean(filt_metric['val'])
                        if domain == 'wikipedia.org':
                            print(subset, metric, round(val, 2))
                        #if metric == 'domain_appears' and subset == 'top_three' and val > 0.1:
                        if metric == 'domain_appears' and subset == 'full' and val > 0.1:

                            cat_to_rate[category].append({
                                'domain':domain,
                                'val': val
                            })
    for key, d in cat_to_rate.items():
        newlist = sorted(d, key=lambda k: k['val'], reverse=True)
        newlist = [x for x in newlist if x['domain'] not in ['NewsCarousel', 'MapsLocations', 'people also ask', 'UserTweetCarousel']]
        for i in range(len(newlist)):
            newlist[i]['rank'] = i+1
        print('Category', key)
        print(newlist)
if __name__ == '__main__':
    main()