"""
This script runs through the "output folder"
to find all relevant comparisons

Then it formats these into a nice CSV for publication!
"""

import argparse

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from data_helpers import strip_domain_strings_wrapper
from constants import FULL, TOP_THREE
from analysis import UGC_WHITELIST




def parse():
    """
    todo
    """
    parser = argparse.ArgumentParser(description='see file.')
    parser.add_argument(
        '--output_dirs', help='Where are the outputs', nargs='+', required=False)
    parser.set_defaults(print_all=False)

    args = parser.parse_args()

    if args.output_dirs:
        paths = args.output_dirs
    else:
        comparisons_and_queries = [
            ('urban-rural', 'all',),
            ('income', 'all'),
            ('voting', 'all'),
            ('2018-01-16 14%3A04%3A08.388985_income', 'extra'),
            ('2018-01-17 04%3A43%3A51.097462_voting', 'extra'),
            ('2018-01-17 19%3A23%3A11.438607_urban-rural', 'extra'),
        ]
        categories = [
            'popular',
            'procon_popular',
            'trending',
            'top_insurance',
            'top_loans',
            'med_sample_first_20',
        ]
        paths = []

        for cq in comparisons_and_queries:
            for category in categories:
                path = 'output/{comparison}_10_{queries}.db__{category}'.format(**{
                    'comparison': cq[0],
                    'queries': cq[1],
                    'category': category,
                })
                paths.append(path)

    row_dicts = []
    ugc_row_dicts = []
    count = 0
    for path in paths:
        pre, category = path.split('__')
        comparison = pre.split('_')[0][7:]

        print(path, comparison, category)
        full = path + '/full_pval_summary.csv'
        top3 = path + '/top_three_pval_summary.csv'

        try:
            full_df = pd.read_csv(full)
            top3_df = pd.read_csv(top3)
        except:
            print('Not found: {}'.format(full))
            continue

        strip_full = strip_domain_strings_wrapper(FULL)
        strip_top3 = strip_domain_strings_wrapper(TOP_THREE)

        for _, row in full_df.iterrows():
            row_dict = {}
            row_dict['domain'] = strip_full(row['column'])
            row_dict['increase'] = round(row.add_inc, 2)
            row_dict['winner'] = row.winner
            row_dict['subset'] = FULL
            row_dict['category'] = category
            row_dicts.append(row_dict)
            # row_dict['ratio'] = row.mult_inc
            if row_dict['domain'] in UGC_WHITELIST:
                count += 1
            if row_dict['domain'] in UGC_WHITELIST + ['MapsLocations']:
                ugc_row_dicts.append(row_dict)
        for _, row in top3_df.iterrows():
            row_dict = {}
            row_dict['domain'] = strip_top3(row['column'])
            row_dict['increase'] = round(row.add_inc, 2)
            row_dict['winner'] = row.winner
            row_dict['subset'] = TOP_THREE
            row_dict['category'] = category
            # row_dict['ratio'] = row.mult_inc
            row_dicts.append(row_dict)
            if row_dict['domain'] in UGC_WHITELIST:
                count += 1
            if row_dict['domain'] in UGC_WHITELIST + ['MapsLocations']:
                ugc_row_dicts.append(row_dict)
    outdf = pd.DataFrame(row_dicts)
    print(outdf)
    outdf.to_csv('collect_comparisons.csv')
    ugc_outdf = pd.DataFrame(ugc_row_dicts)

    domains_plus_winners = [
        str(x) + '\n' + str(y) for x, y in zip(
            list(ugc_outdf.domain),
            list(ugc_outdf.winner)
        )
    ]
    ugc_outdf = ugc_outdf.assign(domains_plus_winners=domains_plus_winners)

    sorted_ugc_outdf = ugc_outdf.sort_values(['domain', 'category', 'winner'])
    sorted_ugc_outdf[['category', 'domain', 'winner', 'subset', 'increase']].to_csv('PUB_comparisons.csv', index=False)
    row1_df = ugc_outdf[(ugc_outdf.winner == 'urban') | (ugc_outdf.winner == 'DEM') | (ugc_outdf.winner == 'high-income')]
    order1 = sorted(row1_df.domains_plus_winners.drop_duplicates())
    row2_df = ugc_outdf[(ugc_outdf.winner == 'rural') | (ugc_outdf.winner == 'GOP') | (ugc_outdf.winner == 'low-income')]
    order2 = sorted(row2_df.domains_plus_winners.drop_duplicates())

    orders = [order1, order2]

    matplotlib.rcParams.update({
        'font.family': 'Times New Roman',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 8))

    for rownum, row_df in enumerate([row1_df, row2_df]):
        col1_df = row_df[row_df.subset == FULL]
        col2_df = row_df[row_df.subset == TOP_THREE]
        sns.barplot(x='increase', y='domains_plus_winners', hue='category', order=orders[rownum],
                            data=col1_df, ax=axes[rownum, 0], ci=None,)
        sns.barplot(x='increase', y='domains_plus_winners', hue='category', order=orders[rownum],
                            data=col2_df, ax=axes[rownum, 1], ci=None,)
        axes[rownum, 0].set(ylabel='', xlabel='')
        axes[rownum, 0].legend(loc='lower right', frameon=True)
        axes[rownum, 1].set(ylabel='', xlabel='', yticks=[])
        axes[rownum, 1].legend(loc='lower right', frameon=True)
        axes[rownum, 0].set(xlabel='Increase in domain appearance')
        axes[rownum, 1].set(xlabel='Increase in domain appearance')
    title_template = 'Domains that appear more, considering {subset_str}\nIn {locations_str}'
    axes[0, 0].set_title(
        title_template.format(**{
            'subset_str': 'full results',
            'locations_str': 'urban, higher-income, and Democratic-voting areas',
        })
    )
    axes[0, 1].set_title(
        title_template.format(**{
            'subset_str': 'top three results',
            'locations_str': 'urban, higher-income, and Democratic-voting areas',
        })
    )
    axes[1, 0].set_title(
        title_template.format(**{
            'subset_str': 'full results',
            'locations_str': 'rural, lower-income, and GOP-voting areas',
        })
    )
    axes[1, 1].set_title(
        title_template.format(**{
            'subset_str': 'top three results',
            'locations_str': 'rural, lower-income, and GOP-voting areas',
        })
    )
    plt.tight_layout()
    fig.savefig('comparisons.png')
    plt.show()


if __name__ == '__main__':
    parse()
