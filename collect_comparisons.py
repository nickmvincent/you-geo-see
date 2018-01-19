"""
This script runs through the "output folder"
to find all relevant comparisons

Then it formats these into a nice CSV for publication!
"""

import argparse

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

    if  args.output_dirs:
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
    count = 0
    for path in paths:
        print(path)
        pre, category = path.split('__')
        comparison = pre.split('_')[0][7:]

        print(comparison, category)
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
            row_dict['increase'] = row.add_inc
            row_dict['winner'] = row.winner
            row_dict['subset'] = FULL
            row_dict['category'] = category
            row_dicts.append(row_dict)
            if row_dict['domain'] in UGC_WHITELIST:
                count +=1 
        for _, row in top3_df.iterrows():
            row_dict = {}
            row_dict['domain'] = strip_top3(row['column'])
            row_dict['increase'] = row.add_inc
            row_dict['winner'] = row.winner
            row_dict['subset'] = TOP_THREE
            row_dict['category'] = category
            row_dicts.append(row_dict)

            if row_dict['domain'] in UGC_WHITELIST:
                count +=1 
    outdf = pd.DataFrame(row_dicts)
    print(outdf)
    outdf.to_csv('collect_comparisons.csv')

    print(count)


if __name__ == '__main__':
    parse()
