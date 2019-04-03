"""
This script runs through the "output folder"
to find all relevant comparisons

Then it formats these into a nice CSV for publication!
"""

import argparse
import json

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from data_helpers import strip_domain_strings_wrapper
from constants import FULL, TOP_THREE
from analysis import UGC_WHITELIST


sns.set(style="whitegrid", palette='muted', color_codes=True)

def parse():
    """
    Collects comparisons
    """
    parser = argparse.ArgumentParser(description='see file.')
    parser.add_argument(
        '--output_dirs', help='Where are the outputs', nargs='+', required=False)
    parser.add_argument(
        '--criteria', help='how are we identifying the significant comparisons', default='ttest')
    parser.set_defaults(print_all=False)

    args = parser.parse_args()

    if args.output_dirs:
        paths = args.output_dirs
    else:
        comparisons_and_queries = [
            ('urban-rural', 'all',),
            ('income', 'all'),
            ('voting', 'all'),
            ('2018-01-16_income', 'extra'),
            ('2018-01-17_voting', 'extra'),
            ('2018-01-17_urban-rural', 'extra'),
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
    all_tests_together = None
    ugc_row_dicts = []
    n_ugc_diffs = 0
    n_tests = 0
    n_ugc_diffs_strict = 0


    sub_to_domain = {
        FULL: {},
        TOP_THREE: {}
    }
    for path in paths:
        pre, category = path.split('__')
        comparison = pre.split('__')[0][7:]

        print(path, '|', comparison, '|', category)
        if args.criteria == 'ttest':
            full = path + '/full_pval_summary.csv'
            top3 = path + '/top_three_pval_summary.csv'
            
        elif args.criteria == 'fisher':
            full = path + '/full_fisher_summary.csv'
            top3 = path + '/top_three_fisher_summary.csv'
        
        all_tests = path + '/comparisons.csv'

        try:
            full_df = pd.read_csv(full)
            top3_df = pd.read_csv(top3)
            all_tests_df = pd.read_csv(all_tests)
        except:
            print('Not found: {}'.format(full))
            continue
        n_tests += len(all_tests_df.index)


        all_tests_df['category'] = category
        all_tests_df['comparison'] = comparison
        if all_tests_together is None:
            all_tests_together = all_tests_df
        else:
            all_tests_together = pd.concat([
                all_tests_together, all_tests_df
            ])
        

        strip_full = strip_domain_strings_wrapper(FULL)
        strip_top3 = strip_domain_strings_wrapper(TOP_THREE)

        for (df, subset, strip_func) in [
            (full_df, FULL, strip_full),
            (top3_df, TOP_THREE, strip_top3)
        ]:
            for _, row in df.iterrows():
                row_dict = {}
                domain = strip_func(row['column'])

                
                row_dict['domain'] = domain
                row_dict['increase'] = round(row.add_inc, 2)
                row_dict['winner'] = row.winner
                row_dict['subset'] = subset
                row_dict['category'] = category
                row_dict['fisher_pval'] = row.fisher_pval
                row_dicts.append(row_dict)
                # row_dict['ratio'] = row.mult_inc
                should_plot = (
                    domain  == 'wikipedia.org' or
                    domain == 'MapsLocations' or
                    ':' in domain # domain was coded
                )
                if should_plot:
                    code_index = domain.find(':')
                    if code_index != -1:
                        code = domain[domain.find(':')+1:]
                        print(code)
                        if code[:2] != 'tt':
                            print('Skipping code {} b/c not UGC'.format(code))
                            continue
                        else:
                            row_dict['domain'] = domain = domain.replace(':'+code, '')
                    if domain != 'MapsLocations':
                        n_ugc_diffs += 1
                        if row['fisher_pval'] < 0.05 / 115:
                            n_ugc_diffs_strict += 1
                            print(domain)
                    ugc_row_dicts.append(row_dict)
                    if domain not in sub_to_domain[subset]:
                        sub_to_domain[subset][domain] = {
                            'categories': [],
                            'locations': [],
                            'category_in_location': [],
                            'max': 0,
                        }

                    if round(row.add_inc, 2) > sub_to_domain[subset][domain]['max']:
                        sub_to_domain[subset][domain]['max'] = round(row.add_inc, 2)
                    if category not in sub_to_domain[subset][domain]['categories']:
                        sub_to_domain[subset][domain]['categories'].append(category) 
                    if row.winner not in sub_to_domain[subset][domain]['locations']:
                        sub_to_domain[subset][domain]['locations'].append(row.winner)
                    tmp = category + ' in ' + row.winner
                    if tmp not in sub_to_domain[subset][domain]['category_in_location']:
                        sub_to_domain[subset][domain]['category_in_location'].append(tmp)
    outdf = pd.DataFrame(row_dicts)
    #print(outdf)
    outdf.to_csv('collect_comparisons.csv')
    all_tests_together.to_csv('all_tests_together.csv')
    ugc_outdf = pd.DataFrame(ugc_row_dicts)
    
    for subset, domain_to_vals in sub_to_domain.items():
        for domain, vals in domain_to_vals.items():
            vals['categories'] = ', '.join(vals['categories'])
            vals['locations'] = ', '.join(vals['locations'])
            vals['category_in_location'] = ', '.join(vals['category_in_location'])
    max_full_df, max_top3_df = pd.DataFrame(sub_to_domain[FULL]).transpose(), pd.DataFrame(sub_to_domain[TOP_THREE]).transpose()
    max_full_df.sort_values(['category_in_location']).to_csv('max_full_comparisons.csv')
    max_top3_df.sort_values(['category_in_location']).to_csv('max_top3_comparisons.csv')

    domains_plus_winners = [
        str(x) + ' (' + str(y) + ')' for x, y in zip(
            list(ugc_outdf.domain),
            list(ugc_outdf.winner)
        )
    ]
    ugc_outdf = ugc_outdf.assign(domains_plus_winners=domains_plus_winners)

    sorted_ugc_outdf = ugc_outdf.sort_values(['domain', 'category', 'winner'])

    # visualize it
    ugc_outdf.loc[ugc_outdf.category == 'top_insurance', 'category'] = 'insurance'
    ugc_outdf.loc[ugc_outdf.category == 'top_loans', 'category'] = 'loans'

    sorted_ugc_outdf[['category', 'domain', 'winner', 'subset', 'increase', 'fisher_pval']].to_csv('{}_comparisons.csv'.format(args.criteria), index=False)
    with open('{}_comparison_deets.json'.format(args.criteria), 'w') as f:
        json.dump({
            'n_ugc_diffs': n_ugc_diffs,
            'n_tests': n_tests,
            'n_ugc_diffs_strict': n_ugc_diffs_strict,
        }, f)
        
    row1_df = ugc_outdf[(ugc_outdf.winner == 'urban') | (ugc_outdf.winner == 'DEM') | (ugc_outdf.winner == 'high-income')]
    order1 = sorted(row1_df.domains_plus_winners.drop_duplicates())
    row2_df = ugc_outdf[(ugc_outdf.winner == 'rural') | (ugc_outdf.winner == 'GOP') | (ugc_outdf.winner == 'low-income')]
    order2 = sorted(row2_df.domains_plus_winners.drop_duplicates())

    # ugc_outdf.loc[ugc_outdf.winner == 'high-income', 'winner'] = 'top SES'
    # ugc_outdf.loc[ugc_outdf.winner == 'low-income', 'winner'] = 'bottom SES'
    # ugc_outdf.loc[ugc_outdf.winner == 'GOP', 'winner'] = 'top GOP'
    # ugc_outdf.loc[ugc_outdf.winner == 'DEM', 'winner'] = 'top DEM'
    orders = [order1, order2]

    sns.set_context("paper", rc={"font.size":10, "font.family": "Times New Roman", "axes.titlesize":10,"axes.labelsize":10})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.5), gridspec_kw = {'height_ratios':[8, 3]}, dpi=300)

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
        for colnum in [0, 1]:
            ax = axes[rownum, colnum]
            sns.despine(ax=ax, bottom=True, left=True)
            # ax.set(xlabel='Increase in incidence rate')
            ax.set(xlabel='')
            ax.hlines([x + 0.5 for x in range(len(orders[rownum]))], *ax.get_xlim(), linestyle='--', color='lightgray')
    title_template = 'Increases in incidence rate\nIn {locations_str}\nConsidering {subset_str}'
    axes[0, 0].set_xlabel(
        title_template.format(**{
            'subset_str': 'full results',
            'locations_str': 'urban/high-income/DEM areas',
        })
    )
    axes[0, 1].set_xlabel(
        title_template.format(**{
            'subset_str': 'top three results',
            'locations_str': 'urban/high-income/DEM areas',
        })
    )
    axes[1, 0].set_xlabel(
        title_template.format(**{
            'subset_str': 'full results',
            'locations_str': 'rural/low-income/GOP areas',
        })
    )
    axes[1, 1].set_xlabel(
        title_template.format(**{
            'subset_str': 'top three results',
            'locations_str': 'rural/low-income/GOP areas',
        })
    )
   
    plt.tight_layout()
    fig.savefig('comparisons.png')
    plt.show()


if __name__ == '__main__':
    parse()
