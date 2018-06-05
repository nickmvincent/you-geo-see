"""
Does plots for paper
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from constants import FULL, TOP_THREE
from data_helpers import strip_domain_strings_wrapper

sns.set(style="whitegrid", palette='colorblind', color_codes=True)

ALL_SUBSET_STRING = 'considering all results'
TOP_THREE_SUBSET_STRING = 'considering only top three results'

AGGREGATE_STRING = 'aggregated'
QUERYSET_BREAKDOWN_STRING = 'broken down by query set'


def plot_importance(df):
    """
    Plot the importance of each domain.

    """
    #TODO: parametrize
    df = df[df.link_type == 'news']

    counts_copy = df[(df.metric == 'domain_count') & (df.category != 'all')]
    df = df.fillna(0)
    df.loc[df.category == 'top_insurance', 'category'] = 'insurance'
    df.loc[df.category == 'top_loans', 'category'] = 'loans'
    df.loc[df.category == 'med_sample_first_20', 'category'] = 'medical'
    df.loc[df.category == 'procon_popular', 'category'] = 'controversial'
    df.loc[df.domain == 'people also ask', 'domain'] = 'PeopleAlsoAsk'
    title_template = '{metric}\n {subset}, {type}'
    # color palettes that will be overlayed.
    pal = 'colorblind'

    # placeholder for qual coded values
    # df['fake_val'] = df['val'].map(lambda x: x / 2)
    categorized = df[df.category != 'all']
    print('df head', df.head())
    plot_col_dfs = [categorized]
    sns.set_context("paper", rc={
        "font.size":10, "font.family": "Times New Roman", 
        "axes.titlesize":10,"axes.labelsize":10})

    width, height = 5, 4.5
    fig, axes = plt.subplots(ncols=2, nrows=len(plot_col_dfs), figsize=(width, height), dpi=300)

    # this is currently wrong! There are two columns and only one row.
    # this seems terribly confusing...
    # is there an easy fix?
    for colnum, subdf in enumerate(plot_col_dfs):
        if subdf.empty:
            print('empty')
            continue
        tmpdf = subdf[
            (subdf.subset == FULL) & (subdf.metric == 'domain_appears')
        ][['domain', 'val', 'is_ugc_col']]
        grouped_and_sorted = tmpdf.groupby('domain').mean().sort_values(
                'val', ascending=False)
        order = list(grouped_and_sorted.index)
        ugc_cols = list(grouped_and_sorted[grouped_and_sorted.is_ugc_col == True].index)
        print('ugc_cols', ugc_cols)
        nonugc_count = 0
        selected_order = []
        for domain in order:
            if domain in ugc_cols:
                selected_order.append(domain)
            else:
                nonugc_count += 1
                if nonugc_count <= 20: #TODO
                    selected_order.append(domain)
        
        # for domain in ugc_cols:
        #     if domain not in order:
        #         order.append(domain)
        ranks_and_counts = []
        for domain in selected_order:
            ranks = subdf[(subdf.metric == 'domain_rank') & (subdf.domain == domain)]
            ranks = ranks[ranks.val != 0].val
            print('rank', domain, ranks.mean())
            counts = counts_copy[(subdf.domain == domain)].val
            print('count', domain, counts.mean())
            ranks_and_counts.append({
                'domain': domain,
                'average rank': round(ranks.mean(), 1),
                'average count': round(counts.mean(), 1)
            })
            # _, histax = plt.subplots()
            # sns.distplot(
            #     ranks.dropna(), rug=True, bins=list(range(1, 13)), 
            #     kde=False, color="b", ax=histax)
            # histax.set_title('Histogram for {}'.format(domain))
        ranks_and_counts_df = pd.DataFrame(ranks_and_counts)
        ranks_and_counts_df.to_csv('ranks_and_counts.csv')
        title_kwargs = {}
        # for subset in [FULL, TOP_THREE]:
        #     subdf.loc[:, 'domain'] = subdf['domain'].apply(strip_domain_strings_wrapper(subset))
        print(subdf)
        if colnum in [0]:
            mask1 = (subdf.metric == 'domain_appears') & (subdf.subset == FULL)
            mask2 = (subdf.metric == 'domain_appears') & (subdf.subset == TOP_THREE)
            sns.barplot(x='val', y='domain', hue='category', order=selected_order,
                        data=subdf[mask1], ax=axes[0], ci=None, palette=pal)
            # sns.barplot(x='fake_val', y='domain', hue='category', order=order,
            #             data=subdf[mask1], ax=axes[0], ci=None, palette=pal_lower)

            sns.barplot(x='val', y='domain', hue='category', order=selected_order,
                        data=subdf[mask2], ax=axes[1], ci=None, palette=pal)

            # sns.barplot(x='fake_val', y='domain', hue='category', order=order,
            #             data=subdf[mask2], ax=axes[1], ci=None, palette=pal_lower)
            title_kwargs['metric'] = 'Fraction of pages where domain appears'

        # might be swapped.
        for rownum in [0, 1]:
            ax = axes[rownum]
            ax.set_xlim([0, 1])
            ax.legend(loc='lower right', frameon=True)
            
            if rownum == 0:
                title_kwargs['subset'] = ALL_SUBSET_STRING
                ax.set_xlabel('Full-page incidence rate', fontname = "Times New Roman")
                ax.legend().set_visible(False)
            elif rownum == 1:
                title_kwargs['subset'] = TOP_THREE_SUBSET_STRING
                ax.set_xlabel('Top-three incidence rate', fontname = "Times New Roman")
                ax.set(yticklabels=[], ylabel='')
                num_rows = len(selected_order)
                the_table = plt.table(cellText=ranks_and_counts_df.as_matrix(columns=['average rank', 'average count']),
                    bbox=(1.2,0,1,1))
                the_labels = plt.table(cellText=[['average\nrank', 'average\ncount']],
                    bbox=(1.2,-1.1/num_rows,1,1/num_rows))
                for key, cell in the_table.get_celld().items():
                    # cell.set_linewidth(0)
                    cell.set_linestyle('--')
                    cell.set_linewidth(1.5)
                    cell.set_edgecolor('lightgray')
                for key, cell in the_labels.get_celld().items():
                    cell.set_linewidth(0)
                ax.add_table(the_table)
                ax.add_table(the_labels)

            title_kwargs['type'] = QUERYSET_BREAKDOWN_STRING
            sns.despine(ax=ax, bottom=True, left=True)
            ax.set_ylabel('')
            ax.hlines([x + 0.5 for x in range(len(order))], *ax.get_xlim(), linestyle='--', color='lightgray')
            # ax.set_title(title_template.format(**title_kwargs))
    fig.savefig('importance.png', bbox_inches='tight')



def plot_comparison(df, groups=['urban', 'rural']):
    """
    Plot a single comparison
    urban-rural
    income
    voting

    """
    _, comparison_axes = plt.subplots(nrows=2, ncols=2)
    title_template = 'Differences in Domain Appears, {subset}, {type}: {left} (left) vs {right} (right)'
    for colnum, is_aggregate in enumerate([True, False]):
        title_kwargs = {
            'left': groups[0],
            'right': groups[1]
        }
        if is_aggregate:
            col_mask = df.category == 'all'
            title_kwargs['type'] = AGGREGATE_STRING
        else:
            col_mask = df.category != 'all'
            title_kwargs['type'] = QUERYSET_BREAKDOWN_STRING
        coldf = df[col_mask]
        for rownum, subset in enumerate([FULL, TOP_THREE]):
            row_mask = coldf.subset == subset
            rowdf = coldf[row_mask]
            if subset == FULL:
                title_kwargs['subset'] = ALL_SUBSET_STRING
            elif subset == TOP_THREE:
                title_kwargs['subset'] = TOP_THREE_SUBSET_STRING

            def invert_add_inc(neg):
                """
                Returns a func to invert the add_inc column based on the argument passed in
                The argument passed will be the one inverted
                """
                def func(row):
                    """ invert column """
                    if row.winner == neg:
                        return row.add_inc * -1
                    return row.add_inc
                return func

            rowdf.loc[:, 'add_inc'] = rowdf.apply(invert_add_inc(groups[0]), axis=1)
            rowdf.loc[:, 'add_inc'] = rowdf.add_inc.astype(float)
            rowdf.loc[:, 'column'] = rowdf['column'].apply(strip_domain_strings_wrapper(subset))
            ax = comparison_axes[rownum, colnum]
            if not rowdf.empty:
                sns.barplot(
                    x='add_inc', y='column', hue='category',
                    data=rowdf,
                    ax=ax, ci=None)
                ax.set_title(title_template.format(**title_kwargs))
                ax.set_xlabel('Difference in domain appears')
                ax.set_ylabel('Domain')
                ax.axvline(0, color='black')
                sns.despine(ax=ax, bottom=True, left=False)
