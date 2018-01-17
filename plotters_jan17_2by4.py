"""
Does plots for paper
"""
import matplotlib.pyplot as plt
import seaborn as sns
from constants import FULL, TOP_THREE

sns.set_context("paper", rc={"font.size":10, "font.family": "Times New Roman", "axes.titlesize":10,"axes.labelsize":10})
sns.set(style="whitegrid", palette='pastel', color_codes=True)

ALL_SUBSET_STRING = 'considering all results'
TOP_THREE_SUBSET_STRING = 'considering only top three results'

AGGREGATE_STRING = 'aggregated'
QUERYSET_BREAKDOWN_STRING = 'broken down by query set'


def strip_domain_strings_wrapper(subset):
    """wraps a function to strip results_type, subset, and domain_frac text"""

    def strip_domain_strings(text):
        """Strip the string"""
        text = text.strip('*')
        text = text.replace('results_', '')
        text = text.replace(subset, '')
        text = text.replace('domain_frac_', '')
        text = text.strip('_')
        return text
    return strip_domain_strings


def plot_importance(df):
    """
    Plot the importance of each domain.

    """
    df = df.fillna(0)
    _, axes = plt.subplots(nrows=2, ncols=4)
    title_template = '{metric}\n {subset}, {type}'
    # color palettes that will be overlayed.
    pal = 'pastel'
    pal_lower = 'muted'

    # placeholder for qual coded values
    df['fake_val'] = df['val'].map(lambda x: x / 2)
    aggregate = df[df.category == 'all']
    categorized = df[df.category != 'all']
    big_order = None
    ugc_order = None
    for colnum, subdf in enumerate([aggregate, aggregate, categorized, categorized]):
        # adding caching to save a second or two?
        # risky code. needs to be double checked
        if colnum == 0:
            subdf = subdf[subdf.is_big_col == True]
            if big_order is None:
                tmpdf = subdf[
                    (subdf.subset == FULL) & (subdf.metric == 'domain_appears')
                ][['domain', 'val']]
                big_order = list(
                    tmpdf.groupby('domain').mean().sort_values(
                        'val', ascending=False).index)[:10]
            order = big_order
        else:
            subdf = subdf[subdf.is_ugc_col == True]
            if ugc_order is None:
                tmpdf = subdf[
                    (subdf.subset == FULL) & (subdf.metric == 'domain_appears')
                ][['domain', 'val']]
                ugc_order = list(
                    tmpdf.groupby('domain').mean().sort_values(
                        'val', ascending=False).index)
            order = ugc_order
        title_kwargs = {}
        for subset in [FULL, TOP_THREE]:
            subdf.loc[:, 'domain'] = subdf['domain'].apply(strip_domain_strings_wrapper(subset))
        if colnum in [0, 1, 2]:
            mask1 = (subdf.metric == 'domain_appears') & (subdf.subset == FULL)
            mask2 = (subdf.metric == 'domain_appears') & (subdf.subset == TOP_THREE)
            sns.barplot(x='val', y='domain', hue='category', order=order,
                        data=subdf[mask1], ax=axes[0, colnum], ci=None, palette=pal)
            sns.barplot(x='fake_val', y='domain', hue='category', order=order,
                        data=subdf[mask1], ax=axes[0, colnum], ci=None, palette=pal_lower)
            sns.barplot(x='val', y='domain', hue='category', order=order,
                        data=subdf[mask2], ax=axes[1, colnum], ci=None, palette=pal)

            sns.barplot(x='fake_val', y='domain', hue='category', order=order,
                        data=subdf[mask2], ax=axes[1, colnum], ci=None, palette=pal_lower)
            if colnum == 0:
                title_kwargs['metric'] = 'Fraction of pages where domain appears, Top 10 domains, '
            else:
                title_kwargs['metric'] = 'Fraction of pages where domain appears, UGC Platforms, '
        else:
            mask1 = (subdf.metric == 'domain_frac') & (subdf.subset == FULL)
            mask2 = (subdf.metric == 'domain_frac') & (subdf.subset == TOP_THREE)
            sns.barplot(x='val', y='domain', hue='category', order=order,
                        data=subdf[mask1], ax=axes[0, colnum], ci=None, palette=pal)
            sns.barplot(x='fake_val', y='domain', hue='category', order=order,
                        data=subdf[mask1], ax=axes[0, colnum], ci=None, palette=pal_lower)
            sns.barplot(x='val', y='domain', hue='category', order=order,
                        data=subdf[mask2], ax=axes[1, colnum], ci=None, palette=pal)
            sns.barplot(x='fake_val', y='domain', hue='category', order=order,
                        data=subdf[mask2], ax=axes[1, colnum], ci=None, palette=pal_lower)
            title_kwargs['metric'] = 'Fraction of results domain comprises'
        for rownum in [0, 1]:
            ax = axes[rownum, colnum]
            if colnum in [0, 1, 2]: # appears
                ax.set_xlim([0, 1])
                ax.set_xlabel('fraction of pages')
            else:
                # ax.set_xlim([0, 0.2])
                ax.set_xlabel('fraction of results')
            if rownum == 0:
                # ax.set(xticklabels=[], xlabel='')
                title_kwargs['subset'] = ALL_SUBSET_STRING
            elif rownum == 1:
                title_kwargs['subset'] = TOP_THREE_SUBSET_STRING

            if colnum in [2, 3]:
                title_kwargs['type'] = QUERYSET_BREAKDOWN_STRING
            else:
                title_kwargs['type'] = AGGREGATE_STRING
            #ax.legend().set_visible(False)
            ax.legend(ncol=2, loc='lower right')
            sns.despine(ax=ax, bottom=True, left=True)
            if colnum != 0:
                ax.set_ylabel('')
            ax.set_title(title_template.format(**title_kwargs))


def plot_comparison(df, groups=['urban', 'rural']):
    """
    Plot a single comparison
    urban-rural
    income
    voting

    """
    _, comparison_axes = plt.subplots(nrows=2, ncols=2)
    title_template = 'Differences in Domain Fractions, {subset}, {type}: {left} (left) vs {right} (right)'
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
                ax.set_xlabel('Difference in domain fraction')
                ax.set_ylabel('Domain')
                ax.axvline(0, color='black')
                sns.despine(ax=ax, bottom=True, left=False)
