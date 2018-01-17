"""
Does plots for paper
"""
import matplotlib.pyplot as plt
import seaborn as sns
from constants import FULL, TOP_THREE

sns.set_context(
    "paper",
    rc={"font.size": 8, "font.family": "Times New Roman", "axes.titlesize": 8, "axes.labelsize": 5})


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

    2 rows x 3 cols
    """
    _, axes = plt.subplots(nrows=2, ncols=3)
    tmpdf = df[
        (df.subset == FULL) & (df.category == 'all') & (df.metric == 'domain_frac')][['domain', 'val']]
    order = list(
        tmpdf.groupby('domain').mean().sort_values(
            'val', ascending=False).index)

    # color palettes that will be overlayed.
    pal = 'pastel'
    pal_lower = 'muted'

    # placeholder for qual coded values
    df['fake_val'] = df['val'].map(lambda x: x / 2)
    categorized = df[df.category != 'all']
    for x, subdf in enumerate([df[df.category == 'all'], categorized]):
        mask1 = (subdf.metric == 'domain_appears') & (subdf.subset == FULL)
        mask2 = (subdf.metric == 'domain_appears') & (subdf.subset == TOP_THREE)
        for subset in [FULL, TOP_THREE]:
            subdf.loc[:, 'domain'] = subdf['domain'].apply(strip_domain_strings_wrapper(subset))
        sns.barplot(x='val', y='domain', hue='category', order=order,
                    data=subdf[mask1], ax=axes[0,x], ci=None, palette=pal)
        sns.barplot(x='fake_val', y='domain', hue='category', order=order,
                    data=subdf[mask1], ax=axes[0,x], ci=None, palette=pal_lower)
        # else:
        #     sns.barplot(x='val', y='domain', label='platform-centric', order=order,
        #                 data=subdf[mask1], ax=axes[0,x], ci=None, palette=pal)
        #     sns.barplot(x='fake_val', y='domain', label='platform-centric', order=order,
        #                 data=subdf[mask1], ax=axes[0,x], ci=None, palette=pal_lower)

        sns.barplot(x='val', y='domain', hue='category', order=order,
                    data=subdf[mask2], ax=axes[1, x], ci=None, palette=pal)
        sns.barplot(x='fake_val', y='domain', hue='category', order=order,
                    data=subdf[mask2], ax=axes[1, x], ci=None, palette=pal_lower)
        if x == 0:
            axes[0, x].set_title('UGC Appears (all categories)')
        else:
            axes[0, x].set_title('UGC Appears (by category)')
        

        if x == 0:
            axes[1, x].set_title('UGC Appears in Top 3 (all categories)')
        else:
            axes[1, x].set_title('UGC Appears in Top 3 (by category)')
        axes[1, x].set_xlabel('Fraction of pages')
    colnum = 2
    subdf = categorized
    mask1 = (subdf.metric == 'domain_frac') & (subdf.subset == FULL)
    mask2 = (subdf.metric == 'domain_frac') & (subdf.subset == TOP_THREE)
    for subset in [FULL, TOP_THREE]:
        subdf.loc[:, 'domain'] = subdf['domain'].apply(strip_domain_strings_wrapper(subset))
    sns.barplot(x='val', y='domain', hue='category', order=order,
                data=subdf[mask1], ax=axes[0, colnum], ci=None, palette=pal)
    sns.barplot(x='fake_val', y='domain', hue='category', order=order,
                data=subdf[mask1], ax=axes[0, colnum], ci=None, palette=pal_lower)
    sns.barplot(x='val', y='domain', hue='category', order=order,
                data=subdf[mask2], ax=axes[1, colnum], ci=None, palette=pal)
    sns.barplot(x='val', y='domain', hue='category', order=order,
                data=subdf[mask2], ax=axes[1, colnum], ci=None, palette=pal_lower)

    axes[0, colnum].set_title('Fraction of total results (by category)')
    axes[1, colnum].set_title('Fraction of top 3 results (by category)')
    axes[1, colnum].set_xlabel('Fraction of results')

    for colnum in [0, 1, 2]:
        for rownum in [0, 1]:
            ax = axes[rownum, colnum]
            # ax.grid(True, 'both')
            if rownum == 0:
                ax.set(xticklabels=[], xlabel='')
            if colnum == 2:
                ax.set_xlim([0, 0.2])
            else:
                ax.set_xlim([0, 1])
            #ax.legend().set_visible(False)
            ax.legend(ncol=2, loc='lower right')
            sns.despine(ax=ax, bottom=True, left=True)
            if colnum != 0:
                ax.set_ylabel('')


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
            title_kwargs['type'] = 'aggregated'
        else:
            col_mask = df.category != 'all'
            title_kwargs['type'] = 'broken down by query set'
        coldf = df[col_mask]
        for rownum, subset in enumerate([FULL, TOP_THREE]):
            row_mask = coldf.subset == subset
            rowdf = coldf[row_mask]
            if subset == FULL:
                title_kwargs['subset'] = 'considering all results'
            elif subset == TOP_THREE:
                title_kwargs['subset'] = 'considering only top three results'

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
