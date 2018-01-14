"""Data Analysis"""
import os
from string import ascii_lowercase
import csv
import argparse

from constants import POPULAR_CATEGORIES
from data_helpers import get_dataframes, load_coded_as_dicts, prep_data
from qual_code import TWITTER_DOMAIN, strip_twitter_screename, UGC_WHITELIST

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyxdameraulevenshtein import damerau_levenshtein_distance

FULL = 'full'
TOP_THREE = 'top_three'
TOP = 'top'
RESULT_SUBSETS = [FULL, TOP_THREE, TOP]

class Comparison():
    """
    A comparison entity
    For comparing two groups of results within a set of results
    """
    def __init__(
            self, df_a, name_a, df_b, name_b, cols_to_compare,
            print_all=False, recurse_on_queries=False
        ):
        self.df_a = df_a
        self.name_a = name_a
        self.df_b = df_b
        self.name_b = name_b
        self.cols_to_compare = cols_to_compare
        self.print_all = print_all
        self.recurse_on_queries = recurse_on_queries

    def print_results(self):
        """
        Compare columns for the two groups belonging to this Comparison entity
        Prints out the results
        """
        ret = []
        err = []
        query_comparison_lists = {key: [] for key in RESULT_SUBSETS}
        pval_summary = {key: [] for key in RESULT_SUBSETS}
        whitelist_summary = {key: [] for key in RESULT_SUBSETS}
        for col in self.cols_to_compare:
            try:
                filtered_df_a = self.df_a[self.df_a[col].notnull()]
                a = list(filtered_df_a[col])
            except KeyError:
                if self.print_all:
                    print('Column {} missing from df_a, {}'.format(col, self.name_a))
                continue
            try:
                filtered_df_b = self.df_b[self.df_b[col].notnull()]
                b = list(filtered_df_b[col])
            except KeyError:
                if self.print_all:
                    print('Column {} missing from df_a, {}'.format(col, self.name_a))
                continue

            if not a and not b:
                err.append('Skipping {} bc Two empty lists'.format(col))
                continue
            mean = np.mean( np.array(a + b), axis=0 )
            mean_a = np.mean(a)
            mean_b = np.mean(b)
            n = len(a) + len(b)

            _, pval = ttest_ind(a, b, equal_var=False)
            if mean_a == mean_b:
                larger, smaller = mean_a, mean_b
                winner = None
            elif mean_a > mean_b:
                larger, smaller = mean_a, mean_b
                winner = self.name_a
            else:
                larger, smaller = mean_b, mean_a
                winner = self.name_b
            mult_increase = round(larger / smaller, 2)
            marker = ''
            if pval <= 0.001:
                marker = '**'
            elif pval <= 0.05:
                marker = '*'
            ret.append({
                'column': marker + col,
                'winner': winner,
                'mult_inc': mult_increase,
                'add_inc': round(larger - smaller, 3),
                'mean_a': round(mean_a, 3),
                'mean_b': round(mean_b, 3),
                'name_a': self.name_a,
                'name_b': self.name_b,
                'pval': pval,
                'len(a)': len(a),
                'len(b)': len(b),
                'n': n,
                'mean': mean,
            })
            is_in_whitelist = False
            for domain in UGC_WHITELIST:
                if domain in col:
                    is_in_whitelist = True
            key = None
            for result_subset in RESULT_SUBSETS:
                if result_subset + '_domain' in col:
                    key = result_subset
            if key:
                if is_in_whitelist:
                    whitelist_summary[key].append({
                        'column': marker + col,
                        'winner': winner,
                        'mult_inc': mult_increase,
                        'add_inc': round(larger - smaller, 3),
                        'mean_a': round(mean_a, 3),
                        'mean_b': round(mean_b, 3),
                        'name_a': self.name_a,
                        'name_b': self.name_b,
                        'pval': pval,
                        'len(a)': len(a),
                        'len(b)': len(b),
                        'n': n,
                        'mean': mean,
                    })
                if marker:
                    pval_summary[key].append({
                        'column': marker + col,
                        'winner': winner,
                        'mult_inc': mult_increase,
                        'add_inc': round(larger - smaller, 3),
                        'mean_a': round(mean_a, 3),
                        'mean_b': round(mean_b, 3),
                        'name_a': self.name_a,
                        'name_b': self.name_b,
                        'pval': pval,
                        'len(a)': len(a),
                        'len(b)': len(b),
                        'n': n,
                        'mean': mean,
                    })
                    if self.recurse_on_queries:
                        # now mark all the comparisons
                        queries = set(
                            list(filtered_df_a['query'].drop_duplicates()) + list(filtered_df_b['query'].drop_duplicates())
                        )
                        for query in queries:
                            query_a = filtered_df_a[filtered_df_a['query'] ==  query]
                            query_b = filtered_df_b[filtered_df_b['query'] ==  query]
                            query_comparison = Comparison(
                                df_a=query_a, name_a=self.name_a,
                                df_b=query_b, name_b=self.name_b,
                                cols_to_compare=[col],
                                print_all=self.print_all,
                                recurse_on_queries=False,
                            )
                            comparison_dicts = query_comparison.print_results()[0]
                            comparison_dicts = [x for x in comparison_dicts if x['mean'] != 0]
                            for d in comparison_dicts:
                                d['query'] = query
                            query_comparison_lists[key] += comparison_dicts
        summary = {
            'pval': pval_summary,
            'whitelist': whitelist_summary
        }
        return ret, summary, err, query_comparison_lists


def get_matching_columns(columns, whitelist):
    """
    Takes a list of columns and returns the ones that match whitelist
    """
    ret = []
    for x in whitelist:
        for column in columns:
            if x in column and column not in ret:
                ret.append(column)
    return ret

def encode_links_as_strings(links1, links2):
    """
    Take two lists of pages and turn them into strings
    For the sole purpose of calculating edit distance
    """
    set1, set2 = set(links1), set(links2)
    union = set1.union(set2) 
    mapping = {}
    # will never have more than 10 results...
    for item, letter in zip(list(union), ascii_lowercase):
        mapping[item] = letter
    string1 = ''.join([mapping[link] for link in links1])
    string2 = ''.join([mapping[link] for link in links2])
    return string1, string2


def jaccard_similarity(x, y):
    """
    set implementation of jaccard similarity
    """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)


def calc_domain_fracs(domains_col, control_domains_col):
    """
    This is specific to a given SERP
    Figure out how many domains of interest appear in search results

    return a dict
    """
    domains_dict = domains_col.value_counts().to_dict()
    if not control_domains_col.empty:
        control_dict = control_domains_col.value_counts().to_dict()
    else:
        control_dict = {}
    ret = {}
    for key, val in domains_dict.items():
        ret[key] = val / len(domains_col)
    for key, val in control_dict.items():
        ret[key] = (ret.get(key, 0) + val / len(control_domains_col)) / 2
    return ret


def calc_domain_ranks(cols):
    """
    average rank of a given domain

    danger
    if you ever use this with control_domains you should probably double check it
    """
    ret = {}
    domains_dict = cols.domain.value_counts().to_dict()
    for key, val in domains_dict.items():
        if val == 0:
            ret[key] = None
            continue
        matching_rows = cols[cols.domain == key]
        total_rank = sum(matching_rows['rank'])
        ret[key] = total_rank / val
    return ret


def calc_coded_ugc_frac(code_col, control_code_col):
    """
    This is specific to a given SERP
    Figure out how many domains of interest appear in search results

    return a dict
    """
    return calc_domain_fracs(code_col, control_code_col)


def compute_serp_features(
        links, cols, 
        control_links, control_cols,
    ):
    """
    Computes features for a set of results corresponding to one serp
    Args:
        links - a list of links (as strings)
        control_links - a list of links (as strings)
        domains_col - a pandas series corresponding to the "domain" column
        code_col - a pandas series corresponding to the "code" column
    Returns:
        A dictionary of computed values
        ret: {
            jaccard index with control,
            edit distance with control,
            domain_fracs for results, top3, top1,
        }
    """
    string, control_string = encode_links_as_strings(links, control_links)
    ret = {}
    if control_links and control_cols:
        ret['control_jaccard'] = jaccard_similarity(
            links, control_links
        )
        ret['control_edit'] = damerau_levenshtein_distance(
            string, control_string
        )

    ret[FULL] = {
        'domain_fracs': calc_domain_fracs(cols.domain, control_cols.domain),
    }
    ret[FULL]['domain_ranks'] = calc_domain_ranks(cols)
    
    
    ret[TOP_THREE] = {
        'domain_fracs': calc_domain_fracs(
            cols.iloc[:3].domain, control_cols.iloc[:3].domain)
    }
    ret[TOP] = {
        'domain_fracs': calc_domain_fracs(
            cols.iloc[:1].domain, control_cols.iloc[:1].domain)
    }
    for subset in RESULT_SUBSETS:
        ret[subset]['domain_appears'] = {}
        for key, val in ret[subset]['domain_fracs'].items():
            if val > 0:
                ret[subset]['domain_appears'][key] = 1
            else:
                ret[subset]['domain_appears'][key] = 0
    ret[FULL]['coded_ugc_fracs'] = calc_coded_ugc_frac(cols.code, control_cols.code)
    ret[TOP_THREE]['coded_ugc_fracs'] = calc_coded_ugc_frac(cols.iloc[:3].code, control_cols.iloc[:3].code)
    ret[TOP]['coded_ugc_fracs'] = calc_coded_ugc_frac(cols.iloc[:1].code, control_cols.iloc[:1].code)

    ret[FULL]['coded_ugc_fracs'].update(calc_coded_ugc_frac(
        cols.domains_plus_codes, control_cols.domains_plus_codes))
    ret[TOP_THREE]['coded_ugc_fracs'].update(calc_coded_ugc_frac(
        cols.iloc[:3].domains_plus_codes, control_cols.iloc[:3].domains_plus_codes))
    ret[TOP]['coded_ugc_fracs'].update(calc_coded_ugc_frac(
        cols.iloc[:1].domains_plus_codes, control_cols.iloc[:1].domains_plus_codes))
    return ret


def analyze_subset(data, location_set, config):
    """
    A subset consists of results of a certain TYPE for a certain QUERY
    Args:
        data - a dataframe object with rows matching a TYPE and QUERY
        location_set - a set of strings corresponding to locations queried
    """
    # d holds the results and editdistances
    # good variable naming was skipped for coding convenience. refactor later?
    d = {}
    for loc in location_set:
        results = data[data.reported_location == loc]
        if results.empty:
            continue
        treatment = results[results.is_control == 0]
        links = list(treatment.link)
        snippets = list(treatment.snippet)
        titles = list(treatment.title)
        if config.get('check_ranks'):
            ranks = list(treatment['rank'])
            largest_rank = ranks[-1]
            perfect_sequence = set(range(1, largest_rank + 1))
            missing_ranks = perfect_sequence.difference(set(ranks))
            if missing_ranks and missing_ranks != set([1]):
                print(results[['query', 'link', 'rank']])
                print(set(ranks))
                print(perfect_sequence)
                print(missing_ranks)
                input()
        if config.get('use_control'):
            control = results[results.is_control == 1]
            control_links = list(control.link)
            if not control_links:
                print('Missing expected control links for loc {}'.format(loc))
                continue
            if not links:
                print('Missing expected links for loc {}'.format(loc))
                continue
        else:
            control = pd.DataFrame(
                data={
                    'domain': [],
                    'code': [],
                    'rank': [],
                    'domains_plus_codes': [],
                }
            )
            control_links = []

        first_row = results.iloc[0]

        d[loc] = {}
        d[loc]['links'] = links
        d[loc]['has_' + first_row.link_type] = 1 if links else 0
        d[loc]['domains'] = list(treatment.domain)
        d[loc]['control_links'] = control_links
        d[loc]['computed'] = compute_serp_features(
            links, treatment[['domain', 'code', 'rank', 'domains_plus_codes']],
            control_links, control[['domain', 'code', 'rank', 'domains_plus_codes']],
        )
        d[loc]['serp_id'] = first_row.serp_id
        sid = SentimentIntensityAnalyzer()
        snippet_polarities = [sid.polarity_scores(x)['compound'] for x in snippets if x]
        title_polarities = [sid.polarity_scores(x)['compound'] for x in titles if x]
        for polarities, textname in [
                (snippet_polarities, 'snippet'),
                (title_polarities, 'title')
        ]:
            for prefix, subset in [
                    (FULL, polarities),
                    (TOP_THREE, polarities[:3]),
                    (TOP, polarities[:1]),
            ]:
                if subset:
                    mean_polarity = sum(subset) / len(subset)
                    d[loc]['computed'][prefix + '_' + textname + '_mean_polarity'] = mean_polarity

    for loc in location_set:
        if loc not in d:
            continue
        d[loc]['comparisons'] = {}
        tmp = d[loc]['comparisons']
        for comparison_loc in location_set:
            if comparison_loc not in d:
                continue
            if loc == comparison_loc:
                continue
            tmp[comparison_loc] = {}
            string1, string2 = encode_links_as_strings(
                    d[loc]['links'], d[comparison_loc]['links'])
            tmp[comparison_loc]['edit'] = \
                damerau_levenshtein_distance(
                string1, string2
            )
            try:
                jac = jaccard_similarity(
                        d[loc]['links'], 
                        d[comparison_loc]['links']
                    )
            except ZeroDivisionError:
                jac = float('nan')
            tmp[comparison_loc]['jaccard'] = jac
    return d


def prep_paths(db, category):
    """
    Creates paths in the filesystem and return the path names
    """
    path1 = 'output'
    path2 = '{}/{}'.format(path1, db)
    if category:
        path2 += '__' + category
    for path in [path1, path2]:
        try:
            os.mkdir(path)
        except OSError:
            pass
    return path1, path2


def main(args, db, category):
    """Do analysis"""
    data, serp_df = get_dataframes(db)
    data = prep_data(data)
    if args.group_popular:
        pop_mask = serp_df['category'].isin(POPULAR_CATEGORIES)
        serp_df.loc[pop_mask, 'category'] = 'popular'
        pop_mask = data['category'].isin(POPULAR_CATEGORIES)
        data.loc[pop_mask, 'category'] = 'popular'
    categories = list(data['category'].drop_duplicates())
    print(categories)
    if category not in categories:
        print('Skipping category {}'.format(category))
        return pd.DataFrame()
    if 'dbs' in db:
        shortened_db = db[4:]
    else:
        shortened_db = db
    _, path2 = prep_paths(shortened_db, category)

    link_codes_file = 'link_codes.csv'
    twitter_user_codes_file = 'twitter_user_codes.csv'
    link_codes, twitter_user_codes = load_coded_as_dicts(link_codes_file, twitter_user_codes_file)
    for link, code in link_codes.items():
        data.loc[data.link == link, 'code'] = code
    twitter_data = data[data.domain == TWITTER_DOMAIN]
    
    twitter_links = twitter_data.link.drop_duplicates()
    for link in twitter_links:
        screen_name = strip_twitter_screename(link)
        code = twitter_user_codes.get(screen_name)
        if not code:
            print('Could not get code for screen_name {}'.format(screen_name))
        data.loc[data.link == link, 'code'] = code
    data.code = data.code.astype('category')
    domains_plus_codes = [
        str(x) + '_' + str(y) for x, y in zip(
            list(data.domain),
            list(data.code)
        )
    ]
    data = data.assign(domains_plus_codes=domains_plus_codes)
    data.domains_plus_codes = data.domains_plus_codes.astype('category')
    data.describe(include='all').to_csv(path2+'/data.describe().csv')
    serp_df.reported_location.value_counts().to_csv(path2 + '/values_counts_reported_location.csv')
    serp_df['query'].value_counts().to_csv(path2 + '/values_counts_query.csv')
    scraper_search_id_set = data.scraper_search_id.drop_duplicates()
    

    link_types = [
        'results',
        #'tweets',
        #'top_ads',
        #'knowledge_panel',
        # ['results', 'tweets'],
        # ['results', 'knowledge_panel']
    ]

    serp_comps = {}
    config = {}
    config['use_control'] = False
    config['check_ranks'] = False

    link_type_to_domains = {}

    for i, link_type in enumerate(link_types):        
        if isinstance(link_type, list):
            mask = data.link_type == link_type[0]
            for x in link_type:
                mask = (mask) | (data.link_type == x)
            link_type_specific_data = data[mask]
            link_type = '_and_'.join(link_type)
            link_types[i] = link_type # carry this beyond the for loop
        else:
            link_type_specific_data = data[data.link_type == link_type]
        if category in [
            'trending', 'procon_popular', 'popular', 'top_insurance', 'top_loans',
            'med_sample_first_20'
        ]:
            link_type_specific_data = link_type_specific_data[link_type_specific_data['category'] == category]
        path3 = '{}/{}'.format(path2, link_type)
        try:
            os.mkdir(path3)
        except OSError:
            pass
        link_type_specific_data.domain.value_counts().to_csv(path3 + '/values_counts_domain.csv')

        top_domains = list(link_type_specific_data.domain.value_counts().to_dict().keys())[:30]
        top_domains = [domain for domain in top_domains if isinstance(domain,str)]
        link_type_to_domains[link_type] = top_domains
        for scraper_search_id in scraper_search_id_set:
            filtered = link_type_specific_data[link_type_specific_data.scraper_search_id == scraper_search_id]
            if filtered.empty:
                continue
            queries = list(filtered['query'].drop_duplicates())
            if len(queries) != 1:
                raise ValueError('Multiple queries found in a single serp')
            location_set = filtered.reported_location.drop_duplicates()
            d = analyze_subset(filtered, location_set, config)

            for loc, vals in d.items():
                sid = vals['serp_id']
                tmp = d[loc]['computed']
                dist_sum, jacc_sum, count = 0, 0, 0
                for _, metrics in vals['comparisons'].items():
                    dist_sum += metrics['edit']
                    jacc_sum += metrics['jaccard']
                    count += 1
                if count:
                    avg_edit = dist_sum / count
                    avg_jacc = jacc_sum / count
                else:
                    avg_edit = avg_jacc = float('nan')
                tmp[link_type + '_avg_edit'] = avg_edit
                tmp[link_type + '_avg_jaccard'] = avg_jacc

                # make sure we're NOT overwriting an already existent sub-dict!
                # (this comment suggests a foolish programmer did this in the past)
                if sid not in serp_comps:
                    serp_comps[sid] = { 'id': sid }
                serp_comps[sid][link_type + '_avg_edit'] = avg_edit
                serp_comps[sid][link_type + '_avg_jacc'] = avg_jacc
                has_type_key = 'has_' + link_type
                serp_comps[sid][has_type_key] = d[loc].get(has_type_key, 0)
                for comp_key in RESULT_SUBSETS:
                    domain_fracs = tmp[comp_key]['domain_fracs']
                    coded_ugc_fracs = tmp[comp_key]['coded_ugc_fracs']
                    for domain_string, frac in domain_fracs.items():
                        for top_domain in top_domains:
                            if domain_string == top_domain:
                                concat_key = '_'.join(
                                    [link_type, comp_key, 'domain_frac', str(domain_string)]
                                )
                                serp_comps[sid][concat_key] = frac
                                domain_appears_concat_key = concat_key.replace('_frac', '_appears')
                                serp_comps[sid][domain_appears_concat_key] = tmp[comp_key]['domain_appears'][domain_string]
                                
                                if comp_key == FULL:
                                    domain_ranks_concat_key = concat_key.replace('_frac', '_rank')
                                    serp_comps[sid][domain_ranks_concat_key] = tmp[comp_key]['domain_ranks'][domain_string]
                    for code, frac in coded_ugc_fracs.items():
                        concat_key = '_'.join(
                            [link_type, comp_key, 'coded_ugc_frac', str(code)]
                        )
                        serp_comps[sid][concat_key] = frac
                    for textcol in ['snippet', 'title']:
                        pol_key = '_'.join([link_type, comp_key, textcol, 'mean_polarity'])
                        serp_comps[sid][pol_key] = tmp.get(
                            '_'.join([comp_key, textcol, 'mean_polarity'])
                        )

    serp_comps_df = pd.DataFrame.from_dict(serp_comps, orient='index')
    serp_comps_df.index.name = 'id'
    serp_df = serp_df.merge(serp_comps_df, on='id')
    serp_df.reported_location = serp_df.reported_location.astype('category')
    serp_df.describe(include='all').to_csv(path2+'/serp_df.describe().csv')

    # ANCHOR: plotting
    ret_cols = []
    # TODO: consider adding something to w/ codes here
    cols = get_matching_columns(list(serp_df.columns.values), UGC_WHITELIST + [
        'NewsCarousel', 'MapsPlaces', 'MapsLocations'
    ])
    cols_with_nonzero_mean = [
        x for x in cols if serp_df[x].mean() != 0
    ]
    serp_df[cols_with_nonzero_mean].describe().to_csv(path2+'/ugcin_serp_df.csv')
    # nz for non-zero (variable name was too long)
    results_domain_fracs_cols_nz = [
        x for x in cols_with_nonzero_mean if 'results_' in x and 'domain_frac' in x
    ]
    results_domain_ranks_cols_nz = [
        x for x in cols_with_nonzero_mean if 'results_' in x and 'domain_rank' in x
    ]
    results_domain_appears_cols_nz = [
        x for x in cols_with_nonzero_mean if 'results_' in x and 'domain_appears' in x
    ]
    _, domain_fracs_ax = plt.subplots(nrows=3)
    _, axes2 = plt.subplots(nrows=4)
    _, big_fracs_ax = plt.subplots(nrows=3)
    _, dist_axes = plt.subplots(nrows=2)
    _, personalization_ax = plt.subplots(nrows=2)
    for index, subset in enumerate(RESULT_SUBSETS):
        results_domain_fracs_cols_nz_subset = [
            x for x in results_domain_fracs_cols_nz if subset + '_domain_frac' in x
        ]
        results_domain_appears_cols_nz_subset = [
            x for x in results_domain_appears_cols_nz if subset + '_domain_appears' in x
        ]
        big_fraction_cols = [
            x for x in list(serp_df.columns.values) if 'results_' + subset + '_domain_frac' in x and serp_df[x].mean() > 0.01
        ]
        if results_domain_fracs_cols_nz_subset:
            serp_df[results_domain_fracs_cols_nz_subset].mean().sort_values().plot(
                kind='barh', ax=domain_fracs_ax[index], title='Category: {}, Domain Fractions: {}'.format(category, subset))
            ret_cols += results_domain_appears_cols_nz_subset
        if results_domain_appears_cols_nz_subset:
            serp_df[results_domain_appears_cols_nz_subset].mean().sort_values().plot(
                kind='barh', ax=axes2[index], title='Domain Appears: {}'.format(subset))
            ret_cols += results_domain_appears_cols_nz_subset
        if big_fraction_cols:
            serp_df[big_fraction_cols].mean().sort_values().plot(
                kind='barh', ax=big_fracs_ax[index], title='Big Fractions: {}'.format(subset))
    serp_df[results_domain_ranks_cols_nz].mean().sort_values().plot(
        kind='barh', ax=axes2[3], title='Domain Ranks')

    
    sns.set(style="white", palette="colorblind", color_codes=True)
    wp_vals = serp_df[
        'results_full_domain_rank_wikipedia.org'][serp_df['results_full_domain_rank_wikipedia.org'].notnull() == True]
    sns.distplot(
        wp_vals, bins=list(range(1,13)), norm_hist=True,
        kde=False, color="b", ax=dist_axes[0]) 
    dist_axes[0].axvline(wp_vals.mean(), color='b', linestyle='dashed', linewidth=2)
    try:
        tw_vals = serp_df[
            'results_full_domain_rank_UserTweetCarousel'][serp_df['results_full_domain_rank_UserTweetCarousel'].notnull() == True]
        sns.distplot(
            tw_vals, bins=list(range(1,13)), norm_hist=True,
            kde=False, color="g", ax=dist_axes[1])
        dist_axes[1].axvline(tw_vals.mean(), color='g', linestyle='dashed', linewidth=2)
    except:
        pass
    # PERSONALIZATION
    jacc_vals = serp_df[serp_df['results_avg_jacc'].notnull() == True]['results_avg_jacc']
    sns.distplot(
        jacc_vals, norm_hist=True,
        kde=False, color="b", ax=personalization_ax[0])
    personalization_ax[0].axvline(jacc_vals.mean(), color='b', linestyle='dashed', linewidth=2)
    edit_vals = serp_df[serp_df['results_avg_edit'].notnull() == True]['results_avg_edit']
    sns.distplot(
        edit_vals, norm_hist=True,
        kde=False, color="g", ax=personalization_ax[1]) 
    personalization_ax[1].axvline(edit_vals.mean(), color='g', linestyle='dashed', linewidth=2)

    outputs, errors = [], []
    pval_summaries = {key: [] for key in RESULT_SUBSETS}
    whitelist_summaries = {key: [] for key in RESULT_SUBSETS}
    query_comparison_listss = {key: [] for key in RESULT_SUBSETS}
    for link_type in link_types:
        path3 = '{}/{}'.format(path2, link_type)
        cols_to_compare = []
        if link_type != 'tweets':
            # tweets all have the same domain - twitter.com!
            for top_domain in set(link_type_to_domains[link_type]):
                for prefix in [
                        '_full_domain_frac_', '_top_three_domain_frac_',
                        '_top_domain_frac_',
                ]:
                    cols_to_compare.append(link_type + prefix + top_domain)
        for col in [
            '_full_snippet_mean_polarity', '_top_three_snippet_mean_polarity', '_top_snippet_mean_polarity',
            '_full_snippet_mean_polarity', '_top_three_snippet_mean_polarity', '_top_snippet_mean_polarity',
            '_avg_jacc', '_avg_edit'
        ]:
            cols_to_compare.append(link_type + col)

        # SERPS that have NO TWEETS or NO NEWS (etc)
        # will have nan values for any related calculations (e.g. avg_jacc of Tweets)
        if link_type == 'results':
            cols_to_fillna = [
                'has_knowledge_panel',
                'has_top_ads',
                'has_bottom_ads',
            ]
            serp_df = serp_df.fillna({
                col: 0 for col in cols_to_fillna
            })
            for col in cols_to_fillna:
                cols_to_compare.append(col)

        comparisons = []
        if args.comparison in ['urban-rural', 'all']:
            comparisons.append(Comparison(
                df_a=serp_df[(serp_df['urban_rural_code'] == 5) | (serp_df['urban_rural_code'] == 6)],
                name_a='rural',
                df_b=serp_df[(serp_df['urban_rural_code'] == 1) | (serp_df['urban_rural_code'] == 2)],
                name_b='urban',
                cols_to_compare=cols_to_compare,
                print_all=args.print_all,
                recurse_on_queries=True
            ))
        if args.comparison in ['income', 'all']:
            comparisons.append(Comparison(
                df_a=serp_df[serp_df['median_income'] <= 45111],
                name_a='low-income',
                df_b=serp_df[serp_df['median_income'] > 45111],
                name_b='high-income',
                cols_to_compare=cols_to_compare,
                print_all=args.print_all,
                recurse_on_queries=True
            ))
        if args.comparison in ['voting', 'all']:
            comparisons.append(Comparison(
                df_a=serp_df[serp_df['percent_dem'] <= 0.5],
                name_a='GOP',
                df_b=serp_df[serp_df['percent_dem'] > 0.5],
                name_b='DEM',
                cols_to_compare=cols_to_compare,
                print_all=args.print_all,
                recurse_on_queries=True
            ))

        for comparison in comparisons:
            out, summary, error, query_comparison_lists = comparison.print_results()
            for key in RESULT_SUBSETS:
                pval_summaries[key] += summary['pval'][key]
                whitelist_summaries[key] += summary['whitelist'][key]
                query_comparison_listss[key] += query_comparison_lists[key]
            outputs += out
            errors += error

        # write out the comparisons
        output_df = pd.DataFrame(outputs)
        output_df.to_csv(path2+ '/comparisons.csv')

        # write out a summary of statistically significant comparisons
        paper_table_list = []
        for key in RESULT_SUBSETS:
            paper_table_list += pval_summaries[key]
            paper_table_list += whitelist_summaries[key]
            pval_summary_df = pd.DataFrame(pval_summaries[key])
            pval_summary_df.to_csv(path2 + '/' + key + '_pval_summary.csv')
            whitelist_summary_df = pd.DataFrame(whitelist_summaries[key])
            whitelist_summary_df.to_csv(path2 + '/' + key + '_whitelist_summary.csv')
            query_comparison_df = pd.DataFrame(query_comparison_listss[key])
            query_comparison_df.to_csv(path3 + '/' + key +'_query_comparisons.csv')
        pd.DataFrame(paper_table_list).to_csv(path2+'/paper_table.csv')

        with open(path2 + '/errs.csv','w') as outfile:
            writer = csv.writer(outfile)
            for row in errors:
                writer.writerow([row])
    return serp_df[list(set(ret_cols)) + ['category']]
    


def parse():
    """parse args"""
    parser = argparse.ArgumentParser(description='Perform anlysis.')

    parser.add_argument(
        '--comparison', help='What comparison to do', default='all')
    parser.add_argument(
        '--category', help='Which category to include in the analysis', default='each')
    parser.add_argument(
        '--db', help='Name of the database(s)', nargs='?', required=True)
    parser.add_argument(
        '--print_all', dest='print_all', help='Whether to print ALL comparisons', action='store_true')
    parser.add_argument(
        '--plot', dest='plot', help='Whether to plot', action='store_true')
    parser.add_argument(
        '--group_popular', dest='group_popular', help='treat all popular queries as once group for the purposes of plotting', action='store_true', default=True)
    parser.set_defaults(print_all=False)

    args = parser.parse_args()
    df = None
    for db in args.db:
        if args.category == 'each':
            for cat in ['popular', 'trending', 'procon_popular', 'top_insurance', 'top_loans', 'med_sample_first_20', ]:
                df_for_cat = main(args, db, cat)
                if df is None:
                    df = df_for_cat
                else:
                    df = pd.concat([df, df_for_cat])
        else:
            df = main(args, db, args.category)
        row_dicts = []
    for col in df.columns.values:
        if col:
            if '_domain_appears_' in col:
                # tmp is of the form resulttype_subset
                # e.g. results_top_three
                tmp, domain = col.split('_domain_appears_')
                link_type, subset = None, None
                for key in RESULT_SUBSETS:
                    if key + '_domain' in col:
                        subset = key
                        link_type = tmp.replace(key, '').strip('_')
                for _, row in df.iterrows():
                    row_dict = {
                        'link_type': link_type,
                        'subset': subset,
                        'metric': 'domain_appears',
                        'domain': domain,
                        'val':  row[col],
                        'category': row['category'],
                    }
                    row_dicts.append(row_dict)
            # _, axes = plt.subplots()
            # sns.barplot(x='category', y=col, data=df, ax=axes)
    df2 = pd.DataFrame(row_dicts)
    plt.close('all')
    _, axes = plt.subplots(nrows=2)
    row1_mask = (df2.metric == 'domain_appears') & (df2.subset == FULL)
    row2_mask = (df2.metric == 'domain_appears') & (df2.subset == TOP_THREE)
    tmpdf = df2[df2.subset == FULL][['domain', 'val']]
    order = list(tmpdf.groupby('domain').mean().sort_values('val', ascending=False).index)
    sns.barplot(x='domain', y='val', hue='category', order=order, data=df2[row1_mask], ax=axes[0], ci=99)
    sns.barplot(x='domain', y='val', hue='category', order=order, data=df2[row2_mask], ax=axes[1], ci=99)

    if args.plot:
        plt.show()



if __name__ == '__main__':
    parse()
