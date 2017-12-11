"""Data Analysis"""
import sqlite3
import os
from string import ascii_lowercase
import csv
import argparse

from constants import POPULAR_CATEGORIES

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
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
        print_all=False, recurse_on_queries=False):
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
        summary = {key: [] for key in RESULT_SUBSETS}
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
                ret.append({
                    'column': col,
                    'winner': None,
                    'mult_increase': 1,
                    'mean_a': round(mean_a, 4),
                    'mean_b': round(mean_b, 4),
                    'name_a': self.name_a,
                    'name_b': self.name_b,
                    'pval': None,
                    'len(a)': len(a),
                    'len(b)': len(b),
                    'n': n,
                    'mean': mean,
                })
                continue
            elif mean_a > mean_b:
                larger, smaller = mean_a, mean_b
                winner, loser = self.name_a, self.name_b
            else:
                larger, smaller = mean_b, mean_a
                winner, loser = self.name_b, self.name_a
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
            if marker:
                key = None
                # what does this do
                for result_subset in RESULT_SUBSETS:
                    if result_subset + '_domain' in col:
                        key = result_subset
                if key:
                    summary[key].append({
                        'column': marker + col,
                        'winner': winner,
                        'mult_inc': mult_increase,
                        # 'add_inc': round(larger - smaller, 3),
                        'mean_a': round(mean_a, 3),
                        'mean_b': round(mean_b, 3),
                        'len(a)': len(a),
                        'len(b)': len(b),
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

        return ret, summary, err, query_comparison_lists


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
    if control_domains_col:
        control_dict = control_domains_col.value_counts().to_dict()
    else:
        control_dict = {}
    ret = {}
    for key, val in domains_dict.items():
        ret[key] = val / len(domains_col)
    for key, val in control_dict.items():
        ret[key] = (ret.get(key, 0) + val / len(control_domains_col)) / 2
    return ret

def compute_serp_features(links, domains_col, control_links, control_domains_col):
    """
    Computes features for a set of results corresponding to one serp
    Args:
        links - a list of links (as strings)
        control_links - a list of links (as strings)
    Returns:
        A dictionary of computed values
        ret: {
            jaccard index with control,
            edit distance with control,
            number of wikipedia articles in results,
            number of wikipedia articles in top 3,
            number of wikipedia articles in top 1,
        }
    """
    string, control_string = encode_links_as_strings(links, control_links)
    ret = {}
    if control_links and control_domains_col:
        ret['control_jaccard'] = jaccard_similarity(
            links, control_links
        )
        ret['control_edit'] = damerau_levenshtein_distance(
            string, control_string
        )
        control_domains_col_3 = control_domains_col.iloc[:3]
        control_domains_col_1 = control_domains_col.iloc[:1]
    else:
        control_domains_col_3 = []
        control_domains_col_1 = []

    ret[FULL] = {
        'domain_fracs': calc_domain_fracs(domains_col, control_domains_col)
    }
    ret[TOP_THREE] = {
        'domain_fracs': calc_domain_fracs(
            domains_col.iloc[:3], control_domains_col_3)
    }
    ret[TOP] = {
        'domain_fracs': calc_domain_fracs(
            domains_col.iloc[:1], control_domains_col_1)
    }
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
        if config.get('use_control'):
            control = results[results.is_control == 1]
            control_links = list(control.link)
            control_domain_col = control.domain
            if not control_links:
                print('Missing expected control links for loc {}'.format(loc))
                continue
            if not links:
                print('Missing expected links for loc {}'.format(loc))
                continue
        else:
            control = None
            control_links = []
            control_domain_col = []

        first_row = results.iloc[0]

        d[loc] = {}
        d[loc]['links'] = links
        d[loc]['has_' + first_row.link_type] = 1 if links else 0
        d[loc]['domains'] = list(treatment.domain)
        d[loc]['control_links'] = control_links
        d[loc]['computed'] = compute_serp_features(links, treatment.domain, control_links, control_domain_col)
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

def prep_data(data):
    """
    Prep operation on the dataframe:
        change nulls to false for Boolean variables
        fill null links w/ empty string
        make domain categorical variable
    args:
        data - dataframe with results
    returns:
        prepped dataframe
    """
    data = data.fillna({
        'isTweetCarousel': False,
        'isMapsPlaces': False,
        'isMapsLocations': False,
        'isNewsCarousel': False,
    })
    data.loc[data.link.isnull(), 'link'] = ''
    tweet_mask = data.isTweetCarousel == True 
    news_mask = data.isNewsCarousel == True
    kp_mask = data.link_type == 'knowledge_panel'
    maps_location_mask = data.isMapsLocations == True
    maps_places_mask = data.isMapsPlaces == True

    data.loc[tweet_mask, 'domain'] = 'TweetCarousel'
    data.loc[news_mask, 'link'] = 'NewsCarousel'
    data.loc[news_mask, 'domain'] = 'NewsCarousel'
    data.loc[kp_mask, 'link'] = 'KnowledgePanel' 
    data.loc[kp_mask, 'domain'] = 'KnowledgePanel' 

    data.loc[maps_location_mask, 'link'] = 'MapsLocations' 
    data.loc[maps_location_mask, 'domain'] = 'MapsLocations'

    data.loc[maps_places_mask, 'link'] = 'MapsPlaces' 
    data.loc[maps_places_mask, 'domain'] = 'MapsPlaces' 

    data.domain = data.domain.astype('category')
    return data


def get_dataframes(dbname):
    """
    Get rows from the db and convert to dataframes
    """
    conn = sqlite3.connect('dbs/' + dbname)
    select_results = (
        """
        SELECT serp.*, link.*, scraper_searches_serps.scraper_search_id from serp INNER JOIN link on serp.id = link.serp_id
        INNER JOIN scraper_searches_serps on serp.id = scraper_searches_serps.serp_id;
        """
    )
    select_serps = (
        """
        SELECT * from serp;
        """
    )
    data = pd.read_sql_query(select_results, conn)
    serp_df = pd.read_sql_query(select_serps, conn)
    conn.close()
    return data, serp_df


def main(args, category):
    """Do analysis"""
    if 'dbs' in args.db:
        args.db = args.db[4:]
    data, serp_df = get_dataframes(args.db)
    data = prep_data(data)
    path1 = 'output'
    path2 = '{}/{}'.format(path1, args.db)
    if category:
        path2 += '__' + category
    for path in [path1, path2]:
        try:
            os.mkdir(path)
        except OSError:
            pass
    serp_df.reported_location.value_counts().to_csv(path2 + '/values_counts_reported_location.csv')
    # slight improvement below
    scraper_search_id_set = data.scraper_search_id.drop_duplicates()

    link_types = [
        'results',
        'tweets',
        'top_ads',
        'knowledge_panel',
        # 'bottom_ads',
        # 'news'
    ]

    serp_comps = {}

    config = {}
    config['use_control'] = False

    link_type_to_domains = {}

    for link_type in link_types:        
        link_type_specific_data = data[data.link_type == link_type]
        if category in ['trending', 'procon_popular']:
            link_type_specific_data = link_type_specific_data[link_type_specific_data['category'] == category]
        elif category == 'popular':
            link_type_specific_data = link_type_specific_data[link_type_specific_data['category'].isin(POPULAR_CATEGORIES)]
        path3 = '{}/{}'.format(path2, link_type)
        try:
            os.mkdir(path3)
        except OSError:
            pass
        link_type_specific_data.domain.value_counts().to_csv(path3 + '/values_counts_domain.csv')

        top_domains = list(link_type_specific_data.domain.value_counts().to_dict().keys())
        top_domains = [domain for domain in top_domains if isinstance(domain,str)]
        # top_domains += ['TweetCarousel', 'MapsLocations', 'MapsPlaces']
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
                    # when there are NO comparisons at all edit is 0 and jacc is nan
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
                    for domain_string, frac in domain_fracs.items():
                        for top_domain in top_domains:
                            if domain_string == top_domain:
                                concat_key = '_'.join(
                                    [link_type, comp_key, 'domain_frac', str(domain_string)]
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

    outputs, errors = [], []
    summaries = {key: [] for key in RESULT_SUBSETS}
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
                summaries[key] += summary[key]
            outputs += out
            errors += error

        output_df = pd.DataFrame(outputs)
        output_df.to_csv(path2+ '/comparisons.csv')
        for key in RESULT_SUBSETS:
            subset_summary_df = pd.DataFrame(summaries[key])
            subset_summary_df.to_csv(path2 + '/' + key + '_summary.csv')
            query_comparison_df = pd.DataFrame(query_comparison_lists[key])
            query_comparison_df.to_csv(path3 + '/' + key +'_query_comparisons.csv')

        with open(path2 + '/errs.csv','w') as outfile:
            writer = csv.writer(outfile)        
            for row in errors:
                writer.writerow([row])


def parse():
    """parse args"""
    
    parser = argparse.ArgumentParser(description='Perform anlysis.')

    parser.add_argument(
        '--comparison', help='What comparison to do', default='all')
    parser.add_argument(
        '--category', help='What comparison to do', default='all')
    parser.add_argument(
        '--db', help='Name of the database', default='tmp/test_5kw_1loc.db')
    parser.add_argument(
        '--print_all', dest='print_all', help='Whether to print ALL comparisons', action='store_true')
    parser.set_defaults(print_all=False)        

    args = parser.parse_args()
    if args.category == 'each':
        for cat in ['popular', 'trending', 'procon_popular']:    
            main(args, cat)
    else:
        main(args, args.category)

parse()
