import sqlite3
import operator
from pprint import pprint
from collections import defaultdict
from string import ascii_lowercase

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from pyxdameraulevenshtein import damerau_levenshtein_distance
import argparse


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

    ret['full'] = {
        'domain_fracs': calc_domain_fracs(domains_col, control_domains_col)
    }
    ret['top_three'] = {
        'domain_fracs': calc_domain_fracs(
            domains_col.iloc[:3], control_domains_col_3)
    }
    ret['top'] = {
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
        polarity_scores = [sid.polarity_scores(x)['compound'] for x in snippets if x]
        for prefix, subset in [
            ('full', polarity_scores),
            ('top_three', polarity_scores[:3]),
            ('top', polarity_scores[:1]),
        ]:
            mean_polarity = sum(subset) / len(subset)
            d[loc]['computed'][prefix + '_mean_polarity'] = mean_polarity

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
                jac = 0
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
    maps_location_mask = data.link_type == 'maps_locations'
    maps_places_mask = data.link_type == 'maps_places'

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
    conn = sqlite3.connect(dbname)
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


def main(args):
    """Do analysis"""
    data, serp_df = get_dataframes(args.db)
    data = prep_data(data)
    # print(serp_df['query'].value_counts())
    # print(serp_df.reported_location.value_counts())

    
    # slight improvement below
    scraper_search_id_set = data.scraper_search_id.drop_duplicates()

    link_types = [
        'results',
        'tweets',
        # 'bottom_ads',
        # 'news'
    ]

    serp_comps = {}

    config = {}
    config['use_control'] = False

    all_domains = []

    for link_type in link_types:
        link_type_specific_data = data[data.link_type == link_type]
        top_domains = list(link_type_specific_data.domain.value_counts().to_dict().keys())[:20]
        top_domains = [domain for domain in top_domains if isinstance(domain,str)]
        top_domains += ['TweetCarousel', 'MapsLocations', 'MapsPlaces']
        all_domains += top_domains
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
                tmp['avg_edit'] = avg_edit
                tmp['avg_jaccard'] = avg_jacc
                if sid not in serp_comps:
                    serp_comps[sid] = { 'id': sid }
                serp_comps[sid][link_type + 'avg_edit'] = avg_edit
                serp_comps[sid][link_type + 'avg_jacc'] = avg_jacc
                has_type_key = 'has_' + link_type
                serp_comps[sid][has_type_key] = d[loc].get(has_type_key, 0)
                for comp_key in ['full', 'top_three', 'top', ]:
                    domain_fracs = tmp[comp_key]['domain_fracs']
                    for domain_string, frac, in domain_fracs.items():
                        for top_domain in top_domains:
                            if domain_string == top_domain:
                                concat_key = '_'.join(
                                    [link_type, comp_key, 'domain_frac', str(domain_string)]
                                )
                                serp_comps[sid][concat_key] = frac
                    pol_key = link_type + '_' + comp_key + '_mean_polarity'
                    serp_comps[sid][pol_key] = tmp[comp_key + '_mean_polarity']

    serp_comps_df = pd.DataFrame.from_dict(serp_comps, orient='index')
    serp_comps_df.index.name = 'id'
    serp_df = serp_df.merge(serp_comps_df, on='id')
    serp_df.reported_location = serp_df.reported_location.astype('category')

    cols_to_compare = []
    for link_type in link_types:
        for top_domain in set(all_domains):
            for prefix in [
                '_full_domain_frac_', '_top_three_domain_frac_',
                '_top_domain_frac_',
            ]:
                cols_to_compare.append(link_type + prefix + top_domain)
        for col in [
            '_full_mean_polarity', '_top_three_mean_polarity', '_top_mean_polarity',
        ]:
            cols_to_compare.append(link_type + col)
        cols_to_compare.append(link_type +  '_avg_jacc')
        cols_to_compare.append('has_' + link_type)
    serp_df = serp_df.fillna({
        col: 0 for col in cols_to_compare
    })

    urban_rows = serp_df[(serp_df['urban_rural_code'] == 1) | (serp_df['urban_rural_code'] == 2)]
    rural_rows = serp_df[(serp_df['urban_rural_code'] == 5) | (serp_df['urban_rural_code'] == 6)]
    for col in cols_to_compare:
        try:
            x = list(urban_rows[col])
            y = list(rural_rows[col])
        except KeyError:
            # print('Skipping {} bc KeyError'.format(col))
            continue
        if not x and not y:
            # print('Skipping {} bc Two empty lists'.format(col))
            continue
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        _, pval = ttest_ind(x, y, equal_var=False)
        if mean_x == mean_y:
            print('Exactly equal means for {}'.format(col))
            continue
        elif mean_x > mean_y:
            larger, smaller = mean_x, mean_y
            winner, loser = 'urban', 'rural'
        else:
            larger, smaller = mean_y, mean_x
            winner, loser = 'rural', 'urban'

        mult_increase = round(larger / smaller, 2)
        if mult_increase > 1.1 or pval <= 0.05:
            if pval <= 0.05:
                print('***')
            print(col)
            print('{} > {} by {}x ({} > {}), pval: {}, n: {}'.format(
                winner, loser, mult_increase, round(larger, 4), round(smaller, 4), pval, len(x) + len(y)
            ))
        
    


def parse():
    """parse args"""
    parser = argparse.ArgumentParser(description='Perform anlysis.')
    parser.add_argument(
        '--db', help='Name of the database', default='tmp/test_5kw_1loc.db')
    

    args = parser.parse_args()
    main(args)

parse()

