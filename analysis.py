import sqlite3
import operator
from pprint import pprint
from collections import defaultdict
from string import ascii_lowercase

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from pyxdameraulevenshtein import damerau_levenshtein_distance


DBNAME = './tmp/collection1.db'


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
    control_dict = control_domains_col.value_counts().to_dict()
    ret = {}
    for key, val in domains_dict.items():
        ret[key] = val / len(domains_col)
    for key, val in control_dict.items():
        ret[key] = (ret.get(key, 0) + val / len(control_domains_col)) / 2
    return ret

def compute_serp_features(links, control_links, domains_col, control_domains_col):
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

    return {
        'control_jaccard': jaccard_similarity(
            links, control_links
        ),
        'control_edit': damerau_levenshtein_distance(
            string, control_string
        ),
        'full': {
            'domain_fracs': calc_domain_fracs(domains_col, control_domains_col)
        },
        'top_three': {
            'domain_fracs': calc_domain_fracs(
                domains_col.iloc[:3], control_domains_col.iloc[:3])
        },
        'top': {
            'domain_fracs': calc_domain_fracs(
                domains_col.iloc[:1], control_domains_col.iloc[:1])
        },
    }


def analyze_subset(data, location_set):
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
        control = results[results.is_control == 1]
        treatment = results[results.is_control == 0]
        control_links = list(control.link)
        links = list(treatment.link)
        if not control_links:
            print('Empty control links for loc {}'.format(loc))
            continue
        if not links:
            print('Empty links, continue')
            continue
        d[loc] = {}        
        d[loc]['links'] = links
        d[loc]['control_links'] = control_links
        d[loc]['computed'] = compute_serp_features(links, control_links, treatment.domain, control.domain)
        d[loc]['serp_id'] = results.iloc[0].serp_id

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

    data.loc[tweet_mask, 'link'] = 'TweetCarousel'
    data.loc[tweet_mask, 'domain'] = 'TweetCarousel'
    data.loc[news_mask, 'link'] = 'NewsCarousel'
    data.loc[news_mask, 'domain'] = 'NewsCarousel'
    data.loc[kp_mask, 'link'] = 'KnowledgePanel' 
    data.loc[kp_mask, 'domain'] = 'KnowledgePanel' 

    data.domain = data.domain.astype('category')
    return data


def get_dataframes():
    """
    Get rows from the db and convert to dataframes
    """
    conn = sqlite3.connect(DBNAME)
    select_results = (
        """
        SELECT * from serp INNER JOIN link on serp.id = link.serp_id;
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


def main():
    """Do analysis"""
    data, serp_df = get_dataframes()
    data = prep_data(data)

    location_set = data.reported_location.drop_duplicates()
    query_set = data['query'].drop_duplicates()

    link_types = [
        'results',# 'tweets', 'news'
    ]

    print(data.domain.value_counts())
    top_ten_domains = list(data.domain.value_counts().to_dict().keys())[:10]
    print(top_ten_domains)

    serp_comps = {}
    
    for link_type in link_types:
        for query in query_set:
            print(link_type, query)
            filtered = data[data.link_type == link_type]
            filtered = filtered[filtered['query'] == query]
            d = analyze_subset(filtered, location_set)

            for loc, vals in d.items():
                sid = vals['serp_id']
                tmp = d[loc]['computed']
                dist_sum, jacc_sum, count = 0, 0, 0
                for _, metrics in vals['comparisons'].items():
                    dist_sum += metrics['edit']
                    jacc_sum += metrics['jaccard']
                    count += 1
                avg_edit = dist_sum / count
                avg_jacc = jacc_sum / count
                tmp['avg_edit'] = avg_edit
                tmp['avg_jaccard'] = avg_jacc
                serp_comps[sid] = {
                    'avg_edit': avg_edit,
                    'avg_jacc': avg_jacc,
                    'id': sid,
                }
                for comp_key in ['full', 'top', 'top_three', ]:
                    domain_fracs = tmp[comp_key]['domain_fracs']
                    for domain_string, frac, in domain_fracs.items():
                        for top_domain in top_ten_domains:
                            if domain_string == top_domain:
                                concat_key = '_'.join(
                                    [comp_key, 'domain_frac', domain_string]
                                )
                                serp_comps[sid][concat_key] = frac

    serp_comps_df = pd.DataFrame.from_dict(serp_comps, orient='index')
    serp_comps_df.index.name = 'id'
    serp_df = serp_df.merge(serp_comps_df, on='id')
    serp_df.reported_location = serp_df.reported_location.astype('category')

    # TODO: tukey
    urban_rows = serp_df[(serp_df['urban_rural_code'] == 1) | (serp_df['urban_rural_code'] == 2)]
    rural_rows = serp_df[(serp_df['urban_rural_code'] == 5) | (serp_df['urban_rural_code'] == 6)]
    cols_to_compare = []
    for top_domain in top_ten_domains:    
        for prefix in [
            'full_domain_frac_', 'top_three_domain_frac_',
            'top_domain_frac_',
        ]:
            cols_to_compare.append(prefix + top_domain)
    for col in cols_to_compare:
        x = list(urban_rows[col])
        y = list(rural_rows[col])
        print(col)
        _, pval = ttest_ind(x, y, equal_var=False)
        print('Means', np.mean(x), np.mean(y), 'pval', pval)
    


main()

