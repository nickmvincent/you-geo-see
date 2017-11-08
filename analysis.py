import sqlite3
from collections import defaultdict
import operator
from sklearn.metrics import jaccard_similarity_score

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


DBNAME =  './tmp/chicago_neighborhoods2.db'

def main():
    conn = sqlite3.connect(DBNAME)
    conn.row_factory = dict_factory
    cur = conn.cursor()
    query_to_loc_to_pages = defaultdict(dict)
    for zipcode in ['60653', '60660']:
        domain_counter = defaultdict(int)
        select_query = (
            """
            SELECT * from serp INNER JOIN link on serp.id = link.serp_id WHERE reported_location LIKE '{}%'
            """
        )
        print('Here are rows in zip code %s' % zipcode)
        for row in cur.execute(select_query.format(zipcode)):
            if row['domain']:
                domain_counter[row['domain']] += 1
            if row['reported_location'] not in query_to_loc_to_pages[row['query']]:
                query_to_loc_to_pages[row['query']][row['reported_location']] = []
            query_to_loc_to_pages[row['query']][row['reported_location']].append(row)
        sorted_domain_counter = sorted(
            domain_counter.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_domain_counter)
    conn.close()
    for query, loc_to_pages in query_to_loc_to_pages.items():
        links0 = [d['link'] for d in list(loc_to_pages.values())[0] if d['link_type'] == 'results' and d['link']]
        links1 = [d['link'] for d in list(loc_to_pages.values())[1] if d['link_type'] == 'results' and d['link']]
        print('For query {}, using jaccard to compare {} and {}'.format(query, *loc_to_pages.keys()))
        try:            
            jac = jaccard_similarity_score(links0, links1)
            print(jac)
            if jac != 1.0:
                for item0, item1 in zip(links0, links1):
                    print(item0, item1)
        except:
            print('mismatched number of results for the query "{}"'.format(query))

main()
