import sqlite3
from collections import defaultdict
import operator
import pandas as pd
from sklearn.metrics import jaccard_similarity_score

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


DBNAME =  './tmp/cities.db'

def main():
    conn = sqlite3.connect(DBNAME)
    select_query = (
        """
        SELECT * from serp INNER JOIN link on serp.id = link.serp_id;
        """
    )
    df = pd.read_sql_query(select_query, conn)
    conn.close()
    
    df = df.fillna({
        'isTweetCarousel': False,
        'isMapsPlaces': False,
        'isMapsLocations': False,
        'isNewsCarousel': False,
    })
    df.domain = df.domain.astype('category')
    location_set = df.reported_location.drop_duplicates()
    print(df.describe())
    print(df.domain.value_counts())
    for reported_location in location_set:
        filtered = df[df.reported_location == reported_location]
        print(filtered.describe())
        print(filtered.domain.value_counts())

    
    for serp_id in df[df.is_control == True].id:
        filtered = df[df.id == serp_id]
    # for query, loc_to_pages in query_to_loc_to_pages.items():
    #     links0 = [d['link'] for d in list(loc_to_pages.values())[0] if d['link_type'] == 'results' and d['link']]
    #     links1 = [d['link'] for d in list(loc_to_pages.values())[1] if d['link_type'] == 'results' and d['link']]
    #     print('For query {}, using jaccard to compare {} and {}'.format(query, *loc_to_pages.keys()))
    #     try:            
    #         jac = jaccard_similarity_score(links0, links1)
    #         print(jac)
    #         if jac != 1.0:
    #             for item0, item1 in zip(links0, links1):
    #                 print(item0, item1)
    #     except:
    #         print('mismatched number of results for the query "{}"'.format(query))

main()
