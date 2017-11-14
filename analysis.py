import sqlite3
from collections import defaultdict
import operator
import pandas as pd
import editdistance

def jaccard_similarity(x,y):
    """
    set implementation of jaccard similarity
    """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)


DBNAME = './tmp/cities.db'

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
    df.loc[df.link.isnull(), 'link'] = '' 

    df.domain = df.domain.astype('category')
    print(df.describe())

    def variation_in_control():
        controls = df[df.is_control == True]
        first_results = {}
        prev_dist = {
            'results': 0,
            'tweets': 0,
            'news': 0,
        }
        for index, serp_id in enumerate(controls.serp_id):
            filtered = df[df.serp_id == serp_id]
            try:
                results = {
                    'results': filtered[filtered.link_type == 'results'].link,
                    'tweets': filtered[filtered.link_type == 'tweets'].link,
                    'news': filtered[filtered.link_type == 'news'].link,
                }
                if index == 0:
                    first_results = results
                for key, links in results.items():
                    dist = editdistance.eval(list(first_results[key]), list(links))
                    jac = jaccard_similarity(list(first_results[key]), list(links))
                    if dist != 0 and dist != prev_dist[key]:
                        prev_dist[key] = dist
                        print('dist', key, dist)
                        print(jac)
                        for item0, item1 in zip(first_results[key], links):
                            if item0 != item1:
                                print(item0)
                                print(item1, '\n====')
                    
            except Exception as err:
                print('mismatch')
                print(err)
    variation_in_control()
    location_set = df.reported_location.drop_duplicates()
    query_set = df['query'].drop_duplicates()
    link_type_set = df.link_type.drop_duplicates()

    def analyze_subset(filtered, location_set):
        """
        A subset consists of results of a certain TYPE for a certain QUERY
        """
        # print(filtered.domain.value_counts())

        # d holds the results and editdistances
        # good variable naming was skipped for coding convenience. refactor later?
        d = {}
        for loc in location_set:
            d[loc] = {}
            d[loc]['results'] = list(filtered[filtered.reported_location == loc].link)

        for loc in location_set:
            for comparison_loc in location_set:
                if loc == comparison_loc:
                    continue
                d[loc][comparison_loc] = {}

                d[loc][comparison_loc]['edit'] = \
                    editdistance.eval(
                        d[loc]['results'], 
                        d[comparison_loc]['results']
                    )
                d[loc][comparison_loc]['jaccard'] = \
                    jaccard_similarity(
                        d[loc]['results'], 
                        d[comparison_loc]['results']
                    )
        return d
    
    links_to_analyze = [
        'results', 'tweets', 'news'
    ]
    for link_type in links_to_analyze:
        for query in query_set:
            print(link_type, query)
            filtered = df[df.link_type == link_type]
            filtered = filtered[filtered['query'] == query]
            filtered = filtered[filtered.is_control == False]
            d = analyze_subset(filtered, location_set)

            for loc, vals in d.items():
                dist_sum, jacc_sum, count = 0, 0, 0
                for comparison_loc, metrics in vals.items():
                    # consider implementing an actual class, that would be the right thing to do here...
                    if comparison_loc == 'results':
                        continue
                    dist_sum += metrics['edit']
                    jacc_sum += metrics['jaccard']
                    count += 1
                avg_dist = dist_sum / count
                avg_jacc = jacc_sum / count
                d[loc]['avg_edit'] = avg_dist
                d[loc]['avg_jacc'] = avg_jacc
                print(loc, avg_dist, avg_jacc)
            print(list(d.items())[0:2])
                    

main()
