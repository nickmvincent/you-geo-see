"""performs scraping"""
import sys
import platform
import argparse
from pprint import pprint
import pandas as pd
from querysets import (
    from_csv, from_trends_top_query_by_category,
    CURATED
)


# Xvfb :1 -screen 1 1024x768x16
# export DISPLAY=:1


# platform specific globals
if platform.system() == 'Windows':
    sys.path.append("C:/Users/Nick/Documents/GitHub/SerpScrap")
    CHROME_PATH = 'C:/Users/Nick/Desktop/chromedriver.exe'
    PHANT_PATH = 'C:/Users/Nick/Desktop/phantomjs-2.1.1-windows/bin/phantomjs.exe'
else:
    sys.path.append("../SerpScrap")
    CHROME_PATH = '../chromedriver'
    PHANT_PATH = '../phantomjs'
import serpscrap


VERSION = 'chrome'
LOCATIONFILENAME = '2017_gaz_counties_all.csv'
# Source: 
CODE_FILENAME = 'NCHSURCodes2013.csv'
URBAN_RURAL_COL = '2013 urban-rural code'

# Source:
INCOME_FILENAME = '2015_household_median_incomes_by_county.csv'
MEDIAN_INCOME_COL = 'HC01_EST_VC13'
# Source:

POLITICAL_FILENAME = '2016_US_County_Level_Presidential_Results.csv'
VOTING_COL = 'per_dem'
# Source:
POPULATION_FILENAME = '2016_estimated_population.csv'
POPULATION_COL = 'POP_ESTIMATE_2016'
# Source: https://www.ers.usda.gov/data-products/county-level-data-sets/county-level-data-sets-download-data/



def load_locations():
    """Loads locations from CSV or JSON"""
    if '.csv' in LOCATIONFILENAME:
        location_df = pd.read_csv(LOCATIONFILENAME)
        codes_df = pd.read_csv(CODE_FILENAME)
        income_df = pd.read_csv(INCOME_FILENAME)
        pol_df = pd.read_csv(POLITICAL_FILENAME)
        pop_df = pd.read_csv(POPULATION_FILENAME)
        # not in a loop b/c the key might change...
        location_df = location_df.merge(codes_df, on='GEOID')
        location_df = location_df.merge(income_df, on='GEOID')
        location_df = location_df.merge(pol_df, on='GEOID')
        location_df = location_df.merge(pop_df, on='GEOID')
    return location_df

def main(args):
    """main driver"""
    dbname = './tmp/test_{}loc_{}kw'.format(
        args.num_locations, args.query_source)
    if args.query_source == 'trends':
        keyword_objs = from_trends_top_query_by_category()
    elif args.query_source == 'csv':
        keyword_objs = from_csv()
    else:
        keywords = CURATED[args.query_source]
        keyword_objs = [
            {
                'keyword': keyword,
                'category': 'manual_news',
            } for keyword in keywords
        ]
    
    print(keyword_objs)

    config = serpscrap.Config()
    config.set('do_caching', False)

    if VERSION == 'chrome':
        config.set('sel_browser', 'chrome')
        config.set('executable_path', CHROME_PATH)
    else:
        config.set('executable_path',
                   PHANT_PATH)

    # config.set('use_own_ip', False)
    # config.set('proxy_file', 'proxy.txt')
    config.set('num_pages_for_keyword', 1)
    config.set('num_results_per_page', 30)  # overshoots actual number of results per page
    config.set('screenshot', False)
    config.set('database_name', dbname)
    config.set('save_html', False)
    config.set('use_control', False)
    location_df = load_locations()
    locations = []

    if args.comparison == 'urban-rural':
        subsets = [
            location_df[location_df[URBAN_RURAL_COL] == 1],
            location_df[location_df[URBAN_RURAL_COL] == 6],
        ]
    elif args.comparison == 'income' or args.comparison == 'voting':
        if args.comparison == 'income':
            sort_col = MEDIAN_INCOME_COL
        else:
            sort_col = VOTING_COL
        print('Going to sort by {}'.format(sort_col))
        location_df = location_df.sort_values(by=[sort_col])
        print(location_df)
        lower_set = location_df.head(args.num_locations)
        upper_set = location_df.tail(args.num_locations)
        subsets = [lower_set, upper_set]
    else:
        subsets = [location_df]
    for subset in subsets:
        sample = subset.sample(n=args.num_locations)
        for _, row in sample.iterrows():
            locations.append({
                'engine': 'google',
                'latitude': row.INTPTLAT,
                'longitude': row.INTPTLONG,
                'urban_rural_code': row[URBAN_RURAL_COL],
                'median_income': row[MEDIAN_INCOME_COL],
                'percent_dem': row[VOTING_COL],
                'population_estimate': row[POPULATION_COL],
                'name': row.NAME
            })
    pprint(locations)
    config.set('search_instances', locations)
    scrap = serpscrap.SerpScrap()
    scrap.init(config=config.get(), keywords=keyword_objs)
    a, b = len(keyword_objs), len(locations)
    estimated_time = round(a * b / 60, 2)
    print("""
        About to run! In total, {} keywords will be searched across {} locations.
        At a rate of ~1 SERP/min, this will take approximately {} hours.
        Keep in mind that going over 28 hours may result in a longer term IP ban.
        """.format(
            a, b, estimated_time
        )
    )
    scrap.run()
    # results_df = pd.DataFrame(results)
    # results_df.to_csv("output.csv")

def parse():
    """parse args"""
    parser = argparse.ArgumentParser(description='Collect SERPs')
    parser.add_argument(
        '--comparison', help='What comparison to do', default=None)
    parser.add_argument(
        '--num_locations', help="""
        Number of location to sample per subset
        If this is 25, you'll get 25 urban and 25 rural samples
        If this is 25 and comparison is None, you'll just get 25 sample
        Default is 25
        """, type=int, default=25)
    parser.add_argument(
        '--query_source', help="""
        Queries to search
        check out querysets.py for more info
        """, default='trends')
    args = parser.parse_args()
    main(args)

parse()
