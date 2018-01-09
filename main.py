"""performs scraping"""
import sys
import platform
import argparse
from pprint import pprint
import time
import os

import pandas as pd
from querysets import (
    from_csv, from_trends_top_query_by_category,
    CURATED
)

import yagmail


# Xvfb :1 -screen 0 1024x768x16
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
        pop_df = pd.read_csv(POPULATION_FILENAME, thousands=',')
        # not in a loop b/c the key might change...
        location_df = location_df.merge(codes_df, on='GEOID')
        location_df = location_df.merge(income_df, on='GEOID')
        location_df = location_df.merge(pol_df, on='GEOID')
        location_df = location_df.merge(pop_df, on='GEOID')
    return location_df

def main(args):
    """main driver"""
    test = False
    dbname = './tmp/{}_{}_{}'.format(
        args.comparison, args.num_locations, args.query_source)
    if args.query_source == 'trends':
        keyword_objs = from_trends_top_query_by_category()
    elif args.query_source == 'csv':
        keyword_objs = from_csv()
    elif args.query_source == 'test':
        test = True
        keyword_objs = [{
            'keyword': 'weather',
            'category': args.query_source,
        },]
    elif args.query_source == 'all':
        keyword_objs = []
        for query_source in ['procon_popular', 'trending', ]:
            keywords = CURATED[query_source]
            keyword_objs += [
                {
                    'keyword': keyword,
                    'category': query_source,
                } for keyword in keywords
            ]
        keyword_objs += CURATED['popular']
    elif args.query_source == 'expanded':
        keyword_objs = []
        keywords = CURATED['procon_a_to_z']
        keyword_objs += [
            {
                'keyword': keyword,
                'category': args.query_source,
            } for keyword in keywords
        ]
        keyword_objs += from_trends_top_query_by_category(15)
    elif args.query_source == 'extra':
        keyword_objs = []
        for query_source in ['top_insurance', 'top_loans', 'top_symptoms', ]:
            keywords = CURATED[query_source] 
            keyword_objs += [
                {
                    'keyword': keyword,
                    'category': query_source,
                } for keyword in keywords
            ]
    else:
        keywords = CURATED[args.query_source]
        keyword_objs = [
            {
                'keyword': keyword,
                'category': args.query_source,
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
    # config.set('mobile_emulation', True)
    config.set('database_name', dbname)
    if args.save_html:
        config.set('save_html', True)
    else:
        config.set('save_html', False)
    config.set('use_control', False)
    location_df = load_locations()
    locations = []

    if args.comparison == 'test':
        # counties = ['Cook County', 'Santa Clara County']
        # for county in counties:
        #     for _, row in location_df[location_df.NAME == county].iterrows():
        #         locations.append({
        #             'engine': 'google',
        #             'latitude': row.INTPTLAT,
        #             'longitude': row.INTPTLONG,
        #             'urban_rural_code': row[URBAN_RURAL_COL],
        #             'median_income': row[MEDIAN_INCOME_COL],
        #             'percent_dem': row[VOTING_COL],
        #             'population_estimate': row[POPULATION_COL],
        #             'name': row.NAME
        #         })
        locations.append({
            'engine': 'google',
            'latitude': 41.8781,
            'longitude': -87.6298,
            'urban_rural_code': 1,
            'median_income': 0,
            'percent_dem': 0,
            'population_estimate': 0,
            'name': 'almaden',
        })
    else:
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
            if args.comparison == 'population_weighted':
                sample = subset.sample(
                    n=args.num_locations, weights=subset.POP_ESTIMATE_2016)
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
    if not test:
        yag = yagmail.SMTP('nickmvincent.mailbot@gmail.com', os.environ['MAILBOT_PASSWORD'])
        start_contents = """
            About to run! In total, {} keywords will be searched across {} locations.
            At a rate of ~1 SERP/min, this will take approximately {} hours.
            Keep in mind that going over 28 hours may result in a longer term IP ban.
            Arguments are {}.
            """.format(
                a, b, estimated_time, args
            )
        yag.send('nickmvincent@gmail.com', 'Scrape starting', start_contents)

    try:
        scrap.run()
    except ValueError as err:
        new_dbname = 'take2' + dbname
        err_contents = ['Error: {}. Going to wait one hour and try again! Results will be in {}'.format(
            err, new_dbname )]
        if not test:
            yag = yagmail.SMTP('nickmvincent.mailbot@gmail.com', os.environ['MAILBOT_PASSWORD'])
            yag.send('nickmvincent@gmail.com', 'Scrape starting', err_contents)
        time.sleep(3600)
        config.set('database_name', new_dbname)
        scrap2 = serpscrap.SerpScrap()
        scrap2.init(config=config.get(), keywords=keyword_objs)
        scrap2.run()

    if not test:
        end_contents = ['you-geo-see main.py finished running! Arguments were: {}'.format(args)]
        yag = yagmail.SMTP('nickmvincent.mailbot@gmail.com', os.environ['MAILBOT_PASSWORD'])
        yag.send('nickmvincent@gmail.com', 'Scrape success', end_contents)

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
    parser.add_argument(
        '--save_html', help="""
        To save html or not.
        """, action='store_true')
    args = parser.parse_args()
    main(args)

parse()
