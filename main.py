"""performs scraping"""
import sys
import platform
import json
import csv
from pprint import pprint
import pandas as pd
from querysets import from_csv, from_trends_top_query_by_category, NUM_KEYWORDS


# Xvfb :1 -screen 1 1024x768x16
# export DISPLAY=:1


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
CODE_FILENAME = 'NCHSURCodes2013.csv'
# INCOME_FILENAME = '2015_household_median_incomes_by_county.csv'

def load_locations():
    """Loads locations from CSV or JSON"""
    if '.csv' in LOCATIONFILENAME:
        location_df = pd.read_csv(LOCATIONFILENAME)
        codes_df = pd.read_csv(CODE_FILENAME)
        location_df = location_df.merge(codes_df, on='GEOID')
    else:
        with open(LOCATIONFILENAME) as locationfile:
            location_df = pd.read_json(locationfile)
    return location_df



NUM_LOCATION_SAMPLES = 1
DBNAME = './tmp/test_{}kw_{}loc'.format(NUM_KEYWORDS, NUM_LOCATION_SAMPLES)

KEYWORD_SOURCE = 'trends'
def main():
    """main driver"""
    if KEYWORD_SOURCE == 'manual':
        keywords = [
            'tax bill', 'alabama senate',
            'al franken', 'impeach trump',
            'support trump'
        ]
        keyword_objs = [
            {
                'keyword': keyword,
                'category': 'manual_news',
            } for keyword in keywords
        ]
    elif KEYWORD_SOURCE == 'trends':
        keyword_objs = from_trends_top_query_by_category()
    elif KEYWORD_SOURCE == 'csv':
        keyword_objs = from_csv()    
    
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
    config.set('database_name', DBNAME)
    config.set('save_html', False)
    config.set('use_control', False)
    location_df = load_locations()
    locations = []
    for subset in [
        location_df[location_df['2013 urban-rural code'] < 3],
        # location_df[(location_df['2013 urban-rural code'] >= 3) & (location_df['2013 urban-rural code'] < 5)],
        location_df[location_df['2013 urban-rural code'] >= 5],
    ]:
        sample = subset.sample(n=NUM_LOCATION_SAMPLES)
        for _, row in sample.iterrows():
            locations.append({
                'engine': 'google',
                'latitude': row.INTPTLAT,
                'longitude': row.INTPTLONG,
                'code': row['2013 urban-rural code'],
                'name': row.NAME,
            })
    pprint(locations)
    config.set('search_instances', locations)
    scrap = serpscrap.SerpScrap()
    scrap.init(config=config.get(), keywords=keyword_objs)
    scrap.run()
    # results_df = pd.DataFrame(results)
    # results_df.to_csv("output.csv")


main()
