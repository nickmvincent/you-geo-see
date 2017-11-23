"""performs scraping"""
import sys
import platform
import json
import csv
from pprint import pprint
import pandas as pd
from pytrends.request import TrendReq


if platform.system() == 'Windows':
    sys.path.append("C:/Users/Nick/Documents/GitHub/SerpScrap")
    CHROME_PATH = 'C:/Users/Nick/Desktop/chromedriver.exe'
    PHANT_PATH = 'C:/Users/Nick/Desktop/phantomjs-2.1.1-windows/bin/phantomjs.exe'
else:
    sys.path.append("../SerpScrap")
    CHROME_PATH = '../chromedriver'
    PHANT_PATH = '../phantomjs'
import serpscrap


DBNAME = './tmp/test1'
VERSION = 'chrome'
LOCATIONFILENAME = '2017_gaz_counties_06.csv'
CODE_FILENAME = 'NCHSURCodes2013.csv'

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


NUM_KEYWORDS = 1
NUM_LOCATION_SAMPLES = 1

def main():
    """main driver"""
    econ_cats = [
        'fast_food_restaurants', 'retail_companies',
        # 'foods', 'fashion_labels', 'auto_companies',
        # 'financial_companies',
    ]
    pol_cats = [
        # 'politicians', 'governmental_bodies',
    ]
    keyword_objs = []
    for cats in [econ_cats, pol_cats]:
        for cid in cats:
            print(cid)
            yearmonth = '2016'
            pytrends = TrendReq(hl='en-US', tz=360)
            keywords = pytrends.top_charts(yearmonth, cid=cid, geo='US')
            keywords = keywords.title.tolist()[:NUM_KEYWORDS]
            keyword_objs += [
                {'keyword': x, 'category': cid} for x in keywords
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
    config.set('database_name', DBNAME)
    config.set('save_html', True)
    location_df = load_locations()
    locations = []
    for subset in [
        location_df[location_df['2013 urban-rural code'] < 3],
        location_df[(location_df['2013 urban-rural code'] >= 3) & (location_df['2013 urban-rural code'] < 5)],
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
