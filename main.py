"""performs scraping"""
import sys
# Adds higher directory to python modules path.
sys.path.append("C:/Users/Nick/Documents/GitHub/SerpScrap")
import pandas as pd
import serpscrap
import time
import json
from pytrends.request import TrendReq


VERSION = 'chrome'
LOCATIONS = [
    # {},
    # {
    #     "city": "New York", 
    #     "growth_from_2000_to_2013": "4.8%", 
    #     "latitude": 40.7127837, 
    #     "longitude": -74.0059413, 
    #     "population": "8405837", 
    #     "rank": "1", 
    #     "state": "New York"
    # },     {
    #     "city": "New York", 
    #     "growth_from_2000_to_2013": "4.8%", 
    #     "latitude": 40.7127837, 
    #     "longitude": -74.0059413, 
    #     "population": "8405837", 
    #     "rank": "1", 
    #     "state": "New York"
    # }, 
    # {
    #     "city": "Los Angeles", 
    #     "growth_from_2000_to_2013": "4.8%", 
    #     "latitude": 34.0522342, 
    #     "longitude": -118.2436849, 
    #     "population": "3884307", 
    #     "rank": "2", 
    #     "state": "California"
    # },
    # {
    #     "city": "Jacksonville", 
    #     "growth_from_2000_to_2013": "14.3%", 
    #     "latitude": 30.3321838, 
    #     "longitude": -81.65565099999999, 
    #     "population": "842583", 
    #     "rank": "13", 
    #     "state": "Florida"
    # },
    # {
    #     "city": "Wichita", 
    #     "growth_from_2000_to_2013": "9.7%", 
    #     "latitude": 37.688889, 
    #     "longitude": -97.336111, 
    #     "population": "386552", 
    #     "rank": "49", 
    #     "state": "Kansas"
    # }, 
]

with open('chicago_neighborhoods.json') as data_file:    
    LOCATIONS = json.load(data_file)

NUM_LOCATIONS = 2
NUM_KEYWORDS = 10



def main():
    """main driver"""
    yearmonth = '201709'
    cid = 'fast_food_restaurants'
    pytrends = TrendReq(hl='en-US', tz=360)

    keywords = pytrends.top_charts(yearmonth, cid, geo='US')
    keywords = keywords.title.tolist()[:NUM_KEYWORDS]
    print(keywords)

    config = serpscrap.Config()
    config.set('do_caching', False)

    if VERSION == 'chrome':
        print('Using chromedriver.exe')
        config.set('sel_browser', 'chrome')
        config.set('executable_path', 'C:/Users/Nick/Desktop/chromedriver.exe')
    else:
        print('using phantomjs.exe')
        config.set('executable_path',
                   'C:/Users/Nick/Desktop/phantomjs-2.1.1-windows/bin/phantomjs.exe')

    # config.set('use_own_ip', False)
    # config.set('proxy_file', 'proxy.txt')
    config.set('num_pages_for_keyword', 1)
    config.set('num_results_per_page', 30)
    config.set('dir_screenshot', './tmp/screenshots')
    config.set('database_name', './tmp/chicago_neighborhoods')
    for location in LOCATIONS:
        location['engine'] = 'google'
    for location in LOCATIONS[:NUM_LOCATIONS]:
        config.set('search_instances', [location])    
        scrap = serpscrap.SerpScrap()
        scrap.init(config=config.get(), keywords=keywords)
        results = scrap.run()
        results_df = pd.DataFrame(results)
        results_df.to_csv("output.csv")


main()
