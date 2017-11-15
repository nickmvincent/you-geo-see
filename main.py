"""performs scraping"""
import sys
import platform
import json
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


DBNAME = './tmp/curated_counties'
VERSION = 'chrome'
LOCATIONFILENAME = 'curated_counties.json'

with open(LOCATIONFILENAME) as locationfile:
    LOCATIONS = json.load(locationfile)

NUM_LOCATIONS = 1
NUM_KEYWORDS = 1


def main():
    """main driver"""
    # yearmonth = '201709'
    # cid = 'fast_food_restaurants'
    # pytrends = TrendReq(hl='en-US', tz=360)
    # keywords = pytrends.top_charts(yearmonth, cid, geo='US')
    # keywords = keywords.title.tolist()[:NUM_KEYWORDS]
    keywords = ['trump', ]
    print(keywords)

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
    for location in LOCATIONS:
        location['engine'] = 'google'
    config.set('search_instances', LOCATIONS[:NUM_LOCATIONS])
    scrap = serpscrap.SerpScrap()
    scrap.init(config=config.get(), keywords=keywords)
    results = scrap.run()
    # results_df = pd.DataFrame(results)
    # results_df.to_csv("output.csv")


main()
