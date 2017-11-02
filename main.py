"""performs scraping"""
import sys
# Adds higher directory to python modules path.
sys.path.append("C:/Users/Nick/Documents/GitHub/SerpScrap")
import pandas as pd
import serpscrap
import pprint

VERSION = 'chrome'  # chrome

LOCATIONS = [
    {
        "city": "New York",
        "latitude": 40.7127837,
        "longitude": -74.0059413,
        "population": "8405837",
        "rank": "1",
        "state": "New York"
    },
    {
        "city": "Los Angeles",
        "latitude": 34.0522342,
        "longitude": -118.2436849,
        "population": "3884307",
        "rank": "2",
        "state": "California"
    },
]


def main():
    """main driver"""
    keywords = ['trump']
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
    config.set('database_name', './tmp/serpscrap')
    config.set('locations', LOCATIONS)
    # config.set('search_engines', ['bing',])

    scrap = serpscrap.SerpScrap()
    scrap.init(config=config.get(), keywords=keywords)
    results = scrap.run()
    # for result in results:
    #     pprint.pprint(result)
    results_df = pd.DataFrame(results)
    results_df.to_csv("output.csv")


main()
