"""performs scraping"""
import sys
sys.path.append("C:/Users/Nick/Documents/GitHub/SerpScrap") # Adds higher directory to python modules path.
import pandas as pd
import serpscrap
import pprint

VERSION = 'chrome'  # chrome

def main():
    """main driver"""
    keywords = ['crm']
    config = serpscrap.Config()
    config.set('do_caching', False)

    if VERSION == 'chrome':
        print('Using chromedriver.exe')
        config.set('sel_browser', 'chrome')
        config.set('executable_path', 'C:/Users/Nick/Desktop/chromedriver.exe')
    else:
        print('using phantomjs.exe')
        config.set('executable_path', 'C:/Users/Nick/Desktop/phantomjs-2.1.1-windows/bin/phantomjs.exe')

    # config.set('use_own_ip', False)
    # config.set('proxy_file', 'proxy.txt')
    config.set('num_pages_for_keyword', 1)
    config.set('num_results_per_page', 20)
    config.set('dir_screenshot', './tmp/screenshots')
    config.set('database_name', './tmp/serpscrap')
    # config.set('mobile_user_agent', True)

    scrap = serpscrap.SerpScrap()
    print('scrap.init')
    scrap.init(config=config.get(), keywords=keywords)
    results = scrap.run()
    for result in results:
        pprint.pprint(result)
    results_df = pd.DataFrame(results)
    results_df.to_csv("output.csv")

main()