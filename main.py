"""performs scraping"""
import serpscrap
import pprint

#VERSION = 'chrome'
VERSION = 'phantom'

def main():
    """main driver"""
    keywords = ['coffee near']
    config = serpscrap.Config()
    config.set('do_caching', False)

    if VERSION == 'chrome':
        print('Using chromedriver.exe')
        config.set('sel_browser', 'chrome')
        config.set('executable_path', 'C:\\Users\\Nick\\Desktop\\chromedriver.exe')
    else:
        print('using phantomjs.exe')
        config.set('executable_path', 'C:\\Users\\Nick\\Desktop\\phantomjs-2.1.1-windows\\bin\\phantomjs.exe')

    config.set('use_own_ip', False)
    config.set('proxy_file', 'proxy.txt')

    scrap = serpscrap.SerpScrap()
    print('scrap.init')
    scrap.init(config=config.get(), keywords=keywords)
    results = scrap.as_csv('output')

main()
