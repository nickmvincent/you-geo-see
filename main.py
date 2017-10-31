import serpscrap
import pprint

keywords = ['berlin']

config = serpscrap.Config()
config.set('sel_browser', 'chrome')
config.set('executable_path', 'C:\\Users\\Nick\\Desktop\\chromedriver.exe')

scrap = serpscrap.SerpScrap()
scrap.init(config=config.get(), keywords=keywords)
results = scrap.as_csv('output')