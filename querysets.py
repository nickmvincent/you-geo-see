import pandas as pd
from pytrends.request import TrendReq

NUM_KEYWORDS = 5

KEYWORD_CSV = 'top_news_queries_20171029.csv'

def from_csv():
    """Loads keywords from CSV"""
    datf = pd.read_csv(KEYWORD_CSV)
    keywords = list(datf[datf.columns[0]])
    keyword_objs = [
        {
            'keyword': keyword,
            'category': 'manual_news',
        } for keyword in keywords
    ]
    return keyword_objs


def from_trends_top_query_by_category():
    """
    Get a set of keyword objects by quering Google Trends
    Each keyword obj is a dict with keys: keyword, category
    """
    econ_cats = [
        'fast_food_restaurants', 'retail_companies',
        'foods', 'fashion_labels', 'financial_companies',
    ]
    pol_cats = [
        'politicians',
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
    return keyword_objs


CONTROVERSIAL = [
    'Progressive Tax',
    'Impose A Flat Tax',
    'End Medicaid',
    'Affordable Health And Care Act',
    'Fluoridate Water',
    'Stem Cell Research',
    'Andrew Wakefield Vindicated',
    'Autism Caused By Vaccines',
    'US Government Loses AAA Bond Rate',
    'Is Global Warming Real',
    'Man Made Global Warming Hoax',
    'Nuclear Power Plants',
    'Offshore Drilling',
    'Genetically Modified Organi',
    'Trump',
]

PROCON_POPULAR = [
    'medical marijuana',
    'gun control',
    'animal testing',
    'death penalty',
    'school uniforms',
    'drinking age',
    'minimum wage',
    'euthanasia',
    'illegal immigration',
    'abortion',
]

# I manually picked political and financial topics
TRENDS_CURATED = [
    'Billy Bush',
    'SNL',
    'Brock Turner',
    'Brias Ross',
    'Tax Reform Bill 2017',
    'Michael Flynn',
    'tax bill',
    'finance',
    'kate steinle',
    'rex tillerson',
    'john conyers',
]

CURATED = {
    'controversial': CONTROVERSIAL,
    'dec3_nov30': TRENDS_CURATED,
    'procon_popular': PROCON_POPULAR,
}