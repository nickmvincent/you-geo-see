import pandas as pd

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
        'foods', 'fashion_labels', 'auto_companies',
        'financial_companies',
    ]
    pol_cats = [
        'politicians', 'governmental_bodies',
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