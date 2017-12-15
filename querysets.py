import pandas as pd
from pytrends.request import TrendReq

NUM_KEYWORDS = 3

KEYWORD_CSV = 'top_news_queries_20171029.csv'

from constants import POPULAR_CATEGORIES

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
    
    keyword_objs = []
    for cid in POPULAR_CATEGORIES:
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

TRENDING = [
    'bitcoin price',
    'al franken',
    'california fires',
    'ryan shazier',
    'eagles',
    'college football playoff',
    'michael flynn',
    'gomer pyle',
    'matt lauer',
    'gertrude jekyll',
]

PROCON_A_TO_Z = [
    'Topics A-Z',
    'Abortion',
    'ACLU - Good or Bad?',
    'Alternative Energy vs. Fossil Fuels',
    'Animal Testing',
    'Big Three Auto Bailout',
    'Born Gay? Origins of Sexual Orientation',
    'Cell Phones - Are They Safe?',
    'Churches and Taxes',
    'Climate Change',
    'College Education Worth It?',
    'College Football Playoffs',
    'Concealed Handguns',
    'Corporate Tax Rate & Jobs',
    'Cuba Embargo',
    'D.A.R.E. - Good or Bad?',
    'Death Penalty',
    'Drinking Age - Lower It?',
    'Drone Strikes Overseas',
    'Euthanasia & Assisted Suicide',
    'Felon Voting',
    'Gay Marriage',
    'Gold Standard',
    'Golf - Is It a Sport?',
    'Gun Control',
    'Illegal Immigration',
    'Insider Trading by Congress',
    'Israeli-Palestinian Conflict',
    'Local Elections - Santa Monica, 2014',
    'Medical Marijuana',
    'Milk - Is It Healthy?',
    'Minimum Wage',
    'Obamacare - Good or Bad?',
    'Obesity a Disease?',
    'Prescription Drug Ads Good?',
    'President Bill Clinton',
    'President Ronald Reagan',
    'Presidential Election, 2008',
    'Presidential Election, 2012',
    'Presidential Election, 2016',
    'Prostitution - Legalize?',
    'Right to Health Care?',
    'School Uniforms',
    'Social Networking - Good or Bad?',
    'Social Security Privatization',
    'Sports and Drugs',
    'Standardized Tests',
    'Tablets vs. Textbooks',
    'Teacher Tenure',
    'Under God in the Pledge',
    'US-Iraq War',
    'Vaccines for Kids',
    'Vegetarianism',
    'Video Games and Violence',
    'Voting Machines',
    'WTC Muslim Center',
]

CURATED = {
    'trending': TRENDING,
    'popular': from_trends_top_query_by_category(),
    'procon_popular': PROCON_POPULAR,
}