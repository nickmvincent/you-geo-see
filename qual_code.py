"""
This module is meant to provide a nice, terminal based interface for qualitatively coding the results!

From the terminal, one can parse through results that need coding, select a result, and save the results to csv.
"""
import sqlite3
import csv
import argparse


import pandas as pd
import numpy as np
from analysis import get_dataframes, prep_data

UGC_WHITELIST = [
    'en.wikipedia.org',
    'TweetCarousel',
    'www.facebook.com',
    'twitter.com',
    'www.yelp.com',
    'www.youtube.com',
    'www.imdb',
    'www.instagram.com',
]

DOMAINS_TO_CODE = [
    'TweetCarousel',
    'www.facebook.com',
    'twitter.com',
    'www.yelp.com',
    'www.youtube.com',
    'www.instagram.com',
    'www.linkedin.com',
]

def main(args):
    """main"""
    if args.db:
        data, serp_df = get_dataframes(args.db)
        data = prep_data(data)
    elif args.csv:
        pass
    else:
        print('Please provide a --db or --csv to load data from')
    # we want to group items together to do coding all at once
    results_to_code = data[data.domain.isin(DOMAINS_TO_CODE)]
    


    

def parse():
    """parse args"""
    parser = argparse.ArgumentParser(description='Perform anlysis.')

    parser.add_argument(
        '--new', action='store_true')
    parser.add_argument(
        '--db', help='Name of the database')
    parser.add_argument(
        '--csv', help='Name of the CSV')

    args = parser.parse_args()
    main(args)

parse()
