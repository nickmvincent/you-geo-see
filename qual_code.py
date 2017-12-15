"""
This module is meant to provide a nice, terminal based interface for qualitatively coding the results!

From the terminal, one can parse through results that need coding, select a result, and save the results to csv.
"""
import sqlite3
import csv
import argparse
import os

import tweepy
import pandas as pd
import numpy as np
from analysis import get_dataframes, prep_data


# axes 1: indivudal or organization
# axes 2: corporate, journalistic, political, or none?

# i = individual
# o = organization

# 
# u = unaffiliated

SHORTHAND = {
    'i': 'individual',
    'o': 'organization',
    'n': 'none',
    'c': 'corporate',
    'j': 'journalistic',
    'p': 'political',
}


CODES = {
    'individual-none': 1,
    'individual-journalistic': 2,
    'individual-corporate': 3,
    'individual-political': 4,
    'organization-journalistic': 5,
    'organization-corporate': 6,
    'organization-political': 7,
    'organization-none': 8,
}

# this should probably go somewhere else?
UGC_WHITELIST = [
    'en.wikipedia.org',
    'TweetCarousel',
    'facebook.com',
    'twitter.com',
    'youtube.com',
    'imdb',
    'instagram.com',
]

DOMAINS_TO_CODE = [
    'facebook.com',
    'youtube.com',
    'instagram.com',
    'linkedin.com',
]
TWITTER_DOMAIN = 'twitter.com'

# todo strip domains
# see where most of these Twitter Links go...
# todo hit Twitter API

START_STRING = '.com/'
START_STRING_LENGTH = len(START_STRING)

def strip_twitter_screename(link):
    """
    Args: link - a link a twitter page or status
    Returns: the screename of the page/status
    """
    username_starts = link.find(START_STRING) + START_STRING_LENGTH      
    if '/status/' in link:
        username_ends = link.find('/status/')
    else:
        username_ends = link.find('?')
    return link[username_starts:username_ends]
    


def main(args):
    """main"""
    if args.db:
        if 'dbs' in args.db:
            args.db = args.db[4:]
        data, _ = get_dataframes(args.db)
        data = prep_data(data)
    elif args.csv:
        if 'csvs' in args.csv:
            args.csv = args.csv[5:]
        data = pd.read_csv(args.csv)
    else:
        print('Please provide a --db or --csv to load data from')
    if args.twitter:
        auth = tweepy.OAuthHandler(os.environ['twitter_consumer_key'], os.environ['twitter_consumer_secret'])
        auth.set_access_token(os.environ['twitter_access_token_key'], os.environ['twitter_access_token_secret'])
        api = tweepy.API(auth)
        DOMAINS_TO_CODE.append(TWITTER_DOMAIN)
    else:
        api = None
    # we want to group items together to do coding all at once
    for domain in DOMAINS_TO_CODE:
        df_in_domain = data[data.domain == domain]
        links = df_in_domain.link.drop_duplicates()
        n = len(links)
        print('There are {} links to code for the domain {}'.format(
            n, domain
        ))
        for link in links:
            if domain == 'linkedin.com':
                if '/company/' in link:
                    print('auto coding as a LI company page')
                    print(link)
                    data.loc[data.link == link, 'code'] = CODES['organization-corporate']
                    continue
            print(link)
            code = input()
            if code == 'more':
                print('Showing more!')
                snippet = df_in_domain[df_in_domain.link == link].iloc[0].snippet
                print('Snippet:', snippet)
                if domain == TWITTER_DOMAIN:
                    screen_name = strip_twitter_screename(link)
                    user_obj = api.get_user(screen_name=screen_name)
                    print('Bio:', user_obj.description)
                    print('Location', user_obj.location)
                    print('Verified?', user_obj.verified)

                code = input()
            while len(code) != 2:
                print('Please enter a 2 digit code with the following format')
                print('First character: i (individual) or o (organization')
                print('Second character: j (journalistic), c (corporate), p (political), or n (none)')
                code = input()
            code_num = SHORTHAND[code[0]] + '-' + SHORTHAND[code[1]]
            data.loc[data.link == link, 'code'] = code_num
    csv_name = 'csvs/' + args.db.replace('.db', '_coded.csv')
    data.to_csv(csv_name)


def parse():
    """parse args"""
    parser = argparse.ArgumentParser(description='Perform anlysis.')

    parser.add_argument(
        '--new', action='store_true')
    parser.add_argument(
        '--db', help='Name of the database')
    parser.add_argument(
        '--csv', help='Name of the CSV')
    parser.add_argument(
        '--twitter', action='store_true', help='include Twitter links or not'
    )

    args = parser.parse_args()
    main(args)

parse()
