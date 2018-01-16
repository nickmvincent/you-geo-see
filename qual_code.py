"""
This module is meant to provide a nice, terminal based interface for
qualitatively coding the results!

From the terminal, one can parse through results that need coding, select a result,
and save the results to csv.
"""
import argparse
import os
from datetime import datetime

import tweepy
import pandas as pd
from data_helpers import get_dataframes, load_coded_as_dicts, prep_data

# axes 1: creative effort?
# axes 2: made outside professional practice?
# axes 3: indivudal or organization
# axes 4: corporate, journalistic, political, or none?

SHORTHAND = {
    'creative': {
        't': 'true',
        'f': 'false',
    },
    'outside_prof': {
        't': 'true',
        'f': 'false',
    },
    'author': {
        'i': 'individual',
        'o': 'organization',
        'b': 'bot',
    },
    'type': {
        'n': 'nonprofit',
        'c': 'celebrity',
        '$': 'corporate',
        'j': 'journalistic',
        'p': 'political',
        'z': 'other',        
    }
}


DOMAINS_TO_CODE = [
    'facebook.com',
    'youtube.com',
    'instagram.com',
    'linkedin.com',
    'yelp.com',
    'pinterest.com',
    'tripadvisor.com',
]
TWITTER_DOMAIN = 'twitter.com'

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
    elif '?' in link:
        username_ends = link.find('?')
    else:
        username_ends = len(link)
    return link[username_starts:username_ends]

def code_item(domain, link=None, screen_name=None, snippet=None, api=None):
    """
    This function helps a qualitative coder apply a code to a single link
    or twitter screen name
    """
    if domain == TWITTER_DOMAIN and screen_name:
        user_obj = api.get_user(screen_name=screen_name)
        print('Bio:', user_obj.description)
        print('Location', user_obj.location)
        print('Verified?', user_obj.verified)
    code = input()
    if code == 'snippet':
        print('Showing snippet!')
        print('Snippet:', snippet)
        code = input()
    while (
        len(code) != 4 or
        code[0] not in SHORTHAND['creative'].keys() or
        code[1] not in SHORTHAND['outside_prof'].keys() or
        code[2] not in SHORTHAND['author'].keys() or
        code[3] not in SHORTHAND['type'].keys()
    ):
        print('Please enter 4 digits like so:')
        print('First second character: t (true) or f (false) regarding creative effort')
        print('Second second character: t (true) or f (false) regarding "outside professional practice"')
        print('Third character: i (individual), o (organization, or b (bot)')
        print('Fourth character: j (journalistic), c (corporate), p (political), n (nonprofit), or z (other)')
        print(link)
        code = input()
    code_str = SHORTHAND[code[0]] + '-' + SHORTHAND[code[1]]
    return code_str
    

def quote(x):
    return '"' + x + '"'

def main(args):
    """main"""
    ret = []
    sources = args.db if args.db else [args.csv]
    domain_to_df = {}
    for source in sources:
        if args.db:            
            data, _ = get_dataframes(source)
            data = prep_data(data)
        elif args.csv:
            data = pd.read_csv(source, encoding='utf-8')
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
        link_codes_file = 'link_codes.csv'
        twitter_user_codes_file = 'twitter_user_codes.csv'
        link_codes, twitter_user_codes = load_coded_as_dicts(link_codes_file, twitter_user_codes_file)

        for domain in DOMAINS_TO_CODE:
            df_in_domain = data[data.domain == domain]
            deduped_on_link = df_in_domain.drop_duplicates(subset='link')
            # filter twitter searches (Can't code those...)
            deduped_on_link = deduped_on_link[~deduped_on_link.link.str.contains('twitter.com/search')]
            #filter twitter hashtags
            deduped_on_link = deduped_on_link[~deduped_on_link.link.str.contains('twitter.com/hashtag')]
            print('There are {} deduped_on_link to code for the domain {}'.format(
                len(deduped_on_link), domain
            ))
            if args.produce_reliability_doc:
                if domain_to_df.get(domain) is None:
                    domain_to_df[domain] = deduped_on_link
                else:
                    domain_to_df[domain] = pd.concat([domain_to_df[domain], deduped_on_link])
                continue
            for row in deduped_on_link:
                link = row.link
                print(link)
                if domain == TWITTER_DOMAIN:
                    screen_name = strip_twitter_screename(link)
                    print(screen_name)
                    cached_val = twitter_user_codes.get(screen_name)
                else:
                    screen_name = None
                    cached_val = link_codes.get(link)
                if cached_val:
                    if args.double_check_cached:
                        print('Cached val is {}, want to keep it? (y/n)'.format(cached_val))
                        choice = input()
                        while choice not in ['y', 'n']:
                            print('Cached val is {}, want to keep it? (y/n)'.format(cached_val))
                            choice = input()
                        if choice == 'y':
                            continue
                    else:
                        continue
                snippet = row.snippet
                code_str = code_item(
                    domain, link=link, screen_name=screen_name,
                    snippet=snippet, api=api)
                data.loc[data.link == link, 'code'] = code_str
                if domain == TWITTER_DOMAIN:
                    twitter_user_codes[screen_name] = code_str
                else:
                    link_codes[link] = code_str
        for d, filename in [
                (link_codes, link_codes_file),
                (twitter_user_codes, twitter_user_codes_file)
        ]:
            if not d:
                continue
            keys, values = zip(*d.items())
            df = pd.DataFrame.from_dict({'key': keys, 'code_str': values})
            df.to_csv(filename)

        if args.db:
            csv_name = 'csvs/' + source.replace('.db', '_coded.csv').replace('dbs\\', '').replace('dbs/', '')
        else:
            csv_name = args.csv
        data.to_csv(csv_name)

    if args.produce_reliability_doc:
        for domain, deduped_on_link in domain_to_df.items():
            n = 10 if len(deduped_on_link) > 10 else len(deduped_on_link)            
            # ret.append(domain.upper() + '\n' + '{} sampled out of {}'.format(n, len(deduped_on_link)) + '\n**********\n')
            if not n:
                continue
            sample = deduped_on_link.sample(n=n)
            for _, row in sample.iterrows():
                link = row.link
                snippet = row.snippet
                entry = {
                    'link': quote(link)
                }
                if snippet:
                    entry['snippet'] = quote(snippet)
                if domain == TWITTER_DOMAIN:
                    screen_name = strip_twitter_screename(link)
                    user_obj = api.get_user(screen_name=screen_name)
                    entry['bio'] = quote(user_obj.description)
                    entry['location'] = quote(user_obj.location)
                    entry['verified'] = quote(str(user_obj.verified))
                    entry['follower_count'] = quote(str(user_obj.followers_count))
                ret.append(entry)
    return ret


def parse():
    """parse args"""
    parser = argparse.ArgumentParser(description='Perform anlysis.')

    parser.add_argument(
        '--db', help='Name of the database', nargs='+', required=True)
    parser.add_argument(
        '--csv', help='Name of the CSV')
    parser.add_argument(
        '--twitter', action='store_true', help='include Twitter links or not'
    )
    parser.add_argument(
        '--double_check_cached', action='store_true',
        help='Want to check the cached values or just accept them?'
    )
    parser.add_argument(
        '--produce_reliability_doc', action='store_true',
        help='write out a txt file for another coder to fill in'
    )


    args = parser.parse_args()
    samples = main(args)
    with open('coding_samples.csv','wb') as outfile:
        top_lines = '\n'.join([
            'Sample produced on {}'.format(datetime.now()),
            'Samples come from the following dbs: {}'.format(str(args.db)),
        ])
        outfile.write((top_lines +'\n\n').encode('utf-8'))
        col_order = ['link', 'snippet', 'bio', 'location', 'verified', 'follower_count']
        class_cols = [
            'Shows Creative Effort (t)', 'No Creative Effort (f)', 'Created outside professional practice (t)', 'Not created outside professional practice (f)',
            '',
            'individual (i)', 'organization (o)' , 'bot (b)',
            '',
            'nonprofit (n)', 'celebrity (c)', 'corporate ($)', 'journalistic (j)', 'political (p)', 'other (z)',
        ]
        headers = col_order + class_cols
        line = ','.join(headers) + '\n'
        outfile.write(line.encode('utf-8'))

        for sample in samples:
            row = [sample.get(col, '') for col in col_order] + [
                '' for x in class_cols
            ]
            line = ','.join(row) + '\n'
            outfile.write(line.encode('utf-8'))


if __name__ == '__main__':
    parse()
