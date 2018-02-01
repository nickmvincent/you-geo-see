import pandas as pd
import sqlite3


def load_coded_as_dicts(link_codes_file, twitter_user_codes_file):
    """
    Loads two dictionaries
    link: code_str
    twitter_screen_name: code_str
    """
    try:
        link_codes_df = pd.read_csv(link_codes_file)
        link_codes = pd.Series(link_codes_df.code_str.values, index=link_codes_df.key).to_dict()
    except FileNotFoundError:
        print('Could not load {} creating empty dict'.format(link_codes_file))
        link_codes = {}
    try:
        twitter_user_codes_df = pd.read_csv(twitter_user_codes_file)
        twitter_user_codes = pd.Series(twitter_user_codes_df.code_str.values, index=twitter_user_codes_df.key).to_dict()
    except FileNotFoundError:
        print('Could not load {} creating empty dict'.format(twitter_user_codes_file))
        twitter_user_codes = {}
    return link_codes, twitter_user_codes


def get_dataframes(dbname):
    """
    Get rows from the db and convert to dataframes
    """
    print('dbname,', dbname)
    conn = sqlite3.connect(dbname)
    select_results = (
        """
        SELECT serp.*, link.*, scraper_searches_serps.scraper_search_id from serp INNER JOIN link on serp.id = link.serp_id
        INNER JOIN scraper_searches_serps on serp.id = scraper_searches_serps.serp_id;
        """
    )
    select_serps = (
        """
        SELECT * from serp;
        """
    )
    data = pd.read_sql_query(select_results, conn)
    serp_df = pd.read_sql_query(select_serps, conn)
    conn.close()
    return data, serp_df



def process_domain(x):
    if x.raw_domain == 'TweetCarousel':
        if 'search' in x.link:
            return 'SearchTweetCarousel'
        else:
            return 'UserTweetCarousel'
    try:
        if x.raw_domain.count('.') > 1:
            first_period = x.raw_domain.find('.')
            stripped = x.raw_domain[first_period+1:]
            return stripped
    except TypeError:
        pass
    return x.raw_domain


def prep_data(data):
    """
    Prep operation on the dataframe:
        change nulls to false for Boolean variables
        fill null links w/ empty string
        make domain categorical variable
    args:
        data - dataframe with results
    returns:
        prepped dataframe
    """
    data.fillna({
        'isTweetCarousel': 0,
        'isMapsPlaces': 0,
        'isMapsLocations': 0,
        'isNewsCarousel': 0,
    }, inplace=True)
    data.loc[data.link.isnull(), 'link'] = ''
    tweet_mask = data.isTweetCarousel == True 
    news_mask = data.isNewsCarousel == True
    kp_mask = data.link_type == 'knowledge_panel'
    maps_location_mask = data.isMapsLocations == True
    maps_places_mask = data.isMapsPlaces == True

    people_also_ask_mask = data.misc.astype(str).str.contains(';People also ask')
    data.loc[people_also_ask_mask, 'domain'] = 'people also ask'

    data.loc[tweet_mask, 'domain'] = 'TweetCarousel'
    data.loc[news_mask, 'link'] = 'NewsCarousel'
    data.loc[news_mask, 'domain'] = 'NewsCarousel'
    # data.loc[kp_mask, 'link'] = 'KnowledgePanel' 
    # data.loc[kp_mask, 'domain'] = 'KnowledgePanel' 

    data.loc[maps_location_mask, 'link'] = 'MapsLocations' 
    data.loc[maps_location_mask, 'domain'] = 'MapsLocations'

    data.loc[maps_places_mask, 'link'] = 'MapsPlaces' 
    data.loc[maps_places_mask, 'domain'] = 'MapsPlaces' 



    
    
    def process_each_domain(df):
        return df.apply(process_domain, axis=1)
    
    data = data.rename(index=str, columns={"domain": "raw_domain"})
    data = data.assign(domain = process_each_domain)
    data.raw_domain = data.raw_domain.astype('category')
    data.domain = data.domain.astype('str')
    data.domain = data.domain.astype('category')

    return data


def set_or_concat(df, newdf):
    if df is None:
        ret = newdf
    else:
        ret = pd.concat([df, newdf])
    return ret


def strip_domain_strings_wrapper(subset):
    """wraps a function to strip results_type, subset, and domain_frac text"""

    def strip_domain_strings(text):
        """Strip the string"""
        text = text.strip('*')
        text = text.replace('results_', '')
        text = text.replace(subset, '')
        text = text.replace('domain_frac_', '')
        text = text.replace('domain_appears_', '')
        text = text.replace('code_appears', '')
        text = text.strip('_')
        return text
    return strip_domain_strings