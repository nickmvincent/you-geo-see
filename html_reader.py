import argparse
import webbrowser
import os
from data_helpers import get_dataframes, load_coded_as_dicts, prep_data


def main(args):
    """
    Open the html files in sequence
    """
    # chrome_path = 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'

    file_paths = []
    db = ""
    for filename in os.listdir(args.dir):
        full = os.path.join(args.dir, filename)        
        if filename.endswith(".html") or filename.endswith(".py"): 
            file_paths.append(full)
        else:
            db = full
    print(db)
    data, serp_df = get_dataframes(db)
    for file_path in file_paths:
        print(file_path)
        _, after_searched = file_path.split('_searched')
        after_searched_before_from, after_from = after_searched.split(' from ')
        query = after_searched_before_from.strip()
        reported_location = after_from.replace('.html', '').strip()
        print('*'+query+'*')
        print('*'+reported_location+'*')
        rows = serp_df[
            (serp_df.reported_location==reported_location) 
            & (serp_df['query']==query)
        ]
        print(int(rows.head()['id']))
        webbrowser.open(file_path)
        input()
        
    


def parse():
    """parse args"""
    parser = argparse.ArgumentParser(description='Open web pages to do testing.')

    parser.add_argument(
        '--dir', help='Path to the directory w/ HTML files',
        #default='/'
        default='dbs/urban_rural_extra_tests'
        )

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    parse()
