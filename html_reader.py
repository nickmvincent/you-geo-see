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
            print(full)
            files.append(full)
            input()
        else:
            db = full
    data, serp_df = get_dataframes(args.db)
    for file in files:

        webbrowser.open(full)
    


def parse():
    """parse args"""
    parser = argparse.ArgumentParser(description='Open web pages to do testing.')

    parser.add_argument(
        '--dir', help='Path to the directory w/ HTML files', default='/')

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    parse()
