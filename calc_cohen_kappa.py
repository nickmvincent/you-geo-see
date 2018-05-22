"""
Calculates cohen kappa for the coding in this study
"""
from sklearn.metrics import cohen_kappa_score
import pandas as pd


def labels_to_numbers(labels1, labels2):
    """
    Turn labels, eg. [tfoj, tfoz]
    into numbers, e.g [0, 1]
    """
    uniques = list(set(labels1 + labels2))
    mapping = {}
    for i, label in enumerate(uniques):
        mapping[label] = i
    return [mapping[label] for label in labels1], [mapping[label] for label in labels2], list(mapping.values())


def main():
    """do calc"""
    print('main')

    labels = pd.read_csv('coding_work\\different_codes.csv', header=None, names = ['first', 'second', 'domain'])
    print(labels)

    for domain in list(labels.domain.drop_duplicates()):
        subdf = labels[labels.domain == domain]
    
        y_1, y_2, all_labels = labels_to_numbers(list(subdf['first']), list(subdf['second']))
        print(all_labels)
        print(y_1)
        print(y_2)
        score = cohen_kappa_score(y_1, y_2)
        print(domain, score)

print('called')
main()
