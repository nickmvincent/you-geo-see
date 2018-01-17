"""
Calculates cohen kappa for the coding in this study
"""
from sklearn.metrics import cohen_kappa_score


def labels_to_numbers(labels1, labels2):
    """
    Turn labels, eg. [tfoj, tfoz]
    into numbers, e.g [0, 1]
    """
    uniques = list(set(labels1 + labels2))
    mapping = {}
    for i, label in enumerate(uniques):
        mapping[label] = i
    return [mapping[label] for label in labels1],  [mapping[label] for label in labels2]


def main():
    """do calc"""
    print('main')
    y_1, y_2 = labels_to_numbers([
        'tfoj',
        'tfoz',
        'tfoz',
        'ttiz',
        'tfoj',
        'tfo$',
        'ffo$',
        'ttoz',
        'tfon',
        'tfon',
    ], [
        'tfoj',
        'tfoz',
        'tfoz',
        'ttiz',
        'tfoj',
        'tfo$',
        'ftiz',
        'tfo$',
        'tfon',
        'tfon',
    ])
    print(y_1)
    print(y_2)
    score = cohen_kappa_score(y_1, y_2)
    print(score)

print('called)')
main()
