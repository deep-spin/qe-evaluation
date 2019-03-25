import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr

"""
Script to evaluate document-level MQM as in the WMT19 shared task.
"""


def read_scores(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            doc_id, score = line.strip().split()
            data[doc_id] = score

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', help='System file')
    parser.add_argument('gold', help='Gold output file')
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help='Show all metrics (Pearson r, Spearman r, MAE, '
                             'RMSE). By default, it only computes Pearson r.')
    args = parser.parse_args()

    system_dict = read_scores(args.system)
    gold_dict = read_scores(args.gold)
    assert len(system_dict) == len(gold_dict), \
        'Number of gold and system values differ'

    # get the scores in the same order
    doc_ids = list(gold_dict.keys())
    gold_scores = np.array([float(gold_dict[doc_id]) for doc_id in doc_ids])
    sys_scores = np.array([float(system_dict[doc_id]) for doc_id in doc_ids])

    # pearsonr and spearmanr return (correlation, p_value)
    pearson = pearsonr(gold_scores, sys_scores)[0]
    print('Pearson correlation: %.4f' % pearson)

    if args.verbose:
        spearman = spearmanr(gold_scores, sys_scores)[0]

        diff = gold_scores - sys_scores
        mae = np.abs(diff).mean()
        rmse = (diff ** 2).mean() ** 0.5

        print('Spearman correlation: %.4f' % spearman)
        print('MAE: %.4f' % mae)
        print('RMSE: %.4f' % rmse)
