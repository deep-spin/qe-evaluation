import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr

"""
Script to evaluate outputs of machine translation quality estimation 
systems for the sentence level, in the WMT 2019 format.

The system output and gold files should have one HTER value per line.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', help='System file')
    parser.add_argument('gold', help='Gold output file')
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help='Show all metrics (Pearson r, Spearman r, MAE, '
                             'RMSE). By default, it only computes Pearson r.')
    args = parser.parse_args()

    system = np.loadtxt(args.system)
    gold = np.loadtxt(args.gold)

    assert len(system) == len(gold), 'Number of gold and system values differ'

    # pearsonr and spearmanr return (correlation, p_value)
    pearson = pearsonr(gold, system)[0]
    print('Pearson correlation: %.4f' % pearson)

    if args.verbose:
        spearman = spearmanr(gold, system)[0]

        diff = gold - system
        mae = np.abs(diff).mean()
        rmse = (diff ** 2).mean() ** 0.5

        print('Spearman correlation: %.4f' % spearman)
        print('MAE: %.4f' % mae)
        print('RMSE: %.4f' % rmse)
