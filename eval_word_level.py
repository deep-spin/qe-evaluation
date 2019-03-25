import argparse

"""
Script to evaluate outputs of machine translation quality estimation 
systems for the word level, in the WMT 2019 format.
"""


def read_tags(filename, only_gaps=False, only_words=False):
    all_tags = []

    with open(filename, 'r') as f:
        for line in f:
            tags = line.split()
            if only_gaps:
                tags = tags[::2]
            elif only_words:
                tags = tags[1::2]
            all_tags.append(tags)

    return all_tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', help='System file')
    parser.add_argument('gold', help='Gold output file')
    tag_type_group = parser.add_mutually_exclusive_group()
    tag_type_group.add_argument('-w', help='Only evaluate word tags',
                                action='store_true', dest='only_words')
    tag_type_group.add_argument('-g', help='Only evaluate gap tags',
                                action='store_true', dest='only_gaps')
    args = parser.parse_args()

    system_tags = read_tags(args.system, args.only_gaps, args.only_words)
    gold_tags = read_tags(args.gold, args.only_gaps, args.only_words)

    assert len(system_tags) == len(gold_tags), \
        'Number of lines in system and gold file differ'

    # true/false positives/negatives
    tp = tn = fp = fn = 0

    for i, (sys_sentence, gold_sentence) in enumerate(zip(system_tags,
                                                          gold_tags), 1):
        assert len(sys_sentence) == len(gold_sentence), \
            'Number of tags in system and gold file differ in line %d' % i

        for sys_tag, gold_tag in zip(sys_sentence, gold_sentence):
            if sys_tag == 'OK':
                if sys_tag == gold_tag:
                    tp += 1
                else:
                    fp += 1
            else:
                if sys_tag == gold_tag:
                    tn += 1
                else:
                    fn += 1

    total_tags = tp + tn + fp + fn
    num_sys_ok = tp + fp
    num_gold_ok = tp + fn
    num_sys_bad = tn + fn
    num_gold_bad = tn + fp

    precision_ok = tp / num_sys_ok if num_sys_ok else 1.
    recall_ok = tp / num_gold_ok if num_gold_ok else 0.
    precision_bad = tn / num_sys_bad if num_sys_bad else 1.
    recall_bad = tn / num_gold_bad if num_gold_bad else 0.

    if precision_ok + recall_ok:
        f1_ok = 2 * precision_ok * recall_ok / (precision_ok + recall_ok)
    else:
        f1_ok = 0.

    if precision_bad + recall_bad:
        f1_bad = 2 * precision_bad * recall_bad / (precision_bad + recall_bad)
    else:
        f1_bad = 0.

    f1_mult = f1_ok * f1_bad

    print('F1 OK: %.4f' % f1_ok)
    print('F1 BAD: %.4f' % f1_bad)
    print('F1 Mult: %.4f' % f1_mult)
