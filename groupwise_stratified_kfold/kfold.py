"""
Groupwise, stratified k-fold splits of a dataset for validation.

Current limitation:
- The algorithm prefers similar class distributions over similar sized folds
  i.e. that folds might vary in size depending on how much the groups vary
  in size.
"""

import random
from collections import defaultdict, deque
from itertools import chain


def absolute_class_counts(data, expected_classes=None):
    """input: an iterable of occurrences of classes
           [0,2,2,1,0,1,2,2,0]
    output: a dict mapping class keys to their absolute counts
           {0:3, 1:2, 2:4}"""
    counts_class = defaultdict(int)
    if expected_classes is not None:
        for c in expected_classes:
            counts_class[c]
    for e in data:
        counts_class[e] += 1
    return counts_class


def relative_class_counts(data):
    """input: a dict mapping class keys to their absolute counts
    output: a dict mapping class keys to their relative counts"""
    counts_items = sum(data.values())
    return {k: 1.0 * v / counts_items for k, v in data.items()}


def diff_distribution(a, b, weights=None):
    """compares two distributions and returns a sum of all (weighted)
    diffs"""
    assert a.keys() == b.keys()
    if weights is not None:
        assert a.keys() == weights.keys()
        diff = {k: weights[k] * abs(a[k] - b[k]) for k in a}
    else:
        diff = {k: abs(a[k] - b[k]) for k in a}
    return sum(diff.values())


def join_distributions(a, b):
    """joins two distributions of absolute class counts by adding the values
    of each key"""
    assert a.keys() == b.keys()
    return {k: a[k] + b[k] for k in a}


class GroupwiseStratifiedKFold:
    def __init__(self, number_of_folds, data, shuffle=False, seed=0):
        """
        Groupwise, stratified k-fold splits of a dataset for validation.
        It is stratified, because the label distributions aim to be similar
        across the splits. It is groupwise, because classification items are
        grouped, i.e. considered belonging together, so that not items but
        groups of items are sampled.

        In our case we sample classification items grouped together because they
        belong to one input group, so that the kfold does not contain fragments
        of groups.

        The input `data` is considered to be a dict mapping from group ids to
        lists of labels.
        """
        self.fold_register = {}
        ungrouped_data = list(chain(*list(data.values())))
        counts_class_absolute = absolute_class_counts(ungrouped_data)
        counts_class_relative = relative_class_counts(counts_class_absolute)
        classes = list(counts_class_absolute)
        class_weights = {k: 1 - v for k, v in counts_class_relative.items()}
        group_distribution = {
            k: absolute_class_counts(list(v), expected_classes=classes)
            for k, v in data.items()
        }
        folds = {
            n: {k: 0 for k in counts_class_relative}
            for n in range(1, number_of_folds + 1)
        }
        fold_register = {n: [] for n in folds}
        pool = set(group_distribution)

        cnt_pass = 0
        while len(pool) > 0:
            # either shuffle the order of filling folds in this pass randomly
            # or rotate it, in order to prevent that the first folds or a pass
            # always get the best possible draw from the pool
            if shuffle:
                random.seed(seed + cnt_pass)
                fold_order_in_this_pass = list(folds)
                random.shuffle(fold_order_in_this_pass)
            else:
                fold_order_in_this_pass = deque(folds)
                fold_order_in_this_pass.rotate(-cnt_pass)

            # in a pass, fill each fold with the best group
            for this_fold in fold_order_in_this_pass:
                if len(pool) == 0:
                    # if the number of groups is not a multiple of
                    # the desired number of folds
                    break

                # find the group in the pool, that minimizes the difference of
                # this fold to the base distribution
                min_diff = float("+inf")
                min_group = None
                min_joint_dist = None
                for group in pool:
                    joint_dist = join_distributions(
                        folds[this_fold], group_distribution[group]
                    )
                    diff = diff_distribution(
                        counts_class_relative,
                        relative_class_counts(joint_dist),
                        weights=class_weights,
                    )
                    if diff < min_diff:
                        min_diff = diff
                        min_group = group
                        min_joint_dist = joint_dist

                # remove group from pool, register group in fold and add group
                # absolutes to fold
                pool.remove(min_group)
                fold_register[this_fold].append(min_group)
                folds[this_fold] = min_joint_dist

            cnt_pass += 1

        self.fold_register = fold_register

    def __iter__(self):
        """Yields group ids of training and testing items."""
        for test_fold in self.fold_register:
            train_foldes = list(self.fold_register)
            train_foldes.remove(test_fold)
            train_ids_per_fold = [self.fold_register[f] for f in train_foldes]
            train_ids = list(chain(*train_ids_per_fold))
            test_ids = self.fold_register[test_fold]
            yield train_ids, test_ids


class RepeatedGroupwiseStratifiedKFold:
    def __init__(
        self, number_of_folds, data, shuffle=False, seed=0, repeats=10
    ):
        """
        Repeated, groupwise, stratified k-fold splits of a dataset for validation.
        The GroupwiseStratifiedKFold is repeated with different random seeds in
        order to yield different kfold splits of the same dataset.
        """
        self.iterations = []
        for repeat_nr in range(repeats):
            foldes = GroupwiseStratifiedKFold(
                number_of_folds, data, shuffle=shuffle, seed=seed + repeat_nr
            )
            for fold_nr, (train, test) in enumerate(foldes):
                self.iterations.append(
                    (train, test, "%d-%d" % (repeat_nr, fold_nr))
                )

    def __iter__(self):
        """Yields group ids of training items, testing items, and the iteration id."""
        yield from self.iterations
