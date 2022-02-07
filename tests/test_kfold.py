"""
Tests for the implementation of the GroupwiseStratifiedKFold
"""

import math

from groupwise_stratified_kfold import (
    GroupwiseStratifiedKFold,
    RepeatedGroupwiseStratifiedKFold,
)
from groupwise_stratified_kfold.kfold import (
    absolute_class_counts,
    diff_distribution,
    join_distributions,
    relative_class_counts,
)

example_data_1 = {
    "g01": "BBDE",
    "g02": "CBEDF",
    "g03": "BBAA",
    "g04": "ABCD",
    "g05": "ABBDF",
    "g06": "ABC",
    "g07": "ABBAA",
    "g08": "ACBD",
    "g09": "DEBBA",
    "g10": "AABC",
    "g11": "AAAAF",
    "g12": "CCDACD",
    "g13": "CCADB",
    "g14": "CBAF",
    "g15": "ABCD",
    "g16": "CDBD",
}

example_data_2 = {
    "g01": "BBDE",
    "g02": "CBEDF",
    "g03": "BBAA",
    "g04": "ABCD",
    "g05": "ABBDF",
    "g06": "ABC",
    "g07": "ABBAA",
    "g08": "ACBD",
    "g09": "DEBBA",
    "g10": "AABC",
    "g11": "AAAAF",
    "g12": "CCDACD",
    "g13": "CCADB",
    "g14": "CBAF",
    "g15": "ABCD",
    "g16": "CDBD",
    "g17": "BBDE",
    "g18": "CBEDF",
    "g19": "BBAA",
    "g20": "ABCD",
    "g21": "ABBDF",
    "g22": "ABC",
    "g23": "ABBAA",
    "g24": "ACBD",
    "g25": "DEBBA",
    "g26": "AABC",
    "g27": "AAAAF",
    "g28": "CCDACD",
    "g29": "CCADB",
    "g30": "CBAF",
    "g31": "ABCD",
    "g32": "CDBD",
}


def test_groupwise_stratified_kfold():
    folds = list(GroupwiseStratifiedKFold(4, example_data_1))

    # Correct length of folds.
    assert len(folds) == 4

    # No overlap between train and tests folds.
    assert all(set(train) & set(test) == set() for train, test in folds)

    # Each train/test split covers the whole data.
    all(set(train) | set(test) == set(example_data_1) for train, test in folds)

    # All test sets of the folding cover the whole data.
    assert {group for tr, test in folds for group in test} == set(
        example_data_1
    )

    # Folding is stratified, i.e. label distributions are similar.
    assert [
        "".join(
            sorted(label for group in train for label in example_data_1[group])
        )
        for train, _ in folds
    ] == [
        "AAAAAAAAAAAAAABBBBBBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEFFF",
        "AAAAAAAAAAAAAABBBBBBBBBBBBBBBCCCCCCCCCCCDDDDDDDDDDEEFFF",
        "AAAAAAAAAAAAAAAAABBBBBBBBBBBBBBCCCCCCCCCCDDDDDDDDEEFFF",
        "AAAAAAAAAAAAAAABBBBBBBBBBBBBBCCCCCCCCDDDDDDDDEEEFFF",
    ]


def test_groupwise_stratified_kfold_groups_not_a_multiple_of_k():
    # Add one extra 17th group to the data.
    data = dict(example_data_1, g17="ABCDEF")
    folds = list(GroupwiseStratifiedKFold(4, data))

    # Correct length of folds.
    assert len(folds) == 4

    # No overlap between train and tests folds.
    assert all(set(train) & set(test) == set() for train, test in folds)

    # Each train/test split covers the whole data.
    all(set(train) | set(test) == set(data) for train, test in folds)

    # All test sets of the folding cover the whole data.
    assert {group for tr, test in folds for group in test} == set(data)


def test_groupwise_stratified_kfold_less_than_k_samples_for_class():
    # Add one extra 17th group to the data with a unique X label.
    data = dict(example_data_1, g17="ABCDEFX")
    folds = list(GroupwiseStratifiedKFold(4, data))

    # Correct length of folds.
    assert len(folds) == 4

    # No overlap between train and tests folds.
    assert all(set(train) & set(test) == set() for train, test in folds)

    # Each train/test split covers the whole data.
    all(set(train) | set(test) == set(data) for train, test in folds)

    # All test sets of the folding cover the whole data.
    assert {group for tr, test in folds for group in test} == set(data)


def test_repeated_groupwise_stratified_kfold():
    folds = list(
        RepeatedGroupwiseStratifiedKFold(
            4, example_data_2, shuffle=True, repeats=2
        )
    )

    # Correct length of folds.
    assert len(folds) == 2 * 4

    # Every fold is different. Note this only works when suffle=True and the
    # dataset is large enough.
    assert len({tuple(sorted(test)) for _, test, _ in folds}) == 2 * 4


def test_absolute_class_counts():
    data = [0, 2, 2, 1, 0, 1, 2, 2, 0]
    expected = {0: 3, 1: 2, 2: 4}

    assert absolute_class_counts(data) == expected

    # Counts sum up to the length of the input data.
    assert sum(expected.values()) == len(data)

    # For every unique item in the data there is one key value.
    assert set(data) == set(expected.keys())


def test_absolute_class_counts_with_expected_classes():
    expected_classes = [0, 1, 2, 9]
    data = [0, 2, 2, 1, 0, 1, 2, 2, 0]
    expected = {0: 3, 1: 2, 2: 4, 9: 0}

    assert (
        absolute_class_counts(data, expected_classes=expected_classes)
        == expected
    )

    # Counts sum up to the length of the input data.
    assert sum(expected.values()) == len(data)

    # For every unique item in the data there is one key value.
    assert set(data).issubset(set(expected.keys()))


def test_relative_class_counts():
    data = {0: 0, 1: 1, 2: 2, 7: 7}
    expected = {0: 0.0, 1: 0.1, 2: 0.2, 7: 0.7}
    assert relative_class_counts(data) == expected

    # All keys appear in the relative counts
    assert data.keys() == expected.keys()

    # Values sum up to 1.
    assert sum(expected.values()) == 1.0


def test_diff_distribution():
    # Equal distributions yield zero diff.
    assert math.isclose(
        diff_distribution({0: 0.1, 9: 0.9}, {0: 0.1, 9: 0.9}), 0.0
    )

    # Similar distributions yield small diff.
    assert math.isclose(
        diff_distribution({0: 0.1, 9: 0.9}, {0: 0.2, 9: 0.8}), 0.2
    )

    # Different distributions yield a larger diff.
    assert math.isclose(
        diff_distribution({0: 0.1, 9: 0.9}, {0: 0.5, 9: 0.5}), 0.8
    )

    # Weights can defined per class how much a diff for that class contributes
    # to the total diff.
    assert math.isclose(
        diff_distribution(
            {0: 0.1, 9: 0.9}, {0: 0.5, 9: 0.5}, weights={0: 0.99, 9: 0.01}
        ),
        0.4,
    )


def test_join_distributions():
    assert join_distributions({0: 0.1, 9: 0.9}, {0: 0.1, 9: 0.9}) == {
        0: 0.2,
        9: 1.8,
    }
