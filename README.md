# Groupwise, stratified k-fold splits of a dataset for validation

[![Build Status](https://github.com/peldszus/groupwise-stratified-kfold/actions/workflows/main.yaml/badge.svg?branch=main)](https://github.com/peldszus/groupwise-stratified-kfold/actions)
[![codecov](https://codecov.io/gh/peldszus/groupwise-stratified-kfold/branch/main/graph/badge.svg)](https://codecov.io/gh/peldszus/groupwise-stratified-kfold)
[![GitHub](https://img.shields.io/github/license/peldszus/groupwise-stratified-kfold)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

## Requirements

- Python >= 3.8
- Make


## Installation

```
$ pip install git+https://github.com/peldszus/groupwise-stratified-kfold
```


## Usage

```python
from groupwise_stratified_kfold import GroupwiseStratifiedKFold
from groupwise_stratified_kfold import RepeatedGroupwiseStratifiedKFold

grouped_labels = {
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
k = 4

for train_group_ids, test_group_ids in GroupwiseStratifiedKFold(
    k, grouped_labels, shuffle=False, seed=0
):
    print(test_group_ids)


for train_group_ids, test_group_ids, iteration_id in RepeatedGroupwiseStratifiedKFold(
    k, grouped_labels, shuffle=True, seed=0, repeats=2
):
    print(iteration_id, test_group_ids)
```


## Current limitation

- The algorithm prefers similar class distributions over similar sized folds
  i.e. that folds might vary in size depending on how much the groups vary
  in size.


## How to develop

Run `make` to build, `make test` to test and `make lint` to lint.


## About

This piece of code was taken from https://github.com/peldszus/evidencegraph/blob/fba4bc10dada1dc1574b5110be77263b5a8a1b6e/src/evidencegraph/folding.py and turned into a small library.
