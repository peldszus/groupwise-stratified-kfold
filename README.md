# Groupwise, stratified k-fold splits of a dataset for validation


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

- Every class has to appear in at least k groups as you want to k-fold.
  Less frequent than k classes or even singletons are not covered and
  will lead to errors.
- The algorithm prefers similar distributions over similar size, i.e.
  that folds might vary in size depending on how much the groups vary
  in size.


## How to develop

Run `make` to build, `make test` to test and `make lint` to lint.


## About

This piece of code was taken from https://github.com/peldszus/evidencegraph/blob/fba4bc10dada1dc1574b5110be77263b5a8a1b6e/src/evidencegraph/folding.py and turned into a small library.
