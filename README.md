# Groupwise, stratified k-fold splits of a dataset for validation

Current limitation:
- Every class has to appear in at least k groups as you want to k-fold.
  Less frequent than k classes or even singletons are not covered and
  will lead to errors.
- The algorithm prefers similar distributions over similar size, i.e.
  that folds might vary in size depending on how much the groups vary
  in size.
