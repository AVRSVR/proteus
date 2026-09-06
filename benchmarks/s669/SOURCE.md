# S669 — third-party benchmark data

The CSVs in this directory are **not** part of Proteus and are not covered by
its MIT license. They are redistributed here so the benchmark scripts are
reproducible, and belong to their original authors.

`ddg_experimental.csv` is the S669 dataset: 669 single-point mutations across
94 proteins, curated to share no more than 30% sequence identity with the
training sets of the predictors it was built to evaluate. That last property is
the reason it is used here rather than a larger set — it is one of the few
benchmarks where a published number is not contaminated by training data.

The file also carries the per-mutation predictions of every method compared in
the original paper (FoldX, ACDC-NN, DDGun3D, INPS3D, RaSP and others), which is
where the reference correlations quoted in the README come from. They were read
off this table, not re-run here.

> Pancotti, C., Benevenuta, S., Birolo, G., Alberini, V., Repetto, V.,
> Sanavia, T., Capriotti, E., Fariselli, P. (2022). Predicting protein
> stability changes upon single-point mutation: a thorough comparison of the
> available tools on a new dataset. *Briefings in Bioinformatics*, 23(2).
> https://doi.org/10.1093/bib/bbab555

Structures (`pdb/`) are not redistributed; they are downloaded from the RCSB
PDB, and are gitignored.
