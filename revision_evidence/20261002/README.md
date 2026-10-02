# TASL selected revision evidence 20261002

This snapshot preserves selected recorded results and source/configuration files.
It is not a complete reproduction of every historical table, a retraining
guarantee, or a deployed threshold-CKKS/IPFS/blockchain system.

## Verify recorded results without GPU or network

From this directory, run:

    python docx_qa/verify_local_reproduction_20261002.py

Python 3.9+ standard library is sufficient. Generated statistics are written to
`generated/`. The verifier checks file SHA-256 values; all 120 legacy Table 4/5
values at saved precision; 130 indexed CIFAR workbook values; five-seed Tables
A.4/A.5/C.1 and Fig. 2 coordinates; the six recent-baseline trajectories; and ten
fixed-constant sensitivity trajectories. The workbook itself is represented by a
cell-addressed extract and whole-file hash. `manuscript/main.tex` contains only
table excerpts for the validator, not the complete manuscript.

`artifact_source_map.json` indexes all 24 current tables and two figures and
states their evidence scope. Indexing a table does not imply that its original
training history has been recovered. Some mapped assets are outside this selected
snapshot; original host paths in saved manifests are historical provenance only.

## Recent LASA comparison

The LASA authors' source is not vendored. In
`r4_tasl_experiments_20260921/recent_baseline_20261002`, obtain it with:

    git clone https://github.com/JiiahaoXU/LASA.git LASA_upstream
    git -C LASA_upstream checkout 8477367a4e8708cde264f7572805040c650af59f

Use the captured Python/Torch environment and a CUDA GPU. The unchanged author
function needs CUDA. `test_lasa_adapter.py` checks exact reference equivalence.
For a new training run, select a NEW output directory, for example:

    python run_matched.py --attack none --output rerun_clean
    python run_matched.py --attack label_flip --output rerun_lf

The public Fashion-MNIST dataset downloads into `pipeline/data/temp`. Source hash,
seed, initial model, partitions, configuration and package-version snapshots are
saved with the records. LASA uses the author CLI defaults sparsity=0.3,
lambda_n=lambda_s=1.0; TASL defaults are unchanged. This is aggregation adaptation
to matched training, not reproduction of the author's original full setup.
The clean condition configures a zero fault budget and therefore no permanent
TASL exclusion; LF configures a four-client cap. The earlier five-seed clean
comparison uses a four-client cap. These groups must not be pooled, and the new
clean result measures screening/weighting cost without permanent exclusion.

## Boundaries

The baseline and sensitivity additions are single-seed Fashion-MNIST diagnostics,
not universal or statistically significant superiority. All ten parameter results
are included without selecting new defaults. Zero honest-client weights differ
from rejection of honest Executor proposals. Five-seed audit and learning groups
retain their separate experimental settings. Historical executable/environment
certification and complete original main-table trajectories are still incomplete.
Package-version snapshots are not a cross-platform cryptographic dependency lock.

Original LASA reference: Jiahao Xu, Zikai Zhang, Rui Hu, WACV 2025, pp. 1508-1517,
https://arxiv.org/abs/2409.01435 and the author repository above.
