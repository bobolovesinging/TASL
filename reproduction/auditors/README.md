# Final audit and recovery records

This directory contains the audit implementation used for the updated governance rows, the complete-verifier comparison, and sampled-weight sensitivity in the second revision (`paper-v2.3`). The weighting rule and training partitions match the preceding governance study. Numerical results for the preceding implementation remain in `../governance`; the updated auditor rows are here.

`V_S` independently recomputes all weights and checks one post-commit aggregate projection. `V_P` independently derives necessary admissibility intervals for sampled weights and checks the exact aggregate. An interval is a necessary condition, not an exact coefficient recomputation. Recovery reuses `V_S`'s independently computed weights for the same bound inputs and requires fresh endorsements. A `V_P`-only recovery computes the replacement weights once.

## Verify the records

From the repository root:

```text
python reproduction/final/reproduce.py verify
python reproduction/governance/verify.py
python reproduction/auditors/verify.py
```

The third command checks file digests, the retained executed scientific sources against run manifests, all 12 new 50-round trajectories, reference-state matches, coverage, timing medians, and conditional sampling probabilities. Verification requires only the Python standard library. Tests in `executed_sources` additionally exercise the bound checks, input binding, reference reuse, and replacement endorsement in a PyTorch/NumPy environment.

## Repeat training and measurements

Use Python 3.10, PyTorch 2.5.1, torchvision, NumPy 1.26.4, and a compatible CUDA environment. The MNIST and Fashion-MNIST datasets are downloaded through torchvision. Each command requires a fresh workspace outside this repository. The adapter changes file locations only; unmodified executed sources are preserved separately.

```text
python reproduction/auditors/reproduce.py train --dataset mnist --workspace /path/to/mnist-run
python reproduction/auditors/reproduce.py train --dataset fashionmnist --workspace /path/to/fashion-run
```

Each dataset command repeats six conditions: the `V_P`-only rows under both exclusion budgets, and Full TAS compatibility checks for the main-table budget, under LF and scaling. Manifests preserve their distinct configurations; state comparisons use the released paired references.

The original captured updates can be regenerated using `python reproduction/final/reproduce.py c2 --workspace /path/to/capture-run`. Then run:

```text
python reproduction/auditors/reproduce.py benchmark --workspace /path/to/benchmark-run --capture-workspace /path/to/capture-run
python reproduction/auditors/reproduce.py sampling --workspace /path/to/sampling-run --capture-workspace /path/to/capture-run
```

CPU timings use two PyTorch intra-op threads and one OpenBLAS thread. Coverage uses all 50 captured rounds; timings use rounds 1, 3, 10, 25, and 50 with eight rotated repetitions. Timings are serial computation measurements, and will vary by host. Projection challenge seeds and input hashes are recorded. No network latency is inferred from these measurements.

The manifests retain original execution paths and digests for provenance. Operational launch and temperature-monitor scripts are not needed for reproduction and are omitted; all executed scientific sources are included. Large model/update tensors are regenerated rather than distributed.
