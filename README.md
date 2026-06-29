# TASL

Code for the TASL experiments, including robust federated learning, tripartite auditor governance, and storage/economic scalability studies.

## Environment

```bash
pip install -r requirements.txt
```

The code was developed with PyTorch and torchvision. Experiments can run on CPU for smoke tests, but the full CIFAR-10 and multi-seed experiments are intended for GPU execution.

## Main Experiment Entry Points

| Purpose | Script |
|---|---|
| MNIST robustness experiments | `scripts/exp2/exp2_mnist_v3.py` |
| Fashion-MNIST robustness experiments | `scripts/exp2/exp2_fashionmnist.py` |
| CIFAR-10 robustness experiments | `scripts/exp2/exp2_cifar10_v3.py` |
| Unified configuration runner | `scripts/exp2/exp2_run.py` |
| BSP / trust-layer ablation | `scripts/exp2/exp2_fashionmnist_bsp_ablation.py` |
| TAS governance audit | `scripts/exp3/exp3_governance_tasv2.py` |
| Legacy governance ablations | `scripts/exp3/exp_governance_ablation.py`, `scripts/exp3/exp_governance_v8.py` |
| Storage scalability | `scripts/exp4/exp4_storage_scalability.py` |
| Economic scalability | `scripts/exp5/exp5_economic_scalability.py` |

## Quick Smoke Tests

```bash
python scripts/exp3/exp3_governance_tasv2.py \
  --rounds 1 --clients 2 --byzantine-ratio 0.5 \
  --batch-size 8 --local-epochs 0 --dataset mnist \
  --executor-attack mixed12 --attacks-per-block 1
```

```bash
python scripts/exp2/exp2_mnist_v3.py \
  --quick --attack label_flip --skip-clean \
  --output results/smoke_mnist
```

## Reproduction Notes

- Full experiment outputs are written under `results/` by default; this directory is ignored by Git.
- Datasets are downloaded or prepared locally under `data/`, which is also ignored by Git.
- `--skip-clean` skips the clean baseline run. When clean baseline is skipped, ASR values that require a clean baseline are reported as `nan` instead of using placeholder values.
- YAML files under `configs/` capture the main robustness, ablation, and governance settings used by the manuscript experiments.

