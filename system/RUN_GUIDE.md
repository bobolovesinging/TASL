# Run Guide

Run commands from the repository root unless otherwise noted.

## Main Program

```bash
python main.py
```

To use a custom configuration file:

```bash
python main.py --config_path path/to/config.yaml
```

## Experiment Scripts

The manuscript experiments are under `scripts/`:

- `scripts/exp2/`: robustness and ablation experiments.
- `scripts/exp3/`: governance and auditor experiments.
- `scripts/exp4/`: storage scalability experiments.
- `scripts/exp5/`: economic scalability experiments.

Generated outputs should be written under `results/`, which is ignored by Git.
