# Governance comparisons

This directory contains the MNIST/Fashion-MNIST governance comparisons with ten clients, four Byzantine clients, seed 42, Dirichlet alpha 0.3, and 50 rounds. Client attacks are label flipping and scaling by five. Each scenario compares an honest Executor, trust weighting alone, Vs alone, Vp alone, and TAS. Thirty proposals are attacked, ten each of A1/A2/A3; twenty are honest.

`cap0` uses permanent-exclusion budget zero. `cap4` enables permanent exclusion with budget four. Both use the same final norm mask, direction screening, temporal histories, independent role checks, transcript binding, and complete-state recovery. The kernel and audit source files have the same hashes as `../final`.

## Verify recorded results

```sh
python reproduction/governance/verify.py
python reproduction/final/reproduce.py verify
```

These commands verify file digests, recorded statistics, partition coverage, and complete-state pairings without a GPU. They leave the released files unchanged.

## Rerun training

Use the environment specified in `../final/requirements.txt`. Set CUDA_VISIBLE_DEVICES to an available GPU. Each command copies the executed scientific sources into a separate workspace. Training downloads the public dataset and saves regenerated tensor checkpoints there.

```sh
python reproduction/governance/reproduce.py --budget 0 --dataset fashionmnist --workspace /path/to/fresh-cap0
python reproduction/governance/reproduce.py --budget 0 --dataset mnist --workspace /path/to/fresh-cap0
python reproduction/governance/reproduce.py --budget 4 --workspace /path/to/fresh-cap4
```

The cap-four runner evaluates both datasets sequentially. It checks optional source/configuration-matched cached records when present and otherwise trains each condition afresh. Its recorded run used three matched cached Fashion-MNIST LF conditions; their original manifests and digests accompany the records. The cap-zero run trains all twenty conditions afresh. CPU preflight checks run before training. Tensor payloads are regenerated rather than shipped; no secret keys or server-login scripts are included.

`records` contains each condition's configuration, source identities, round-wise numerical outputs, and statistics. Separate learning-entry trajectories verify the same reference states. `code` preserves the executed scientific source files; the explicit driver arguments select the evaluated settings.
