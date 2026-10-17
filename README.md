# TASL

Research code for **TASL: Byzantine-Resilient Swarm Learning with Tripartite Auditing and Hierarchical Blockchain Storage**.

## Start here

| Material | Location |
| --- | --- |
| Training prototype and original baselines | `main.py`, `scripts/`, `system/`, `configs/` |
| Learning results, TAS audits, configurations and analysis | [`reproduction/learning`](reproduction/learning) |
| Weight-sampling checks and CIFAR-10 result sources | [`reproduction/sampling`](reproduction/sampling) |
| Fifty-round HBS storage, archive and recovery checks | [`reproduction/hbs`](reproduction/hbs) |

The revised experiments use the frozen sources included in their corresponding folders. The root training code also contains the original audit configuration; use the experiment-specific instructions for the revised TAS protocol.

## Check the available results

From the repository root, using Python 3.9 or newer:

```bash
python reproduction/learning/docx_qa/verify_local_reproduction_20261002.py
python reproduction/sampling/verify.py
python reproduction/hbs/verify.py
```

These checks recompute recorded statistics and validate stored results without a GPU. Training commands, seeds, configuration files and captured package versions are documented with each experiment. Training uses the public MNIST, CIFAR-10 and Fashion-MNIST datasets. Start with `requirements.txt` and use the recorded environment for the experiment being rerun.

See [reproduction instructions](reproduction/README.md) for the correspondence with the paper. A fixed [paper evidence version](https://github.com/bobolovesinging/TASL/tree/paper-v1) is retained for reference; this directory layout reorganizes the same available experiment records.

Some original training histories and environment records are incomplete. The experiment folders distinguish recorded-result checks from training reruns. The tested prototype does not provide an end-to-end deployed threshold-decryption, IPFS and blockchain service.
