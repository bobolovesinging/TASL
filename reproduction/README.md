# Reproduction guide

| Folder | Paper material | Instructions |
| --- | --- | --- |
| `learning` | Original-table source checks; repeated-run learning and TAS audit results; matched baseline and trust-parameter comparisons | [Experiment instructions](learning/README.md) |
| `sampling` | Audit sampling analysis, source summaries and recorded learning statistics | [Experiment instructions](sampling/README.md) |
| `hbs` | Supplementary Section F: persisted checkpoints, active/archive accounting and recovery verification | [Experiment instructions](hbs/README.md) |

Each folder retains its recorded inputs, configurations, source files and verification scripts. Internal file names and saved table excerpts reflect the experiment version at capture time; the table titles and experiment definitions identify their correspondence with the revised paper.

Run the three commands in the main README to check the available numerical evidence. For training or storage reruns, follow the corresponding folder's commands and environment notes and use a new output directory. Recorded-result verification does not retrain a model or establish identical timing on another machine.
