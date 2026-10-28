# TASL

Research code for **TASL: Byzantine-Resilient Swarm Learning with Tripartite Auditing and Hierarchical Blockchain Storage**.

The final second-revision implementation and reproducible results are in [`reproduction/final`](reproduction/final). Start with its README and run `python reproduction/final/reproduce.py verify` to validate the downloaded numerical records without a GPU. The `paper-v2` tag identifies this version; the commit object fixes its contents.

The final source implements direction-screened trust weighting with post-smoothing norm exclusion, bound two-role proposal checking, complete-state recovery, and differentiated local retention. Its measured scope is a single-host prototype. Threshold key services, independent networked audits, IPFS transfer and ledger confirmation are proposed deployment interfaces.

## Earlier records

The other training entry points and reproduction directories contain the preceding configuration. They are retained for historical traceability in `paper-v1` and are not final-method validation. For current results use the final directory above.
