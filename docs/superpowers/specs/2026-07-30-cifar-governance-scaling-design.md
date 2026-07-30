# CIFAR-10 Scaling Governance Unified Runner Design

## Goal

Fix the CIFAR-10 governance Scaling experiment so that the Sweep TASL
baseline and the governance `No Executor` group execute the same training
path under one explicit seed. Executor attacks and defenses must be layered
only on the Executor proposal step.

The historical `82.32` result is evidence for the intended Sweep
configuration, but it is not a hard output target because that run used
Python's process-randomized `hash()` to derive an unrecorded effective seed.
The new acceptance criterion is exact round-by-round equivalence under a
stable explicit seed.

## Existing Problem

The historical Sweep result came from
`scripts/exp2/exp2_cifar10_v3.py`, while the CIFAR governance experiment was
implemented independently in `scripts/exp3/exp_governance_v9_cifar10.py`.
The two paths differ in model initialization, CIFAR data handling, learning
rate scheduling, random-state reset, trust computation, and temporal trust
state. Consequently, the governance `No Executor` group is not the Sweep
TASL baseline.

The checked-in Sweep file also exposes `scaling` as a CLI choice without
registering a Scaling training function. The server copy that produced the
historical result trained a client normally and amplified its model update
by a factor of five.

## Architecture

`scripts/exp2/exp2_cifar10_v3.py::run_single()` remains the single CIFAR-10
training engine. It will gain one optional Executor callback at the point
where TASL has already computed the correct trust weights and baseline
aggregate.

```text
client training
  -> Scaling x5 for Byzantine clients
  -> TASL trust computation
  -> same-round No Executor aggregate
  -> optional Executor callback
  -> model update
  -> evaluation
```

Normal Sweep execution supplies no callback. Governance `No Executor` also
supplies no callback. They are therefore the same function call with the
same seed and inputs.

The new
`scripts/exp3/exp3_governance_cifar10_scaling.py` owns only:

- the deterministic Executor attack schedule;
- A1 and A2 proposal tampering;
- V1 and V2 verification;
- same-round recovery;
- governance group orchestration and result serialization.

It must not reimplement CIFAR loading, the model, client training, Scaling,
TASL trust scoring, the cosine learning-rate schedule, or evaluation.

## Shared Runner Interface

The Sweep module will define:

```python
@dataclass(frozen=True)
class ExecutorRoundContext:
    round_number: int
    seed: int
    local_updates: Dict[int, Dict[str, torch.Tensor]]
    trust_weights: Dict[int, float]
    similarity_scores: Dict[int, float]
    baseline_aggregate: Dict[str, torch.Tensor]
    byzantine_ids: Tuple[int, ...]


@dataclass
class ExecutorRoundDecision:
    aggregate: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]
```

`run_single()` accepts:

```python
executor_hook: Optional[
    Callable[[ExecutorRoundContext], ExecutorRoundDecision]
] = None
```

The hook is valid only for `algo="tasl"`. With no hook,
`baseline_aggregate` is committed directly. When a hook is present, its
returned aggregate is committed and its metadata is appended to
`executor_history`.

The existing Sweep return dictionary remains compatible and gains
`executor_history`.

## Scaling Attack

Scaling uses the exact semantic rule from the server Sweep:

```text
scaled_local = global + 5 * (trained_local - global)
```

Only floating tensors are amplified. Non-floating state entries retain the
trained value. The helper is pure and separately testable; client training
then delegates to the existing honest CIFAR training routine before applying
the helper.

## Executor Attacks

Executor attacks are active only on scheduled attack rounds.

### A1: Forged Trust

The Executor changes published weights for Byzantine clients to `0.75`,
normalizes those forged weights for aggregation, and submits the resulting
aggregate.

### A2: Tampered Aggregate

The Executor publishes the correct TASL trust weights but increases each
Byzantine aggregation weight by `0.25`, capped at `0.50`, normalizes the
tampered weights, and submits that aggregate.

The attack schedule supports `A1`, `A2`, and `mixed12`. `mixed12` selects
one of A1/A2 deterministically for each scheduled attack round.

## Governance Groups

- `no_executor`: no Executor callback; identical to Sweep TASL.
- `trust_only`: applies A1/A2 and accepts the malicious aggregate.
- `v1`: performs only the V1 defense.
- `v2`: performs only the V2 defense.
- `tas`: performs V1 and V2 and blocks when either detects manipulation.

Compatibility aliases may be accepted at the CLI, but serialized group names
use the five canonical names above.

## Defenses

### V1

V1 fully compares the published trust weights with the independently known
correct TASL weights using a fixed absolute tolerance. It is responsible for
A1 and does not inspect aggregate consistency, so it does not defend A2.

### V2

V2 uses a deterministic random projection to compare the submitted aggregate
against the aggregate implied by the published weights. It is responsible
for A2. Because A1's submitted aggregate is consistent with its published
weights, V2 does not defend A1.

### TAS and Recovery

TAS runs both checks. If the enabled defense detects manipulation, the
Executor proposal is discarded and the already computed
`baseline_aggregate` from the same round is committed. Recovery does not
exclude a client, alter TASL trust state, or roll back to another round.

## Determinism

The matrix loop will stop deriving seeds through `hash(algo + attack)`.
Every group receives one explicit run seed. `run_single()` resets Python,
NumPy, PyTorch, and CUDA random state before model initialization and client
training.

Data partition seed and run seed are explicit in logs and output files.

## Outputs

The governance command writes:

- one summary CSV row per group;
- one JSON result containing configuration, accuracy histories, and
  Executor audit metadata;
- console logs for attack type, detection source, whether recovery was used,
  and round accuracy.

Output directories are created only when the experiment command runs.

## Validation

Automated CPU tests cover:

1. Scaling produces `global + 5 * delta`.
2. A1 changes published trust and is detected by V1.
3. A1 is internally aggregate-consistent and is not detected by V2.
4. A2 preserves published trust and is not detected by V1.
5. A2 changes the aggregate and is detected by V2.
6. `trust_only` accepts the malicious proposal.
7. V1, V2, and TAS recover exactly to the same-round baseline when their
   corresponding defense detects an attack.
8. With no Executor callback, `run_single()` uses the baseline aggregate and
   produces no Executor events.
9. The Sweep CLI uses the explicit stable seed.

A short CIFAR run then executes Sweep and governance `no_executor` with the
same seed and compares their round histories. When GPU access is unavailable
locally, the CPU unit suite and CLI smoke validation remain required, and the
GPU short-run command and limitation are reported explicitly.

## Non-Goals

- Reproducing the historical unrecorded effective seed or forcing `82.32`.
- Modifying the old CIFAR governance v9 script.
- Changing TASL trust-scoring mathematics.
- Adding client exclusion, rollback, or a third Executor attack.
- Refactoring MNIST or Fashion-MNIST governance code.
