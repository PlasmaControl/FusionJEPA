# Coding-agent handoff: Fusion-JEPA

**Purpose:** turn the research plan into a reproducible research codebase.  
**This handoff authorizes:** repository scaffolding, data adapters, models,
training/evaluation/control pipelines, tests, and Frontier job files.  
**It does not authorize:** changing the scientific claims, inventing dataset
semantics, publishing private DIII-D data, or silently expanding scope.

Read these before changing code:

1. [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)
2. [`../reference/D3D_world_model_preprint/main.md`](../reference/D3D_world_model_preprint/main.md)
3. The pinned upstream TokaMark task configs and evaluation implementation.
4. Current OLCF Frontier PyTorch and storage documentation before writing job
   scripts.

## Mission

Deliver the smallest codebase that can run the paper's decisive matched
experiments:

1. load official TokaMark data and preserve its split/metric contract;
2. train a matched raw-space predictor and action-conditioned JEPA;
3. detect collapse and action ignoring during training;
4. evaluate official tasks, low-label transfer, robustness, and action use;
5. train/evaluate latent model-predictive control in Gym-TORAX;
6. train one approximately 10–15M checkpoint jointly on MAST and DIII-D without
   learned device-specific adapters or modules;
7. plug in an internal DIII-D adapter without contaminating the public code or
   artifacts; and
8. reproduce every paper table from immutable run outputs.

Optimize for reliable iteration and traceability, not framework novelty.

## Non-goals for the first implementation

- Do not port the full 1.3B-parameter IGNITE stack.
- Do not implement all video and spectrogram modalities before the
  time-series/profile path passes its gates.
- Do not build a new distributed training framework.
- Do not build a new planner before CEM or MPPI is working.
- Do not copy or fork TokaMark preprocessing when an adapter can call it.
- Do not download data into the repository, a login-node home directory, or a
  CI runner.
- Do not add a loss term without a logged diagnostic and an ablation config.
- Do not access the official test split during model selection.

## Scientific invariants

These are acceptance requirements, not preferences.

1. **Split integrity:** no shot crosses train/validation/test; normalization is
   fit on training only; the official split remains byte-for-byte identifiable.
2. **Physical time:** horizons are represented in seconds/milliseconds, never
   only in sample indices.
3. **Action alignment:** an action sequence must cover the exact transition it
   conditions. Add boundary tests with synthetic ramps.
4. **Missingness is explicit:** zeros are values; masks state availability and
   validity separately.
5. **Units and semantics are explicit:** every canonical signal declares unit,
   role, modality, sampling information, and upstream source.
6. **Matched comparisons:** objective variants share data, backbone, parameter
   budget, optimizer budget, and evaluation.
7. **Shot-level statistics:** overlapping windows are never treated as
   independent replicates in paper confidence intervals.
8. **Causal restraint:** logged-data action swaps are diagnostic probes, not
   counterfactual ground truth.
9. **Public/private separation:** no private DIII-D samples, paths, credentials,
   shot identifiers, or unreleasable metadata enter public artifacts.
10. **Resumability:** every Frontier run can resume exactly from a checkpoint
    after wall-time termination.
11. **One multi-device model:** the headline joint experiment uses one parameter
    graph and checkpoint. Device identity may condition shared modules but may
    not select a device-specific tokenizer, encoder, predictor, head, LoRA, or
    fine-tuned weight set.
12. **Device balance:** every joint run controls and logs device sampling and
    per-device loss contributions; pooled metrics never hide a degraded device.

## Proposed repository layout

Create this structure incrementally. Empty placeholder packages are not useful;
each directory should arrive with a tested interface or first consumer.

```text
fusion_jepa/
├── README.md
├── pyproject.toml
├── uv.lock
├── environment.frontier.yaml
├── configs/
│   ├── data/
│   │   ├── tokamark.yaml
│   │   ├── gym_torax.yaml
│   │   └── d3d.example.yaml
│   ├── model/
│   │   ├── raw_predictor_small.yaml
│   │   ├── jepa_ema_small.yaml
│   │   └── jepa_sigreg_small.yaml
│   ├── experiment/
│   │   ├── mast_smoke.yaml
│   │   ├── mast_groups_2_3.yaml
│   │   ├── mast_d3d_joint.yaml
│   │   ├── torax_control_smoke.yaml
│   │   └── d3d_aligned.example.yaml
│   └── cluster/
│       ├── local.yaml
│       └── frontier.yaml
├── signal_registry/
│   ├── schema.yaml
│   ├── machine_context_schema.yaml
│   ├── mast.yaml
│   ├── torax.yaml
│   └── d3d.example.yaml
├── src/fusion_jepa/
│   ├── cli/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── probe.py
│   │   ├── control.py
│   │   └── inspect_data.py
│   ├── config.py
│   ├── data/
│   │   ├── batch.py
│   │   ├── registry.py
│   │   ├── splits.py
│   │   ├── transforms.py
│   │   ├── multidevice_sampler.py
│   │   ├── tokamark.py
│   │   ├── torax.py
│   │   └── d3d.py
│   ├── models/
│   │   ├── tokenizers.py
│   │   ├── semantic_tokens.py
│   │   ├── encoder.py
│   │   ├── action_encoder.py
│   │   ├── predictor.py
│   │   ├── decoders.py
│   │   ├── raw_world_model.py
│   │   └── jepa.py
│   ├── objectives/
│   │   ├── latent_prediction.py
│   │   ├── collapse_regularizers.py
│   │   ├── action_diagnostics.py
│   │   └── raw_prediction.py
│   ├── training/
│   │   ├── loop.py
│   │   ├── distributed.py
│   │   ├── checkpoint.py
│   │   └── ema.py
│   ├── evaluation/
│   │   ├── tokamark.py
│   │   ├── metrics.py
│   │   ├── probes.py
│   │   ├── robustness.py
│   │   ├── action_sensitivity.py
│   │   ├── multidevice.py
│   │   └── statistics.py
│   ├── control/
│   │   ├── environment.py
│   │   ├── costs.py
│   │   ├── cem.py
│   │   ├── mppi.py
│   │   └── evaluator.py
│   └── utils/
│       ├── reproducibility.py
│       ├── logging.py
│       └── manifests.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── frontier/
├── scripts/
│   ├── acquire_tokamark.py
│   ├── build_dev_subset.py
│   ├── reproduce_table.py
│   └── reproduce_figure.py
├── slurm/
│   ├── smoke.slurm
│   ├── train_single_node.slurm
│   ├── train_multinode.slurm
│   └── eval_array.slurm
├── manifests/
│   ├── upstream.yaml
│   ├── splits/
│   └── datasets/
├── results/
│   └── README.md
└── docs/
```

`results/` should contain small immutable summaries/manifests only. Large
checkpoints, predictions, and data stay on configured external storage.

## Packaging and dependency policy

- Use a normal `src/` Python package and `pyproject.toml`.
- Support Python 3.11 unless the active Frontier environment requires another
  version.
- Use PyTorch directly. Avoid PyTorch Lightning until raw DDP works on
  Frontier; the extra abstraction is not needed for the first paper.
- Use typed structured configs (dataclasses plus OmegaConf/Hydra is acceptable)
  with a fully resolved YAML snapshot saved in every run.
- Required data stack: `numpy`, `scipy`, `pandas`, `xarray`, `zarr`, `fsspec`,
  `s3fs`, and the pinned TokaMark package.
- Required evaluation stack: `scikit-learn` and a tested bootstrap
  implementation.
- Keep experiment tracking backend-agnostic. JSONL/CSV and local artifacts are
  mandatory; Weights & Biases or another service may be optional but cannot be
  required to reproduce a run.
- Pin exact upstream commits and dataset revisions in
  `manifests/upstream.yaml`; do not depend on floating `main` branches.
- Export a lock or exact package list for local CPU tests and the tested
  Frontier ROCm environment.

## CLI contract

The public entry points should be stable from the first end-to-end milestone:

```bash
python -m fusion_jepa.cli.inspect_data experiment=mast_smoke
python -m fusion_jepa.cli.train experiment=mast_smoke seed=0
python -m fusion_jepa.cli.evaluate run=/absolute/path/to/run split=validation
python -m fusion_jepa.cli.probe run=/absolute/path/to/run probe=low_label
python -m fusion_jepa.cli.control experiment=torax_control_smoke seed=0
```

Requirements:

- all path/account overrides are explicit config values or environment inputs;
- `--help` explains inputs and output artifacts;
- a dry-run/validate mode resolves config, checks storage, prints dataset/model
  sizes, and exits without allocating training compute;
- training refuses to evaluate `test` unless an explicit evaluation-only flag
  and frozen checkpoint are supplied; and
- every command logs one machine-readable completion record and a non-zero
  exit code on failure.

## Canonical batch contract

All device adapters must emit the same semantic structure. Use a dataclass or
`TensorDict`, with documented shapes.

```python
FusionBatch(
    context={modality_name: Tensor[B, ...]},
    context_mask={modality_name: BoolTensor[B, ...]},
    target={modality_name: Tensor[B, H, ...]},
    target_mask={modality_name: BoolTensor[B, H, ...]},
    actions=Tensor[B, H, A],
    action_mask=BoolTensor[B, H, A],
    context_times=Tensor[B, T_context],
    target_times=Tensor[B, H],
    horizon_seconds=Tensor[B, H],
    device_id=LongTensor[B],
    device_context=Tensor[B, K_device],
    device_context_mask=BoolTensor[B, K_device],
    shot_id=list[str],
    window_id=list[str],
    metadata=dict,
)
```

The exact tensor ranks can differ by modality, but semantics may not. Never
overload a zero-filled tensor to mean both missing and physically zero.

### Batch validation

On construction, a development-only validator must check:

- finite values wherever masks are true;
- monotonically increasing physical times;
- exact context/action/target alignment;
- declared units and canonical names;
- no duplicate window IDs in a batch;
- target windows remain within their shot; and
- no split mismatch for the shot ID.

Make this validator cheap enough for smoke jobs and sample batches, then allow
it to be disabled for full training after the dataset manifest is certified.

## Signal registry

Use a registry instead of encoding device-specific names in model code. Each
entry must include:

```yaml
canonical_name: electron_density_profile
semantic_quantity: electron_density
device: mast
source: thomson_scattering
upstream_key: n_e
native_signal_id: mast_thomson_ne
role: diagnostic
modality: radial_profile_timeseries
unit: m^-3
sample_rate_hz: null
normalization: standard
valid_range: null
radial_coordinate: upstream
notes: null
```

Action entries additionally specify command versus measured response, bounds,
rate limits if known, and whether the channel is safe to perturb in simulation.

Registry rules:

- one canonical name cannot silently map to different physical meanings;
- unit conversion happens in the adapter and is tested;
- device-specific channels may remain device-specific;
- only physics-reviewed mappings may be labeled `shared_core`; all uncertain
  or non-overlapping channels remain explicitly device-specific; and
- a physics collaborator must approve the first registry version.

## Joint multi-device training contract

The headline multi-device run trains on MAST and DIII-D concurrently and emits
one checkpoint. It is not an unseen-device transfer experiment.

### Permitted device-specific components

- native-to-canonical signal mapping;
- unit and coordinate conversion;
- training-only normalization statistics;
- signal, diagnostic, device, and modality identity embeddings;
- fixed physical machine metadata and shot configuration; and
- masks for absent or invalid signals.

These are inputs to shared weights. They may not select separate learned
networks.

### Prohibited in the headline model

- per-device tokenizers, encoders, predictors, output heads, LoRA modules, or
  checkpoints;
- parameter dictionaries keyed by device;
- branches that route MAST and DIII-D through different learned paths; and
- device-specific objectives disguised as one training job.

Implement an automated model audit that fails the headline config if a module
or trainable parameter is registered under a device-specific name or if the
active parameter set changes with `device_id`.

### Semantic tokens

Every token must carry or be able to recover:

- value or local patch;
- physical time and horizon;
- canonical physical quantity;
- native signal/diagnostic identity;
- observation, action, or target role;
- modality;
- physical/profile/spatial/frequency coordinates where applicable;
- validity and availability;
- device ID; and
- physical machine and shot context.

Use one learned tokenizer per modality family, shared across devices. Signal
embeddings distinguish quantities and native diagnostics; they do not choose a
different tokenizer. Aggregate the variable token set into a fixed 8–16 state
token bottleneck for prediction and planning.

### Machine context

`machine_context_schema.yaml` defines a fixed ordered vector and units for
reviewed static quantities such as major/minor radius, aspect ratio, field
convention or nominal scale, and capability flags. Shot-varying field, current,
shape, heating, and fueling settings remain state/action inputs.

Train these four declared variants through configuration, not code branches:

1. no device context;
2. opaque device ID only;
3. physical machine context only; and
4. both device ID and physical context.

An unknown-device ID may exist for future work, but do not claim or tune
zero-shot behavior in the current paper.

### Shared signal surface

The registry must label every signal as one of:

- `shared_core`: physics-reviewed semantic match across devices;
- `shared_modality`: different signal semantics but processed by the same
  tokenizer family;
- `device_specific`: available only on one device or not honestly alignable;
  or
- `excluded`: ambiguous, poor quality, leaked target, or out of scope.

Version 1 should include approximately 8–15 shared scalar quantities, reviewed
density/temperature profiles on a common coordinate, and only later one
comparable spectrogram family. Do not block joint training on perfect alignment
of every IGNITE modality.

### Sampler and loss balancing

The simplest correct implementation alternates homogeneous-device batches
while updating one shared optimizer. `MultiDeviceBatchSampler` must:

- accept explicit device probabilities or tempered sampling
  `p_d ∝ N_d^alpha`, with `alpha` recorded in config;
- derive deterministic per-device sampler seeds;
- expose cumulative examples, tokens, targets, and optimizer contribution by
  device;
- support distributed sharding without duplicating shots; and
- resume its device and sample sequence from checkpoint.

Loss reduction order is target/channel → modality → example → device →
joint scalar. Save both unreweighted and weighted device losses. Use
LayerNorm/RMSNorm; do not add device-dependent BatchNorm statistics.

### Initial multi-device experiment matrix

| ID | Training setup | Deployed parameters | Purpose |
|---|---|---:|---|
| MD00 | MAST specialist | ~12M | per-device capacity-matched reference |
| MD01 | DIII-D specialist | ~12M | per-device capacity-matched reference |
| MD02 | MAST specialist | ~6M | equal-total-footprint reference |
| MD03 | DIII-D specialist | ~6M | equal-total-footprint reference |
| MD04 | joint, no device context | ~12M | test whether conditioning is necessary |
| MD05 | joint, device ID only | ~12M | opaque conditioning baseline |
| MD06 | joint, physical context only | ~12M | interpretable conditioning baseline |
| MD07 | joint, ID plus physical context | ~12M | proposed headline model |
| MD08 | MD07, shared core only | ~12M | signal-overlap ablation |
| MD09 | MD07, reduced shots on one device | ~12M | joint-data-sharing curve |

All joint and specialist comparisons use the same per-device split, target
surface, per-device data exposure, total token/compute accounting, and
evaluation code. Declare the per-device non-inferiority margins before test
evaluation. Paper tables show each device separately before any pooled score.

## TokaMark integration

Treat upstream TokaMark as the authority for task definitions, official split,
and official metric aggregation.

### Adapter behavior

- Import the installed, pinned `tokamark` package.
- Load the upstream YAML for the requested task.
- Preserve upstream shot IDs and split membership.
- Convert its input/actuator/output objects into `FusionBatch`.
- Preserve masks and distinguish upstream imputation from true values.
- Make window sizes/horizons visible in physical units.
- Call or wrap the upstream `WindowMetricsAccumulator` for official metrics.
- Save the exact upstream task config with each run.

### Do not

- copy the 14 task YAML files and let them drift;
- recompute an "equivalent" split without matching the upstream manifest;
- change normalization or missing-data handling inside the official comparison;
- mix task-specific target labels into unsupervised pretraining context; or
- unpack 2 TB into a default developer path.

### Data acquisition

The acquisition script must:

1. require a destination outside the repository;
2. print expected compressed and unpacked sizes before starting;
3. support resume and checksum verification;
4. record source revision, timestamps, file list, sizes, and checksums;
5. optionally use remote Zarr for the sample smoke test;
6. never delete or overwrite an existing dataset without an explicit flag; and
7. create a small, deterministic development subset after full validation.

## DIII-D integration

Keep the adapter interface public/releasable while keeping data configuration
private.

- `d3d.example.yaml` documents shapes and required semantic fields with fake
  paths and no protected identifiers.
- Real configs live in a gitignored/private location supplied at runtime.
- The adapter reads an existing prepared IGNITE dataset; do not reimplement raw
  DIII-D acquisition in this repository.
- Split manifests and normalization statistics are external inputs with hashes.
- Provide a small synthetic fixture with the same tensor/mask shapes for CI.
- Log only release-approved aggregate metadata.
- Fail closed if a public-output flag is requested while a configured artifact
  contains private shot/window identifiers.
- Treat every checkpoint trained partly on DIII-D, including the joint
  MAST/DIII-D checkpoint, as private by default until data-governance and
  memorization/release review explicitly approves it.

Start with the aligned low-dimensional surface from the research plan. Add
spectrograms/video behind explicit config flags after the first gates pass.

## Gym-TORAX integration

Separate three concepts:

1. simulator environment used to generate/evaluate trajectories;
2. immutable offline trajectory dataset used to train learned models; and
3. planner using a frozen learned world model.

The control contract must be a versioned config containing:

- Gym-TORAX and TORAX revisions;
- environment/scenario name;
- observation and action definitions with units;
- target trajectory;
- hard and soft constraints;
- reward/cost decomposition;
- control interval and episode duration;
- nominal and shifted simulator parameter sets;
- trajectory-generation policy mixture and seeds; and
- evaluation episode seeds.

Never tune on the final control seeds. Record simulator termination reasons and
constraint events separately from reward.

## Model interfaces

### Modality tokenizer

```python
tokens, token_mask, token_metadata = tokenizer(values, value_mask, times)
```

`token_metadata` must retain modality, signal/channel identity, physical time,
and spatial/profile coordinates needed for positional encoding and analysis.

Initial supported modality classes:

1. slow multichannel time series;
2. fast multichannel time series;
3. radial/profile time series; and
4. equilibrium/image-like arrays only when required for the first selected
   TokaMark tasks.

Spectrogram and video tokenizers may reuse IGNITE design ideas later.

### Context and target encoder

```python
z, z_mask, z_metadata = encoder(tokens, token_mask, token_metadata)
```

The target encoder must have an explicit update policy:

- `ema`: no gradient, update after successful optimizer step;
- `shared_stopgrad`: shared parameters, target output stopped; or
- `end_to_end_regularized`: separate documented stable objective.

Do not mix policies implicitly.

### Action encoder

```python
u_tokens, u_mask = action_encoder(actions, action_mask, action_times)
```

It must preserve actuator identity and temporal order. Add a test showing that
permuting action channels or shifting time changes the encoded sequence.

### Predictor

```python
z_hat = predictor(
    context_latents=z,
    context_mask=z_mask,
    action_tokens=u_tokens,
    action_mask=u_mask,
    horizons=horizon_seconds,
    device_id=device_id,
    device_context=device_context,
    device_context_mask=device_context_mask,
)
```

Support direct multi-horizon prediction first. Add autoregressive rollout as a
separate method so one-step and rollout behavior can be compared cleanly.

### Raw-space baseline

Use the same tokenizers, encoder, action encoder, backbone/predictor capacity,
and training examples. Add decoders and train against future raw targets. The
matched baseline must not be an older, smaller, or under-tuned code path.

For the multi-device experiment, decoding must be query-conditioned on semantic
quantity, native signal identity, coordinate, and device context through one
shared decoder. Do not create a MAST output network and a DIII-D output network.

## Objective implementation

Every objective returns:

```python
LossOutput(total, terms: dict[str, Tensor], diagnostics: dict[str, Tensor])
```

### Base latent prediction

- normalized latent cosine or smooth-L1 distance;
- mask-aware averaging;
- explicit horizon weights;
- no target-branch gradient for EMA/stop-gradient variants; and
- per-horizon loss logging.

### Collapse diagnostics, always logged

- per-dimension standard deviation;
- covariance off-diagonal magnitude;
- covariance eigenvalue/effective-rank estimate;
- batch mean norm and latent norm distribution;
- predictor-to-target variance ratio; and
- fraction of near-constant dimensions.

Set warning thresholds in config. A warning must be visible in the run summary;
do not automatically hide collapse by increasing a regularizer.

### Action-use diagnostics, always available

- prediction loss with real actions;
- loss with batch-shuffled actions;
- loss with within-shot time-shifted actions;
- loss with zero actions;
- norm/Jacobian sensitivity of predictions to normalized action perturbations;
  and
- optional inverse-action prediction from latent displacement.

The shuffled variants are evaluation diagnostics. Do not contaminate the main
loss unless a pre-declared method variant uses them.

## Training loop requirements

- Raw PyTorch DDP first; add FSDP only if model memory requires it.
- bf16 where verified on Frontier; unit-test numerically sensitive metrics in
  fp32.
- gradient accumulation configured in samples/tokens, not hidden magic.
- global effective batch size logged.
- deterministic seed derivation for Python, NumPy, PyTorch, dataloader workers,
  distributed sampler, and simulator.
- gradient norm, learning rate, tokens/s, samples/s, data wait time, GPU memory,
  and wall time logged.
- joint runs log examples/tokens, weighted and unweighted loss, gradient norm,
  and validation metrics separately by device; periodically estimate
  per-device gradient cosine on a fixed diagnostic batch.
- validation at fixed optimizer-step intervals.
- best validation checkpoint and most recent resumable checkpoint stored
  separately.
- atomic checkpoint writes through temporary file plus rename.
- checkpoint includes model, target encoder, optimizer, scheduler, scaler,
  sampler/data cursor where possible, step, epoch, RNG states, resolved config,
  upstream manifests, and git commit.
- graceful pre-walltime save signal in Slurm scripts.
- no test evaluation inside training jobs.

## Frontier requirements

Check the live OLCF docs before implementation; module versions drift.

Current architecture assumptions to encode in comments/config validation, not
hard-coded package pins:

- a Frontier node exposes eight MI250X GCDs as eight GPUs to Slurm/ROCm;
- use all eight GPUs per node for normal training;
- launch one task per GPU with CPU/GPU affinity through `srun`, following the
  OLCF PyTorch example;
- use `--gpu-bind=closest` and the documented CPU-core binding;
- do not use a naive `torchrun --nproc_per_node=8` launch that pins processes to
  one NUMA region;
- use the OLCF-supported RCCL/network plugin for multi-node jobs; and
- verify rank/GPU uniqueness in a tiny Frontier integration test.

### Storage

- Store datasets, checkpoints, and run outputs in the appropriate Orion
  member/project work area, not home.
- Use project work for team-shared public data and member work for restricted
  or individual data as policy requires.
- Orion is purged and not backed up; archive release checkpoints and manifests
  separately.
- Avoid millions of tiny files. Prefer consolidated Zarr metadata or larger
  immutable shards, and benchmark stripe/layout choices before mass
  conversion.
- Keep a local per-node/process cache only when it improves measured
  throughput and has safe cleanup.
- Record I/O wait separately from compute throughput.

### Required Slurm files

1. `smoke.slurm`: one node, one short batch sequence, all eight GPUs visible and
   uniquely bound.
2. `train_single_node.slurm`: resumable full training on eight GPUs.
3. `train_multinode.slurm`: explicit nodes/ranks, RCCL setup, rank diagnostics,
   checkpoint/requeue behavior.
4. `eval_array.slurm`: one frozen checkpoint/config per array element, no
   duplicated test writes.

Do not include a real project account, username, private path, or credentials in
committed scripts.

OLCF references:

- [PyTorch on Frontier](https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html)
- [Frontier user guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
- [Data storage and Orion](https://docs.olcf.ornl.gov/data/index.html)

## Run artifact contract

Each run directory is immutable after completion except for an explicit
`annotations.jsonl` file. It contains:

```text
run_id/
├── config.resolved.yaml
├── command.txt
├── environment.txt
├── git.json
├── upstream_manifest.yaml
├── dataset_manifest.yaml
├── split_manifest.yaml
├── metrics.jsonl
├── validation_summary.json
├── completion.json
├── checkpoints/
└── artifacts/
```

`completion.json` contains status, start/end times, runtime, accelerator-hours,
best checkpoint, primary validation metric, warnings, and failure reason.

Use a deterministic human-readable run name plus a unique hash, for example:

```text
mast-g23_jepa-ema-small_h5-25-50_seed2__a1b2c3d4
```

The hash must depend on the resolved scientific config and code/upstream
revision, not output path or timestamp alone.

## Evaluation implementation

### Official TokaMark

- Call upstream official metrics.
- Save raw per-window predictions only outside git.
- Save upstream-compatible per-window, per-shot, per-signal, per-task, and
  per-group summaries.
- Verify persistence and official baseline outputs on the sample data before
  evaluating a new model.

### Low-label transfer

- Freeze one pretrained checkpoint.
- Construct deterministic nested labeled subsets at 1%, 5%, 10%, and 100% by
  shot, not window.
- Compare frozen linear probe, small MLP probe, partial fine-tune, full
  fine-tune, and scratch where budget allows.
- Match optimizer-step/search budgets per fraction.
- Report area under the error-versus-data-fraction curve.

### Robustness

Implement corruption as pure transforms with versioned configs:

- random channel dropout;
- full modality dropout;
- contiguous final-context dropout;
- noise and calibration drift;
- actuator loss; and
- simulator parameter shift.

Record corruption masks so exactly the same corrupted examples are used across
models.

### Action sensitivity

Action shuffle must support:

- batch shuffle;
- same-time cross-shot shuffle;
- within-shot time shift; and
- per-actuator ablation.

Preserve action marginal scales where possible. Report predictive gain with
shot-level intervals. Label all logged-data results as associational.

### Statistics

- paired bootstrap over shots/episodes;
- configurable number of resamples and deterministic seed;
- task aggregation defined in one module;
- multiple-comparison handling documented when making per-task significance
  claims; and
- table generator consumes completed run summaries, never hand-entered values.

### Multi-device joint model

Generate a table with one block per device and these columns:

- approximately 12M specialist;
- approximately 6M specialist;
- joint model without conditioning;
- joint model with device ID;
- joint model with physical machine context;
- joint model with both; and
- joint model with shared-core signals only.

For every row report parameter count, total family footprint, per-device
training examples/tokens, accelerator-hours, primary task metric, confidence
interval, and gap to the matched approximately 12M specialist. The pass/fail
aggregate is the worst-device gap, not the pooled mean.

The reduced-data study holds the other device fixed and trains with predeclared
fractions of shots from the reduced device. It is evidence about data sharing
inside joint training, not zero-shot transfer.

Add an automated checkpoint audit to the result artifact containing:

- one checkpoint hash;
- full trainable parameter-name list;
- device-conditioned active-parameter-set comparison;
- absence of device-keyed learned modules; and
- registry and machine-context hashes.

## Control implementation

### World-model adapter

Expose a planner-facing interface:

```python
state = model.encode_observation(observation, mask)
next_state = model.predict_state(state, candidate_action_sequence, horizons)
cost_terms = cost_model(next_state, candidate_action_sequence, target)
```

The raw-space baseline must expose the same interface through decoded states.

### Planner requirements

- vectorized candidate rollouts;
- bounded actions and optional rate constraints;
- warm-start from the prior MPC solution;
- fixed random seeds/candidate counts for matched comparisons;
- explicit planning horizon and model time step;
- latency broken into encoding, rollout, cost, and selection; and
- invalid/NaN rollout handling that counts as failure, not silent clipping.

Start with CEM. Add MPPI only after one end-to-end control result is verified.

### Control evaluator

For every episode store:

- target, observation, action, cost components, constraint values, and
  termination reason over time;
- simulator/config/model/planner revisions;
- random seed and parameter-shift condition;
- model calls and wall time; and
- compact release-safe trajectory summary.

The paper table must be generated from these episode records.

## Test plan

### Unit tests required before full training

1. unit conversion and registry schema validation;
2. context/action/target window alignment using synthetic ramps;
3. masks distinguish missing from zero;
4. no shot crosses split manifests;
5. normalization ignores masked values and uses training only;
6. tokenizer shape/mask propagation for every supported modality;
7. action channel permutation/time shift changes encoded actions;
8. target encoder receives no gradient in EMA mode;
9. EMA update matches a hand-computed step;
10. latent loss ignores masked tokens and respects horizon weights;
11. collapse metrics identify constant, rank-one, and healthy synthetic
    latents;
12. action-shuffle diagnostics preserve expected shapes/marginals;
13. checkpoint round trip reproduces the next optimization step;
14. bootstrap treats shots, not windows, as samples;
15. planner respects action and rate bounds;
16. run hash changes for scientific config/code changes;
17. shared-core registry entries agree in semantic quantity, unit, and reviewed
    coordinate convention across devices;
18. one model instance processes MAST and DIII-D batches with different signal
    inventories;
19. changing device conditioning changes inputs but not the active parameter
    set;
20. the headline config contains no device-keyed learned module; and
21. multi-device sampling/loss weights match hand-computed toy examples and
    resume exactly.

### Integration tests

- TokaMark sample data → one batch → model → loss → optimizer → validation
  metric.
- One CPU/single-GPU tiny run resumes from checkpoint and produces identical
  final weights within tolerance.
- Gym-TORAX tiny trajectory generation → offline dataset → one model update →
  one CEM action → one evaluated step.
- Synthetic DIII-D-shaped fixture passes through the public adapter.
- Alternating synthetic MAST/DIII-D batches update the same checkpoint and
  produce separate per-device metrics.
- The joint checkpoint audit proves identical parameter names/active parameters
  for both devices.
- Distributed two-process local test produces the same global batch/loss as a
  single process.
- Frontier smoke verifies eight unique ranks/GPUs and a successful all-reduce.

### Regression tests

Store tiny, license-safe fixtures and expected metric summaries. Do not store
real private DIII-D samples or large TokaMark fragments in git.

## Experiment registry

Create a versioned table (YAML/CSV) with one row per paper-relevant run:

- experiment ID and claim;
- hypothesis;
- dataset/split/task;
- model/objective;
- seed list;
- primary metric and pass threshold;
- compute request;
- dependencies;
- owner;
- status;
- run IDs; and
- conclusion.

A job is not paper evidence unless it has a registry row created before test
evaluation. Failed runs remain in the registry.

## Initial experiment matrix

Run on the smallest representative group 2/3 subset before scaling.

| ID | Model | Actions | Objective | Horizons | Purpose |
|---|---|---|---|---|---|
| E00 | persistence | n/a | n/a | task native | pipeline/metric floor |
| E01 | official CNN | task native | raw MSE | task native | upstream reproduction |
| E02 | matched Transformer | yes | raw prediction | single | main objective baseline |
| E03 | matched Transformer | no | EMA JEPA | single | latent/no-action baseline |
| E04 | matched Transformer | yes | EMA JEPA | single | core action test |
| E05 | matched Transformer | yes | EMA JEPA | multi | multi-horizon test |
| E06 | matched Transformer | yes | end-to-end regularized JEPA | multi | collapse strategy |
| E07 | E05 | shuffled | evaluation only | multi | action utilization |
| E08 | E05 | yes | short-rollout fine-tune | multi | control readiness |

Only after E00–E05 are stable should full-data and scale sweeps start.

## Milestones and acceptance tests

### M0 — scaffold (1–2 days)

- package installs locally;
- structured config resolves;
- CPU CI runs lint/type/unit smoke;
- run artifact and manifest utilities work; and
- no data/model implementation placeholders are presented as complete.

### M1 — public data contract (2–4 days)

- TokaMark upstream revision pinned;
- sample-data batch inspected and plotted;
- official split/task config preserved;
- synthetic alignment/mask tests pass; and
- acquisition/dev-subset scripts are resumable and manifest-producing.

### M2 — baselines (2–4 days)

- persistence and official CNN sample result reproduced;
- matched raw-space small baseline trains/evaluates;
- all official metrics and shot-level summaries generated; and
- I/O and GPU utilization measured.

### M3 — JEPA smoke (3–5 days)

- EMA target behavior tested;
- small action-conditioned JEPA trains without collapse;
- real versus shuffled/no-action diagnostic generated;
- raw and latent models are parameter/compute matched; and
- checkpoint resume verified.

### M4 — public experiment engine (1 week)

- groups 2/3 run over planned seeds;
- low-label and robustness pipelines complete;
- tables/figures generated from run artifacts; and
- test access is auditable.

### M5 — public control (1 week, parallel after M2)

- versioned Gym-TORAX control contract;
- immutable offline trajectories;
- baseline controller and raw world-model planner;
- latent CEM controller;
- nominal/shift/dropout episode suite; and
- control table plus trajectories generated automatically.

### M6 — DIII-D and joint multi-device model (1–2 weeks after adapter exists)

- aligned signal registry approved;
- split/normalization manifests frozen;
- scratch/raw/JEPA comparison complete;
- one approximately 10–15M checkpoint alternates MAST/DIII-D batches and passes
  the no-device-specific-module audit;
- MD00–MD08 specialist, conditioning, and signal-surface comparisons complete;
- release-safe aggregate artifact generated; and
- no private information in public result bundle.

### M7 — release reproducibility

- clean environment reproduces one public headline result;
- anonymous/public code bundle contains exact configs;
- README documents storage requirements and sample workflow;
- checkpoint/license/model card complete; and
- paper table/figure scripts require no manual numbers.

## Ready-to-file implementation tickets

### Ticket 1 — repository, config, manifests, and run artifacts

Implement M0 with tests and stable CLI shells. No model logic.

### Ticket 2 — pin and certify TokaMark sample pipeline

Pin upstream repositories/dataset revision, build adapter, validate group 2/3
sample batches, preserve official metrics, and add alignment/mask/split tests.

### Ticket 3 — data acquisition and Frontier I/O benchmark

Create resumable acquisition/verification/dev-subset scripts, stage to Orion,
benchmark remote versus local Zarr and loader settings, and write a short
throughput report before full training.

### Ticket 4 — matched small raw-space baseline

Implement shared tokenizers/encoder/action encoder/predictor plus raw decoders;
train one task end to end and establish parameter/compute accounting.

### Ticket 5 — EMA JEPA core

Implement online/EMA target encoders, multi-horizon predictor, mask-aware latent
loss, EMA tests, and checkpoint support.

### Ticket 6 — collapse and action-use diagnostics

Implement rank/variance/covariance diagnostics and real/zero/shuffled/shifted
action evaluation with synthetic correctness tests.

### Ticket 7 — low-label and robustness evaluation

Build nested shot subsets, frozen/fine-tuned probes, corruption transforms,
shot bootstrap, and table generation.

### Ticket 8 — Gym-TORAX control contract and dataset

After physics/control approval, pin simulator versions, implement trajectory
generation, manifests, episode storage, and baseline controller evaluation.

### Ticket 9 — latent CEM planner

Implement vectorized, bounded receding-horizon planning through frozen raw and
latent models; log cost/safety/latency components.

### Ticket 10 — DIII-D aligned adapter

Implement the public interface and synthetic fixture, then connect private
runtime config outside git. Produce only release-approved aggregates.

### Ticket 11 — shared semantic registry and joint multi-device engine

Implement the reviewed shared-core/device-specific registry labels, machine
context schema, variable-token input path, balanced/resumable sampler,
per-device loss accounting, model audit, and MD00–MD09 configs.

### Ticket 12 — Frontier single/multi-node jobs

Use live OLCF guidance, verify rank/GPU affinity, resume/requeue, data paths,
RCCL, and completion artifacts. Add a documented smoke result.

### Ticket 13 — paper artifact builder

Build declarative table/figure specs that consume registered completed runs and
emit plots, CSV, and LaTeX/Markdown summaries without manual values.

## Coding style and review rules

- Type all public interfaces and scientific tensor shapes in docstrings.
- Prefer small pure transforms and explicit state over hidden global state.
- Assertions guard scientific invariants; user/config errors get actionable
  exceptions.
- Tests accompany every mask, split, alignment, EMA, metric, or planner change.
- Do not catch and ignore NaNs, missing files, failed ranks, or partial
  checkpoints.
- Keep data preprocessing deterministic and separately versioned from model
  code.
- One pull request/ticket should have one scientific purpose and a stated
  acceptance test.
- Benchmark before optimizing; record throughput changes.
- Preserve failed or negative scientific results in the experiment registry.
- Any change to a paper metric, split, target, action, constraint, or aggregate
  requires research-lead review.

## Handoff questions the coding agent must escalate

Stop and ask rather than assume if any of these is unresolved:

1. Which exact TokaMark groups/tasks form the first full run?
2. Which Gym-TORAX observations, actions, targets, and hard constraints were
   approved by the control team?
3. Which collapse strategy is the primary model versus an ablation?
4. Which DIII-D signals and aggregate artifacts are release-approved?
5. Which signals are `shared_core`, which coordinate/units conventions were
   physics-approved, and which device-specific extras enter MD07?
6. Which static machine-context fields and device-sampling probabilities were
   approved for the joint run?
7. What project account/storage areas should the uncommitted Frontier config
   use?
8. Has the official ICLR 2027 guide changed the release/anonymity schedule?

All other ordinary engineering decisions should be made autonomously within
this specification and documented in the relevant config or decision record.
