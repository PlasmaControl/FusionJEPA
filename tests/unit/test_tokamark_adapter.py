"""Tests for the TokaMark -> FusionBatch adapter (Task 1.5).

Offline tests drive the adapter through a fake upstream (monkeypatched
``_upstream`` shim) so the pure translation ``to_fusion_sample`` and the
dataset/split/metrics plumbing are exercised without any data or network
access. The ``@pytest.mark.remote_data`` tests stream a real batch from the
anonymous S3 store and are skipped by default (see the ``addopts`` marker
filter in pyproject.toml).
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from fusion_jepa.data.batch import collate_fusion, validate_batch
from fusion_jepa.data.registry import load_registry
from fusion_jepa.data.tokamark import (
    OfficialMetrics,
    TokamarkAdapterError,
    load_task_config,
    make_dataset,
    official_split,
    to_fusion_sample,
    _upstream,
)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate the Fusion-JEPA Git repository")


REGISTRY = load_registry(_repo_root() / "signal_registry" / "mast.yaml")


# ----------------------------------------------------------------------------
# Fake-upstream builders (offline)
# ----------------------------------------------------------------------------
def _minimal_config() -> dict:
    """A tiny markovian forecasting task config with real registry keys."""
    input_keys = [["summary", "ip"]]
    actuator_keys = [["pf_active", "coil_voltage"]]
    output_keys = [["equilibrium", "q95"]]
    return {
        "task_name": "task_2-3",
        "task_type": "markovian",
        "sources_and_signals": {
            "input_name": input_keys,
            "actuator_name": actuator_keys,
            "output_name": output_keys,
        },
        "task_window_segmenter": {
            "input_keys": input_keys,
            "actuator_keys": actuator_keys,
            "output_keys": output_keys,
            "input_length": 0.004,
            "output_length": 0.003,
            "delta": 0.0,
        },
        "stride_window": 0.001,
    }


def _entry(times: np.ndarray, values: np.ndarray) -> dict:
    return {"time": np.asarray(times), "values": np.asarray(values)}


def _window_from_config(
    config: dict,
    *,
    shot_id: int = 100,
    window_index: int = 0,
    t_cut: float = 0.30,
    dt: float = 0.001,
):
    """Build a full, self-consistent markovian-forecasting window.

    Raw per-signal grids are uniform with spacing ``dt``; the adapter rebuilds
    the reference axes from ``(dt, n, t_cut)`` anchored at ``t_cut``. ``grids``
    holds the float64 axes the adapter is therefore expected to produce.
    """
    seg = config["task_window_segmenter"]
    ik = [f"{s}-{k}" for s, k in (seg["input_keys"] or [])]
    ak = [f"{s}-{k}" for s, k in (seg["actuator_keys"] or [])]
    ok = [f"{s}-{k}" for s, k in (seg["output_keys"] or [])]
    in_len = float(seg["input_length"])
    out_len = float(seg["output_length"])
    delta = float(seg["delta"])

    n_ctx = max(2, round(in_len / dt))
    n_tgt = max(2, round(out_len / dt))
    n_act = max(2, round((in_len + delta + out_len) / dt)) if ak else 0

    def ramp(times: np.ndarray) -> np.ndarray:
        return times.astype(np.float64)[None, :].copy()

    ctx_raw = t_cut - in_len + dt * np.arange(n_ctx, dtype=np.float64)
    tgt_raw = t_cut + dt * np.arange(n_tgt, dtype=np.float64)
    act_raw = t_cut - in_len + dt * np.arange(n_act, dtype=np.float64)

    window = {
        "shot_id": shot_id,
        "window_index": window_index,
        "t_cut": float(t_cut),
        "input": {key: _entry(ctx_raw, ramp(ctx_raw)) for key in ik},
        "actuator": {key: _entry(act_raw, ramp(act_raw)) for key in ak},
        "output": {key: _entry(tgt_raw, ramp(tgt_raw)) for key in ok},
    }
    grids = {
        "context": t_cut - dt * (n_ctx - np.arange(n_ctx, dtype=np.float64)),
        "target": (t_cut + delta) + dt * np.arange(n_tgt, dtype=np.float64),
        "action": (
            (t_cut - in_len) + dt * np.arange(n_act, dtype=np.float64)
            if ak
            else None
        ),
        "dt": dt,
    }
    return window, grids


# ----------------------------------------------------------------------------
# 1. Task config: deep copy + provenance hash
# ----------------------------------------------------------------------------
def test_task_config_copied_and_hash_recorded() -> None:
    cfg = load_task_config("task_2-3")

    assert cfg.task_id == "task_2-3"
    assert cfg.config["task_name"] == "task_2-3"

    # Deep copy: mutating the returned config must not leak into a re-load.
    cfg.config["task_window_segmenter"]["input_length"] = 999.0
    fresh = load_task_config("task_2-3")
    assert fresh.config["task_window_segmenter"]["input_length"] != 999.0

    source = Path(cfg.source_path)
    assert source.exists()
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert cfg.source_sha256 == expected


# ----------------------------------------------------------------------------
# 2 + 3. Split membership is preserved and out-of-split shots are impossible
# ----------------------------------------------------------------------------
def _write_split_csv(path: Path) -> None:
    lines = ["shot_id,train,val,test"]
    for shot in (1, 2, 3):
        lines.append(f"{shot},True,False,False")
    for shot in (4, 5):
        lines.append(f"{shot},False,True,False")
    lines.append("6,False,False,True")
    path.write_text("\n".join(lines) + "\n")


class _FakeMast:
    def __init__(self, shots_list):
        self.shots_list = list(shots_list)

    def __getitem__(self, position):
        # A raw shot with generous bounds that contain every fake window.
        times = np.linspace(0.0, 1.0, 16)
        values = np.ones((1, times.size))
        return {
            key: _entry(times, values)
            for key in (
                "summary-ip",
                "pf_active-coil_voltage",
                "equilibrium-q95",
            )
        }


class _FakeTokaMark:
    """Iterable that yields one window per shot id."""

    def __init__(self, shots, config, override_shot=None):
        self._shots = list(shots)
        self._config = config
        self._override = override_shot

    def __iter__(self):
        shots = self._shots
        if self._override is not None:
            shots = shots + [self._override]
        for idx, shot in enumerate(shots):
            window, _ = _window_from_config(
                self._config, shot_id=shot, window_index=idx
            )
            yield window


def _patch_minimal_upstream(monkeypatch, tmp_path, *, override_shot=None):
    config = _minimal_config()
    csv = tmp_path / "splits.csv"
    _write_split_csv(csv)
    monkeypatch.setattr(_upstream, "split_csv_path", str(csv))
    monkeypatch.setattr(_upstream, "get_task_config", lambda task_id: config)
    monkeypatch.setattr(_upstream, "get_task_metadata", lambda cfg, **k: {})

    captured = {}

    def fake_mast(*, config_task, shots_list, **kwargs):
        captured["shots_list"] = list(shots_list)
        captured["kwargs"] = kwargs
        return _FakeMast(shots_list)

    def fake_tokamark(dataset, task_metadata, config_metadata, **kwargs):
        return _FakeTokaMark(
            dataset.shots_list, config_metadata, override_shot=override_shot
        )

    monkeypatch.setattr(_upstream, "initialize_MAST_dataset", fake_mast)
    monkeypatch.setattr(_upstream, "initialize_TokaMark_dataset", fake_tokamark)
    return captured


class _Cluster:
    tokamark_root = "s3://mast/tokamark/v1"
    tokamark_storage_options = {
        "anon": True,
        "client_kwargs": {"endpoint_url": "https://s3.echo.stfc.ac.uk"},
    }


class _DataCfg:
    limit_shots = None


def test_split_membership_preserved_per_split(monkeypatch, tmp_path) -> None:
    captured = _patch_minimal_upstream(monkeypatch, tmp_path)

    dataset = make_dataset(
        "task_2-3",
        "train",
        cluster=_Cluster(),
        data_cfg=_DataCfg(),
    )
    samples = list(dataset)

    assert captured["shots_list"] == [1, 2, 3]
    assert {s.shot_id for s in samples} == {"1", "2", "3"}
    assert all(s.metadata["split"] == "train" for s in samples)
    # Not the default eval path: std scaling is off.
    assert captured["kwargs"]["use_std_scaling"] is False


def test_requesting_shot_outside_split_impossible(monkeypatch, tmp_path) -> None:
    _patch_minimal_upstream(monkeypatch, tmp_path, override_shot=999)

    dataset = make_dataset(
        "task_2-3",
        "train",
        cluster=_Cluster(),
        data_cfg=_DataCfg(),
    )

    with pytest.raises(ValueError, match="999"):
        list(dataset)


# ----------------------------------------------------------------------------
# 4. Ramp alignment: context < action-covered < target, float64 preserved
# ----------------------------------------------------------------------------
def test_ramp_alignment_context_action_target() -> None:
    # A t_cut carrying float64-only precision: a float32 round-trip anywhere in
    # the adapter would corrupt the low-order digits and fail the checks below.
    t_cut = 0.30000000000007
    cfg = load_task_config("task_2-3")
    window, grids = _window_from_config(cfg.config, t_cut=t_cut)

    sample = to_fusion_sample(window, cfg, "train", REGISTRY)

    for name, grid in (
        ("context_times", grids["context"]),
        ("target_times", grids["target"]),
        ("action_times", grids["action"]),
    ):
        produced = getattr(sample, name)
        assert produced.dtype == torch.float64
        # float64 preserved to well below float32 resolution (~1e-7 near 0.3).
        assert torch.allclose(produced, torch.from_numpy(grid), atol=1e-12)

    # Ordering: context strictly before target; actions cover the transition.
    assert sample.context_times[-1] < sample.target_times[0]
    assert sample.action_times[0] <= sample.context_times[-1]
    assert sample.action_times[-1] >= sample.target_times[-1]

    # And the acceptance oracle agrees end to end.
    batch = collate_fusion([sample])
    assert validate_batch(batch, split_lookup={sample.shot_id: "train"}) == []


# ----------------------------------------------------------------------------
# 5. Missing channel becomes a False mask, not a silently observed zero
# ----------------------------------------------------------------------------
def test_upstream_missing_channel_becomes_false_mask_not_zero() -> None:
    config = _minimal_config()
    _, grids = _window_from_config(config)
    ctx_times = grids["context"]
    values = np.array([[0.0, np.nan, 3.0, 4.0]], dtype=np.float64)

    window = {
        "shot_id": 100,
        "window_index": 0,
        "t_cut": float(ctx_times[-1]),
        "input": {"summary-ip": _entry(ctx_times, values)},
        "actuator": {
            "pf_active-coil_voltage": _entry(
                grids["action"], grids["action"][None, :].copy()
            )
        },
        "output": {
            "equilibrium-q95": _entry(
                grids["target"], grids["target"][None, :].copy()
            )
        },
    }

    sample = to_fusion_sample(window, config, "train", REGISTRY)
    ip = sample.context["mast.summary.ip"]
    ip_mask = sample.context_mask["mast.summary.ip"]

    # Missing entry: mask False, value finite (imputed placeholder, not NaN).
    assert ip_mask[0, 1].item() is False
    assert torch.isfinite(ip[0, 1])
    # A genuinely observed 0.0 stays observed (mask True) -- distinguishable.
    assert ip_mask[0, 0].item() is True
    assert ip[0, 0].item() == 0.0


# ----------------------------------------------------------------------------
# 6. Horizons are reported in SECONDS, matching the task YAML
# ----------------------------------------------------------------------------
def test_horizons_reported_in_seconds_match_task_yaml_ms() -> None:
    # NOTE: the task-YAML lengths are already in SECONDS (verified against the
    # pinned upstream source and docs/decisions/0002); the "_ms" in this
    # brief-mandated test name is a misnomer. The adapter must apply NO unit
    # rescaling: horizon_seconds == delta + output_length in seconds.
    cfg = load_task_config("task_2-3")
    seg = cfg.config["task_window_segmenter"]
    expected = float(seg["delta"]) + float(seg["output_length"])

    window, _ = _window_from_config(cfg.config)
    sample = to_fusion_sample(window, cfg, "train", REGISTRY)

    assert sample.horizon_seconds.dtype == torch.float64
    assert torch.isclose(
        sample.horizon_seconds,
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-9,
    )
    # A ms<->s confusion (x1000) would put the horizon at 25.0, not 0.025.
    assert sample.horizon_seconds.item() < 1.0


# ----------------------------------------------------------------------------
# 7. OfficialMetrics wraps the upstream accumulator + compute_metrics
# ----------------------------------------------------------------------------
def test_official_metrics_adapter_shapes(tmp_path) -> None:
    metrics = OfficialMetrics("task_2-3")
    y_target = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0]]
    )
    y_pred = y_target + 0.1

    metrics.add_batch(
        y_target=y_target,
        y_pred=y_pred,
        shot_ids=np.array([100, 100, 101]),
        window_indices=np.array([0, 1, 0]),
        feature_name="summary-ip",
    )
    df = metrics.compute(tmp_path, save_task_metrics=True)

    assert "summary-ip" in df.index
    assert "task_2-3" in df.index
    assert "NRMSE_mean" in df.columns
    assert (tmp_path / "task_2-3" / "task_metrics.csv").exists()


def test_missing_registry_key_raises_actionable_error() -> None:
    config = _minimal_config()
    window, _ = _window_from_config(config)

    with pytest.raises(TokamarkAdapterError, match="summary-ip"):
        to_fusion_sample(window, config, "train", {})


# ----------------------------------------------------------------------------
# Remote tests: stream a real batch from the anonymous S3 store.
# ----------------------------------------------------------------------------
def _local_cluster():
    from omegaconf import OmegaConf

    return OmegaConf.load(_repo_root() / "configs" / "cluster" / "local.yaml")


@pytest.mark.remote_data
def test_one_real_batch_streams_from_s3() -> None:
    class _Cfg:
        limit_shots = 3

    dataset = make_dataset(
        "task_2-3",
        "validation" if False else "val",
        cluster=_local_cluster(),
        data_cfg=_Cfg(),
    )
    samples = []
    for sample in dataset:
        samples.append(sample)
        if len(samples) >= 2:
            break

    assert samples, "expected at least one streamed window from S3"
    first = samples[0]
    assert first.context, "context signals must be present"
    assert first.context_times.dtype == torch.float64


@pytest.mark.remote_data
def test_real_batch_passes_dev_validator() -> None:
    class _Cfg:
        limit_shots = 4

    manifest = official_split()
    dataset = make_dataset(
        "task_2-3",
        "val",
        cluster=_local_cluster(),
        data_cfg=_Cfg(),
    )
    samples = []
    for sample in dataset:
        samples.append(sample)
        if len(samples) >= 2:
            break

    assert samples, "expected at least one streamed window from S3"
    batch = collate_fusion(samples)
    split_lookup = {s.shot_id: manifest.split_of(s.shot_id) for s in samples}
    assert validate_batch(batch, split_lookup=split_lookup) == []


@pytest.mark.remote_data
def test_official_split_csv_disjoint_and_hash_matches_manifest() -> None:
    manifest = official_split()

    # Disjointness is enforced at construction; assert again explicitly.
    manifest.assert_disjoint()
    assert set(manifest.splits) >= {"train", "val", "test"}

    csv_path = Path(_upstream.split_csv_path)
    expected = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert manifest.source_hash == expected

    out = _repo_root() / "manifests" / "splits" / "tokamark_official.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(out)

    from fusion_jepa.data.splits import SplitManifest

    reloaded = SplitManifest.load(out)
    assert reloaded.source_hash == expected
    assert reloaded.splits == manifest.splits
