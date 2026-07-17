"""Tests for the data-inspection CLI (Task 1.8).

The three offline tests exercise the pure ``summarize_batch``/``plot_batch``
functions against the deterministic synthetic ramp fixture -- no network, no
upstream ``tokamark`` data. ``test_inspect_data_end_to_end_remote`` is the M1
acceptance artifact: it drives ``main()`` end to end against the anonymous S3
store and is skipped by default (see the ``remote_data`` marker filter in
pyproject.toml).
"""

from __future__ import annotations

import json

import pytest

from fusion_jepa.cli.inspect_data import main, plot_batch, summarize_batch
from fusion_jepa.data.batch import collate_fusion, validate_batch
from tests.fixtures.synthetic import make_ramp_sample


def _two_signal_batch():
    """A valid two-sample batch carrying two multi-channel signals."""
    signals = ("plasma_current", "electron_temp")
    first = make_ramp_sample(
        signals=signals,
        channels=2,
        action_dim=2,
        shot_id="shot-1",
        window_id="window-1",
    )
    second = make_ramp_sample(
        signals=signals,
        channels=2,
        action_dim=2,
        shot_id="shot-2",
        window_id="window-2",
        time_offset=20.0,
    )
    return collate_fusion([first, second])


# ----------------------------------------------------------------------------
# 1. summarize_batch reports shapes, dtypes, units, and mask fill-rates
# ----------------------------------------------------------------------------
def test_summarize_synthetic_batch_reports_shapes_units_masks() -> None:
    batch = _two_signal_batch()
    # Mask exactly half of plasma_current's context entries: 2 samples * 2
    # channels * 2 timesteps = 8 of 16 -> 50.0% fill.
    batch.context_mask["plasma_current"][:, :, :2] = False

    summary = summarize_batch(batch)

    # Every context signal is named.
    assert "plasma_current" in summary
    assert "electron_temp" in summary
    # Shape of a context signal tensor: (B=2, C=2, T_ctx=4).
    assert "(2, 2, 4)" in summary
    # Dtype and units surface.
    assert "float32" in summary
    assert "units" in summary.lower()
    # Fill-rates: the half-masked signal is 50.0%, a fully-observed one 100.0%.
    assert "50.0%" in summary
    assert "100.0%" in summary
    # Horizon is displayed in milliseconds.
    assert "ms" in summary


# ----------------------------------------------------------------------------
# 2. plot_batch writes one PNG per modality (each signal + actions)
# ----------------------------------------------------------------------------
def test_plots_written_for_each_modality(tmp_path) -> None:
    batch = _two_signal_batch()

    paths = plot_batch(batch, tmp_path)

    # Two signals (each in context and target) + one actions plot.
    assert len(paths) == 3
    names = {path.name for path in paths}
    assert any("plasma_current" in name for name in names)
    assert any("electron_temp" in name for name in names)
    assert any("action" in name for name in names)
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


# ----------------------------------------------------------------------------
# 3. Validator problems surface in the printed summary (not fatal)
# ----------------------------------------------------------------------------
def test_validator_problems_surface_in_summary() -> None:
    batch = _two_signal_batch()

    # An empty split lookup produces genuine split-mismatch problems.
    problems = validate_batch(batch, split_lookup={}, strict=False)
    assert problems, "expected the empty split lookup to yield problems"

    with_problems = summarize_batch(batch, problems=problems)
    assert "split_lookup" in with_problems
    assert problems[0] in with_problems

    # With no problems, the mismatch text must not leak into the summary.
    clean = summarize_batch(batch)
    assert "split_lookup" not in clean


# ----------------------------------------------------------------------------
# M1 acceptance: one real remote batch, validated, summarized, plotted.
# ----------------------------------------------------------------------------
def _expected_context_signals() -> list[str]:
    """Canonical context-signal names for task_2-3, resolved offline."""
    from fusion_jepa.data.tokamark import _default_registry, load_task_config

    cfg = load_task_config("task_2-3")
    registry = _default_registry()
    canonical = {spec.source_name: name for name, spec in registry.items()}
    pairs = cfg.config["task_window_segmenter"]["input_keys"] or []
    return [canonical[f"{source}-{signal}"] for source, signal in pairs]


@pytest.mark.remote_data
def test_inspect_data_end_to_end_remote(tmp_path, capsys) -> None:
    runs_root = tmp_path / "runs"
    argv = [
        "experiment=mast_smoke",
        f"cluster.runs_root={runs_root}",
        "data.limit_shots=4",
    ]

    exit_code = main(argv)

    assert exit_code == 0
    out = capsys.readouterr().out

    completions = list(runs_root.glob("*/completion.json"))
    assert len(completions) == 1
    completion = json.loads(completions[0].read_text(encoding="utf-8"))
    assert completion["status"] == "succeeded"

    run_dir = completions[0].parent
    pngs = list((run_dir / "artifacts" / "inspect").glob("*.png"))
    assert len(pngs) >= 1

    # The printed summary must mention every context signal.
    for signal in _expected_context_signals():
        assert signal in out, f"summary is missing context signal {signal!r}"
