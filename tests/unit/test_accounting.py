"""Unit tests for parameter / compute accounting (Task 2.8).

These lock the accounting utilities that Milestone 3 later builds its
``verify_matched_capacity`` on. Four behaviours matter:

* :func:`count_parameters` equals a hand-computed ``numel`` sum, honours
  ``trainable_only``, and dedupes tensors shared between sub-modules;
* :func:`parameter_report` breaks a model into per-top-level-component counts
  that sum to a JSON-serializable total;
* :func:`assert_matched_backbones` passes for two structurally identical trunks
  (regardless of initialisation), raises an actionable, component-named error on
  a width change, ignores the decoder by default (the raw-vs-JEPA *matched
  trunk* excludes the decoder + EMA copy), and handles the dict-valued
  ``tokenizers`` component;
* :func:`token_throughput_summary` reports tokens/s and tokens/s/rank.

The tiny models come from the shared ``build_raw_world_model`` constructor in
``test_raw_world_model`` (its widths are fixed constants, so the width-change
failure is produced by swapping in a differently sized component directly --
the accounting check never runs a forward pass, so the resulting width
inconsistency is intentional and harmless).
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fusion_jepa.models.decoders import QueryConditionedDecoder
from fusion_jepa.models.encoder import ContextEncoder
from fusion_jepa.models.tokenizers import ScalarSeriesTokenizer
from fusion_jepa.utils.accounting import (
    assert_matched_backbones,
    count_parameters,
    parameter_report,
    token_throughput_summary,
)
from tests.unit.test_raw_world_model import build_raw_world_model


def _module_with_n_params(n: int) -> nn.Module:
    """A minimal module owning a single parameter of ``n`` elements."""
    module = nn.Module()
    module.p = nn.Parameter(torch.zeros(n))
    return module


def test_param_count_matches_manual_numel():
    torch.manual_seed(0)
    module = nn.Module()
    module.lin = nn.Linear(3, 4)  # 4*3 weight + 4 bias = 16
    module.extra = nn.Parameter(torch.zeros(2, 5))  # 10

    # Hand-computed reference, independent of ``.parameters()``.
    assert count_parameters(module) == 16 + 10
    # And equal to the manual numel sum over the parameter iterator.
    manual = sum(p.numel() for p in module.parameters())
    assert count_parameters(module) == manual


def test_count_parameters_trainable_only_excludes_frozen():
    module = nn.Module()
    module.lin = nn.Linear(3, 4)  # 16, trainable
    frozen = nn.Parameter(torch.zeros(2, 5))  # 10, frozen
    frozen.requires_grad_(False)
    module.frozen = frozen

    assert count_parameters(module) == 26
    assert count_parameters(module, trainable_only=True) == 16


def test_count_parameters_dedupes_shared_params():
    lin_a = nn.Linear(4, 4)
    lin_b = nn.Linear(4, 4)
    lin_b.weight = lin_a.weight  # share the 4x4 weight tensor
    module = nn.Module()
    module.a = lin_a
    module.b = lin_b

    # Shared weight (16) counted once + a.bias (4) + b.bias (4) = 24.
    assert count_parameters(module) == 16 + 4 + 4


def test_count_parameters_handles_dict_of_modules():
    registry = {
        "slow_ts": _module_with_n_params(7),
        "profile": _module_with_n_params(11),
    }
    assert count_parameters(registry) == 18


def test_parameter_report_sums_to_total_and_is_json_serializable():
    model = build_raw_world_model()
    report = parameter_report(model)

    assert set(report) == {
        "tokenizers",
        "encoder",
        "action_encoder",
        "predictor",
        "decoder",
        "total",
    }
    components = {k: v for k, v in report.items() if k != "total"}
    assert sum(components.values()) == report["total"]
    assert report["total"] == count_parameters(model)
    assert all(isinstance(v, int) for v in report.values())
    json.dumps(report)  # must not raise


def test_parameter_report_surfaces_cross_component_sharing():
    """A parameter tensor shared BETWEEN top-level components (the M3
    ``shared_stopgrad`` JEPA reality: online and target encoder share
    weights) appears in each owner's standalone count, and the overlap is
    surfaced under ``_shared_across_components`` so the breakdown always
    reconciles with the deduped total (Codex 2.8 review finding: the
    breakdown silently double-counted shared capacity)."""
    shared = nn.Linear(4, 4)

    class Owner(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

    class TwoOwners(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.online = Owner(shared)
            self.target = Owner(shared)
            self.head = nn.Linear(2, 2)

    report = parameter_report(TwoOwners())
    shared_numel = sum(p.numel() for p in shared.parameters())

    # Each owner reports its standalone capacity (the shared tensor fully
    # counted in both) and the overlap is explicit, never attributed to
    # whichever component iterates first.
    assert report["online"] == shared_numel
    assert report["target"] == shared_numel
    assert report["_shared_across_components"] == shared_numel
    assert (
        report["online"]
        + report["target"]
        + report["head"]
        - report["_shared_across_components"]
        == report["total"]
    )
    json.dumps(report)  # still JSON-serializable


def test_matched_assertion_passes_for_shared_config():
    # Different seeds -> different initialisation, identical structure.
    model_a = build_raw_world_model(seed=0)
    model_b = build_raw_world_model(seed=1)

    # A matched trunk must pass and return the shared per-component counts.
    result = assert_matched_backbones(model_a, model_b)
    assert result["encoder"] == count_parameters(model_a.encoder)
    assert result["tokenizers"] == count_parameters(model_a.tokenizers)


def test_matched_assertion_fails_on_width_change():
    model_a = build_raw_world_model()
    model_b = build_raw_world_model()
    # Widen ONLY model_b's encoder so the first (and only) mismatch is the
    # 'encoder' component and the raised message is deterministic.
    model_b.encoder = ContextEncoder(
        d_model=32, n_heads=4, n_blocks=2, n_state_tokens=4
    )

    with pytest.raises(ValueError, match="encoder"):
        assert_matched_backbones(model_a, model_b)


def test_matched_assertion_message_names_component_and_both_counts():
    model_a = build_raw_world_model()
    model_b = build_raw_world_model()
    model_b.encoder = ContextEncoder(
        d_model=32, n_heads=4, n_blocks=2, n_state_tokens=4
    )
    count_a = count_parameters(model_a.encoder)
    count_b = count_parameters(model_b.encoder)

    with pytest.raises(ValueError) as excinfo:
        assert_matched_backbones(model_a, model_b)
    message = str(excinfo.value)
    assert "encoder" in message
    assert str(count_a) in message
    assert str(count_b) in message


def test_matched_assertion_ignores_decoder_by_default():
    model_a = build_raw_world_model()
    model_b = build_raw_world_model()
    # A different decoder must NOT trip the default matched-trunk check: M3's
    # capacity check reports decoder + EMA copy separately from the trunk.
    model_b.decoder = QueryConditionedDecoder(
        d_latent=16, d_model=16, n_heads=4, n_blocks=4
    )

    assert_matched_backbones(model_a, model_b)  # no raise

    # Explicitly requesting 'decoder' surfaces the mismatch.
    with pytest.raises(ValueError, match="decoder"):
        assert_matched_backbones(model_a, model_b, components=("decoder",))


def test_matched_assertion_handles_dict_valued_tokenizers():
    def _tokenizers(n_channels: int) -> dict[str, nn.Module]:
        return {
            "slow_ts": ScalarSeriesTokenizer(
                n_channels=n_channels,
                d_model=16,
                patch_len=2,
                n_time_freqs=4,
                modality="slow_ts",
            )
        }

    model_a = SimpleNamespace(tokenizers=_tokenizers(3))
    model_b = SimpleNamespace(tokenizers=_tokenizers(3))
    # Plain-dict tokenizers with identical structure match.
    assert_matched_backbones(model_a, model_b, components=("tokenizers",))

    # A channel-count change alters the tokenizer parameter count -> mismatch.
    model_c = SimpleNamespace(tokenizers=_tokenizers(5))
    with pytest.raises(ValueError, match="tokenizers"):
        assert_matched_backbones(model_a, model_c, components=("tokenizers",))


def test_matched_assertion_missing_component_is_actionable():
    model_a = build_raw_world_model()
    model_b = SimpleNamespace()  # exposes no components at all
    with pytest.raises(ValueError, match="encoder"):
        assert_matched_backbones(model_a, model_b, components=("encoder",))


def test_matched_assertion_rel_tol_allows_small_difference():
    model_a = SimpleNamespace(encoder=_module_with_n_params(1000))
    model_b = SimpleNamespace(encoder=_module_with_n_params(1005))  # +0.5%

    # Exact match (rel_tol=0.0) rejects even a tiny difference.
    with pytest.raises(ValueError):
        assert_matched_backbones(model_a, model_b, components=("encoder",))
    # A 1% tolerance accepts it.
    assert_matched_backbones(
        model_a, model_b, components=("encoder",), rel_tol=0.01
    )


def test_token_throughput_summary_scales_by_world_size():
    summary = token_throughput_summary(
        n_tokens=1000, wall_seconds=2.0, world_size=4
    )
    assert summary["tokens_per_s"] == pytest.approx(500.0)
    assert summary["tokens_per_s_per_rank"] == pytest.approx(125.0)
    json.dumps(summary)  # JSON-serializable

    # Default world_size == 1: per-rank equals the aggregate rate.
    single = token_throughput_summary(n_tokens=1000, wall_seconds=2.0)
    assert single["tokens_per_s_per_rank"] == pytest.approx(single["tokens_per_s"])


def test_token_throughput_summary_rejects_nonpositive_inputs():
    with pytest.raises(ValueError):
        token_throughput_summary(n_tokens=1, wall_seconds=0.0)
    with pytest.raises(ValueError):
        token_throughput_summary(n_tokens=1, wall_seconds=1.0, world_size=0)
