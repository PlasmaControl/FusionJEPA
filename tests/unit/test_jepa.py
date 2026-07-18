"""Unit tests for the JEPA model with explicit target-update policies (Task 3.2).

The JEPA reuses the IDENTICAL building blocks as the raw baseline
(:class:`~fusion_jepa.models.raw_world_model.RawWorldModel`) -- tokenizers ->
merge -> :class:`ContextEncoder` -> (:class:`ActionEncoder`,
:class:`LatentPredictor`) -- and only swaps the raw reconstruction readout for a
latent-prediction objective against a *target encoder*. The three target-update
policies differ solely in how that target trunk relates to the online trunk and
whether gradient flows back through it:

* ``EMA`` -- the target tokenizers+encoder are a frozen ``deepcopy`` of the
  online ones (their parameters never require grad; the target forward runs under
  ``torch.no_grad``). A separate :class:`~fusion_jepa.training.ema.EmaUpdater`
  nudges them toward the online twins.
* ``SHARED_STOPGRAD`` -- the target trunk *is* the online trunk (same objects,
  shared weights); the target latent is ``detach()``ed so no gradient flows.
* ``END_TO_END_REGULARIZED`` -- also the online trunk, but the target latent is
  *not* detached (gradient flows through both branches); it requires a collapse
  regularizer at construction to be well-posed.

``build_jepa_model`` is the tiny shared constructor these tests use; it wires the
same (tiny) widths as ``test_raw_world_model.build_raw_world_model`` so the two
models are structurally matched.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from fusion_jepa.models.action_encoder import ActionEncoder
from fusion_jepa.models.encoder import ContextEncoder
from fusion_jepa.models.jepa import JEPAModel, JEPAOutput, TargetUpdatePolicy
from fusion_jepa.models.predictor import LatentPredictor
from fusion_jepa.models.tokenizers import ScalarSeriesTokenizer
from fusion_jepa.training.distributed import DistributedManager
from fusion_jepa.training.ema import EmaUpdater
from tests.fixtures.synthetic import make_synthetic_fusion_batch

# Tiny, mutually consistent widths. ``_D`` is shared by the tokenizers, the
# context encoder, the predicted-latent dim, AND the target-encoder output dim,
# so ``z_hat`` and ``z_target`` line up for the latent-prediction loss.
_D = 16
_S = 4
_N_HEADS = 4
_D_ACTION = 12
_PATCH_LEN = 2
_N_TIME_FREQS = 4
_DEVICE_VOCAB = ("MAST",)
_D_DEVICE_CONTEXT = 3
_MODALITIES = ("slow_ts", "profile")


def build_jepa_model(
    modalities: Sequence[str] = _MODALITIES,
    *,
    policy: TargetUpdatePolicy | str = TargetUpdatePolicy.EMA,
    n_channels: int = 3,
    n_actuators: int = 2,
    ema_decay: float = 0.996,
    collapse_regularizer: object | None = None,
    seed: int = 0,
) -> JEPAModel:
    """Compose a tiny JEPA over ``modalities`` from the shared components."""
    torch.manual_seed(seed)
    tokenizers = {
        modality: ScalarSeriesTokenizer(
            n_channels=n_channels,
            d_model=_D,
            patch_len=_PATCH_LEN,
            n_time_freqs=_N_TIME_FREQS,
            modality=modality,
        )
        for modality in modalities
    }
    encoder = ContextEncoder(
        d_model=_D,
        n_heads=_N_HEADS,
        n_blocks=2,
        n_state_tokens=_S,
    )
    action_encoder = ActionEncoder(
        n_actuators=n_actuators,
        d_model=_D_ACTION,
        n_time_freqs=_N_TIME_FREQS,
    )
    predictor = LatentPredictor(
        d_model=_D,
        n_heads=_N_HEADS,
        n_blocks=2,
        d_latent_in=_D,
        n_state_tokens=_S,
        device_vocab=list(_DEVICE_VOCAB),
        d_device_context=_D_DEVICE_CONTEXT,
        d_action=_D_ACTION,
    )
    return JEPAModel(
        tokenizers=tokenizers,
        encoder=encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        policy=policy,
        ema_decay=ema_decay,
        collapse_regularizer=collapse_regularizer,
    )


def _has_grad(param: torch.Tensor) -> bool:
    """True iff ``param`` accumulated a non-zero gradient."""
    return param.grad is not None and bool(torch.any(param.grad != 0))


def test_target_encoder_receives_no_gradient_in_ema_mode():
    batch = make_synthetic_fusion_batch(
        B=2,
        modalities=_MODALITIES,
        n_channels=3,
        T=4,
        H=3,
        A=2,
        missing_fraction=0.3,
    )
    model = build_jepa_model(policy=TargetUpdatePolicy.EMA)

    out = model(batch)
    # ``z_target`` never carries gradient in EMA mode.
    assert out.z_target.requires_grad is False

    loss = ((out.z_hat - out.z_target) ** 2).mean()
    loss.backward()

    # Every EMA target parameter is frozen and never touched by backward.
    for _online, target_module in model.target_encoder_pairs():
        for param in target_module.parameters():
            assert param.requires_grad is False
            assert param.grad is None

    # Every online trunk parameter that participates in ``z_hat`` gets a grad.
    for modality in _MODALITIES:
        assert _has_grad(model.tokenizers[modality].proj.weight)
    assert any(_has_grad(p) for p in model.encoder.parameters())
    assert _has_grad(model.action_encoder.proj.weight)
    assert _has_grad(model.predictor.out_proj.weight)


def test_shared_stopgrad_shares_parameters_and_detaches():
    batch = make_synthetic_fusion_batch(
        B=2,
        modalities=_MODALITIES,
        n_channels=3,
        T=4,
        H=3,
        A=2,
    )
    model = build_jepa_model(policy=TargetUpdatePolicy.SHARED_STOPGRAD)

    # The target trunk IS the online trunk (same objects, shared weights).
    assert model.target_encoder is model.encoder
    assert model.target_tokenizers is model.tokenizers
    assert all(
        online is target
        for online, target in zip(
            model.encoder.parameters(),
            model.target_encoder.parameters(),
            strict=True,
        )
    )

    out = model(batch)
    # The shared target latent is stop-gradient (detached).
    assert out.z_target.requires_grad is False

    loss = ((out.z_hat - out.z_target) ** 2).mean()
    loss.backward()
    # Backward through ``z_hat`` still populates the shared online parameters.
    assert any(_has_grad(p) for p in model.encoder.parameters())
    assert _has_grad(model.predictor.out_proj.weight)


def test_end_to_end_without_regularizer_raises():
    with pytest.raises(ValueError, match="collapse_regularizer"):
        build_jepa_model(
            policy=TargetUpdatePolicy.END_TO_END_REGULARIZED,
            collapse_regularizer=None,
        )


def test_end_to_end_with_regularizer_constructs_and_stores():
    sentinel = object()
    model = build_jepa_model(
        policy=TargetUpdatePolicy.END_TO_END_REGULARIZED,
        collapse_regularizer=sentinel,
    )
    # Stored verbatim, never called, and (crucially) never registered as a
    # submodule -- so it injects no parameters into the matched-capacity trunk.
    assert model.collapse_regularizer is sentinel
    assert model.target_encoder is model.encoder
    # ``END_TO_END`` lets gradient flow through the target branch (no detach).
    batch = make_synthetic_fusion_batch(
        B=2,
        modalities=_MODALITIES,
        n_channels=3,
        T=4,
        H=3,
        A=2,
    )
    out = model(batch)
    assert out.z_target.requires_grad is True


def test_policy_accepts_lowercase_string_name():
    model = build_jepa_model(policy="ema")
    assert model.policy is TargetUpdatePolicy.EMA
    shared = build_jepa_model(policy="shared_stopgrad")
    assert shared.policy is TargetUpdatePolicy.SHARED_STOPGRAD


def test_unknown_policy_string_raises():
    with pytest.raises(ValueError, match="target-update policy"):
        build_jepa_model(policy="momentum")


def test_forward_shapes_on_synthetic_batch():
    B = 2
    batch = make_synthetic_fusion_batch(
        B=B,
        modalities=_MODALITIES,
        n_channels=3,
        T=4,
        H=3,
        A=2,
    )
    model = build_jepa_model(policy=TargetUpdatePolicy.EMA)

    out = model(batch)

    assert isinstance(out, JEPAOutput)
    assert out.z_hat.shape == (B, 1, _S, _D)
    assert out.z_target.shape == (B, 1, _S, _D)
    assert out.target_valid.shape == (B, 1)
    assert out.target_valid.dtype == torch.bool
    assert torch.isfinite(out.z_hat).all()
    assert torch.isfinite(out.z_target).all()
    # A fully observed synthetic batch flags every target window valid.
    assert out.target_valid.all()


def test_fully_masked_target_window_finite_and_flagged_invalid():
    batch = make_synthetic_fusion_batch(
        B=2,
        modalities=_MODALITIES,
        n_channels=3,
        T=4,
        H=3,
        A=2,
    )
    # Unobserve example 0's ENTIRE target window across every modality, and put
    # a NaN placeholder there (the M1 finite-where-observed reality). The
    # tokenizer's learned missing-fill + the encoder's always-valid state tokens
    # make the target latent structurally finite even so.
    for modality in _MODALITIES:
        batch.target_mask[modality][0] = False
        batch.target[modality][0] = float("nan")
    model = build_jepa_model(policy=TargetUpdatePolicy.EMA)

    out = model(batch)

    assert torch.isfinite(out.z_target).all()
    # target_valid True iff >=1 target sample observed across all modalities.
    assert not bool(out.target_valid[0, 0])
    assert bool(out.target_valid[1, 0])


def test_target_encoder_pairs_compose_with_ema_updater():
    """Composes 3.1 (EmaUpdater) + 3.2 (target_encoder_pairs)."""
    model = build_jepa_model(policy=TargetUpdatePolicy.EMA)
    updater = EmaUpdater(model.target_encoder_pairs(), decay=0.9)

    online_enc = model.encoder.state_tokens
    target_enc = model.target_encoder.state_tokens
    online_tok = model.tokenizers[_MODALITIES[0]].proj.weight
    target_tok = model.target_tokenizers[_MODALITIES[0]].proj.weight

    # deepcopy starts each target equal to its online twin.
    assert torch.equal(online_enc.detach(), target_enc.detach())
    assert torch.equal(online_tok.detach(), target_tok.detach())

    # Perturb the online params so the lerp toward them is observable.
    torch.manual_seed(123)
    with torch.no_grad():
        online_enc.add_(torch.randn_like(online_enc))
        online_tok.add_(torch.randn_like(online_tok))

    enc_before = target_enc.detach().clone()
    tok_before = target_tok.detach().clone()
    enc_online = online_enc.detach().clone()
    tok_online = online_tok.detach().clone()

    updater.step()

    # target <- decay*target + (1 - decay)*online, for BOTH paired trunks.
    assert torch.allclose(target_enc.detach(), 0.9 * enc_before + 0.1 * enc_online)
    assert torch.allclose(target_tok.detach(), 0.9 * tok_before + 0.1 * tok_online)


def test_target_encoder_pairs_requires_ema():
    model = build_jepa_model(policy=TargetUpdatePolicy.SHARED_STOPGRAD)
    with pytest.raises(ValueError, match="EMA"):
        model.target_encoder_pairs()


def _ddp_jepa_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_path: str,
) -> None:
    """Rank body: DDP-wrap an EMA JEPA (frozen target params) and train a step.

    Must live at module scope so ``mp.spawn`` can pickle it. Uses gloo + a
    ``file://`` rendezvous under ``tmp_path`` (never a fixed TCP port) so it
    runs on CPU with no GPU. Locks plan risk R10: DDP with
    ``find_unused_parameters=False`` must tolerate the frozen target parameters
    (they are excluded from the reducer) and keep them consistent across ranks.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"

    dm = DistributedManager(backend="gloo", init_method=f"file://{init_file}")
    try:
        model = build_jepa_model(policy=TargetUpdatePolicy.EMA, seed=0)
        wrapped = dm.wrap(model)  # default find_unused_parameters=False
        assert isinstance(wrapped, DistributedDataParallel)

        batch = make_synthetic_fusion_batch(
            B=2,
            modalities=_MODALITIES,
            n_channels=3,
            T=4,
            H=3,
            A=2,
            missing_fraction=0.3,
        )
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        optimizer.zero_grad()
        out = wrapped(batch)
        loss = ((out.z_hat - out.z_target) ** 2).mean()
        loss.backward()
        optimizer.step()

        # Gather the frozen target parameters from every rank; they must remain
        # bitwise-identical across ranks after the step (never updated, and DDP
        # broadcast them at construction).
        unwrapped = dm.unwrap(wrapped)
        target_flat = torch.cat(
            [
                param.detach().reshape(-1).to(torch.float64)
                for _online, target_module in unwrapped.target_encoder_pairs()
                for param in target_module.parameters()
            ]
        )
        gathered = dm.gather_concat(target_flat)
        if dm.is_main:
            length = target_flat.numel()
            assert torch.equal(gathered[:length], gathered[length:])
            Path(result_path).write_text("ok", encoding="utf-8")
    finally:
        dm.close()


def test_ddp_wrap_with_frozen_target_params_does_not_error(tmp_path):
    init_file = tmp_path / "jepa_rendezvous"
    result_path = tmp_path / "jepa_ddp_ok"

    mp.spawn(
        _ddp_jepa_worker,
        args=(2, str(init_file), str(result_path)),
        nprocs=2,
        join=True,
    )

    assert result_path.read_text(encoding="utf-8") == "ok"
