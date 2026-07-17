"""Unit tests for the shared query-conditioned decoder (Task 2.6).

Locks the brief-mandated behaviours of the ONE readout head reused for every
modality:

* :func:`test_query_decoder_shapes` -- cross-attention from ``Q`` query tokens
  onto the ``S`` predicted latents (per horizon) emits one scalar per query,
  ``[B, H, Q]`` float32, finite;
* :func:`test_masked_queries_do_not_affect_valid_outputs` -- queries are read
  out independently, so perturbing masked-query vectors leaves every valid
  query's output bit-identical (and masked outputs are zeroed);
* :func:`test_single_shared_decoder_no_modality_keyed_modules` -- the decoder
  carries no per-modality parameters: its parameter set is identical regardless
  of how many modalities the surrounding model tokenizes, and no decoder
  parameter name mentions a modality.

Forward is deterministic at ``dropout=0.0``; tests seed parameter init with
``torch.manual_seed``.
"""

import torch

from fusion_jepa.models.decoders import QueryConditionedDecoder


def _make_decoder(
    *,
    d_latent=8,
    d_model=16,
    n_heads=4,
    n_blocks=2,
    seed=0,
) -> QueryConditionedDecoder:
    torch.manual_seed(seed)
    return QueryConditionedDecoder(
        d_latent=d_latent,
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks,
    )


def test_query_decoder_shapes():
    B, H, S, Q, d_latent, d_model = 2, 2, 5, 7, 8, 16
    decoder = _make_decoder(d_latent=d_latent, d_model=d_model)
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(B, H, S, d_latent, generator=gen)
    queries = torch.randn(B, H, Q, d_model, generator=gen)
    query_mask = torch.ones(B, H, Q, dtype=torch.bool)

    out = decoder(z, queries, query_mask)
    assert out.shape == (B, H, Q)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    # The single-horizon layout used by the raw world model (H == 1).
    z1 = torch.randn(B, 1, S, d_latent, generator=gen)
    q1 = torch.randn(B, 1, Q, d_model, generator=gen)
    out1 = decoder(z1, q1, torch.ones(B, 1, Q, dtype=torch.bool))
    assert out1.shape == (B, 1, Q)


def test_masked_queries_do_not_affect_valid_outputs():
    B, H, S, Q, d_latent, d_model = 2, 1, 4, 6, 8, 16
    decoder = _make_decoder(d_latent=d_latent, d_model=d_model)
    gen = torch.Generator().manual_seed(2)
    z = torch.randn(B, H, S, d_latent, generator=gen)
    queries = torch.randn(B, H, Q, d_model, generator=gen)

    valid = torch.ones(Q, dtype=torch.bool)
    valid[2] = False
    valid[4] = False
    query_mask = valid.view(1, 1, Q).expand(B, H, Q).contiguous()

    out_ref = decoder(z, queries, query_mask)

    # Perturb ONLY the masked query vectors with finite garbage; every valid
    # query is read out independently, so its scalar must not move.
    mutated = queries.clone()
    mutated[:, :, ~valid, :] = 1.0e6
    out_mut = decoder(z, mutated, query_mask)

    assert torch.equal(out_ref[:, :, valid], out_mut[:, :, valid])
    # Masked outputs are zeroed (they carry no observed target to predict).
    assert torch.all(out_ref[:, :, ~valid] == 0.0)
    assert torch.all(out_mut[:, :, ~valid] == 0.0)


def test_single_shared_decoder_no_modality_keyed_modules():
    # The tiny shared builder lives in the raw-world-model test module (kept to
    # the task's four new files); importing it here exercises the SAME decoder
    # class both tests wire up.
    from tests.unit.test_raw_world_model import build_raw_world_model

    one = build_raw_world_model(modalities=("slow_ts",))
    two = build_raw_world_model(modalities=("slow_ts", "profile"))

    # The decoder is a SINGLE shared module reused for every modality.
    assert one.decoder is not None
    assert isinstance(one.decoder, QueryConditionedDecoder)

    decoder_params_one = sum(p.numel() for p in one.decoder.parameters())
    decoder_params_two = sum(p.numel() for p in two.decoder.parameters())
    # A modality-keyed readout would grow with the modality set; a shared one
    # does not.
    assert decoder_params_one == decoder_params_two

    # No decoder parameter/module is keyed by a modality name, and there is no
    # per-modality container (nn.ModuleDict) hidden inside the readout head.
    param_names = [name for name, _ in two.decoder.named_parameters()]
    for name in param_names:
        assert "slow_ts" not in name
        assert "profile" not in name
    for _, module in two.decoder.named_modules():
        assert not isinstance(module, torch.nn.ModuleDict)
