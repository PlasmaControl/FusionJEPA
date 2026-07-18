"""Parameter and compute accounting utilities (Task 2.8).

These helpers make the *matched-capacity* contract between the raw baseline and
the JEPA (Milestone 3) checkable rather than aspirational, and give training
runs a small, JSON-serializable throughput summary.

The load-bearing helper is :func:`assert_matched_backbones`: the raw world model
and the JEPA share the IDENTICAL tokenizer / encoder / action-encoder / predictor
classes and configuration, so their *trunk* parameter counts must match exactly.
The decoder (and, in M3, the EMA target copy) are deliberately excluded from that
default trunk check -- they differ between the two objectives and are reported
separately. This utility is the basis M3's ``verify_matched_capacity`` builds on.

Component lookup is by attribute name (models expose ``.tokenizers``,
``.encoder``, ``.action_encoder``, ``.predictor``, ``.decoder``); the
``tokenizers`` component may be an :class:`~torch.nn.ModuleDict` or a plain
``{modality: module}`` mapping, and both are handled.
"""

from collections.abc import Iterable, Mapping

from torch import nn

__all__ = [
    "count_parameters",
    "parameter_report",
    "assert_matched_backbones",
    "token_throughput_summary",
]

# The trunk shared verbatim between the raw baseline and the JEPA. The decoder
# and any EMA target copy are intentionally NOT here: they are objective-specific
# and reported separately.
_DEFAULT_MATCHED_COMPONENTS = (
    "tokenizers",
    "encoder",
    "action_encoder",
    "predictor",
)


def _iter_modules(obj: object) -> list[nn.Module]:
    """Normalise ``obj`` to a flat list of :class:`~torch.nn.Module`.

    Accepts a single module, a mapping ``{name: module}`` (e.g. a tokenizer
    registry), or any iterable of modules.
    """
    if isinstance(obj, nn.Module):
        return [obj]
    if isinstance(obj, Mapping):
        candidates: Iterable[object] = obj.values()
    elif isinstance(obj, Iterable):
        candidates = obj
    else:
        raise TypeError(
            "expected an nn.Module, a mapping of modules, or an iterable of "
            f"modules, got {type(obj).__name__}"
        )
    modules = list(candidates)
    for module in modules:
        if not isinstance(module, nn.Module):
            raise TypeError(
                "every element must be an nn.Module, got "
                f"{type(module).__name__}"
            )
    return modules


def count_parameters(module: object, trainable_only: bool = False) -> int:
    """Total number of parameters, deduping tensors shared between sub-modules.

    Args:
        module: an :class:`~torch.nn.Module`, a mapping ``{name: module}``, or
            any iterable of modules. Parameters shared (by identity) across the
            collection are counted once.
        trainable_only: if ``True``, count only parameters with
            ``requires_grad``.

    Returns:
        The parameter count as a plain ``int``.
    """
    seen: set[int] = set()
    total = 0
    for sub in _iter_modules(module):
        for param in sub.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            if trainable_only and not param.requires_grad:
                continue
            total += param.numel()
    return total


def parameter_report(model: nn.Module, *, trainable_only: bool = False) -> dict:
    """Per-top-level-component parameter counts plus a deduped total.

    Every immediate child module becomes one entry (the ``tokenizers``
    ``ModuleDict`` aggregates its per-modality tokenizers). Parameters owned
    directly by ``model`` (not by any child) are reported under ``"_root"`` when
    present. ``"total"`` is the whole-model deduped count.

    Component entries are each component's *standalone* capacity (deduped
    within the component). A parameter tensor shared BETWEEN components (an
    M3 reality: ``shared_stopgrad`` JEPA shares the whole online encoder
    with the target encoder) therefore appears in every component that owns
    it; the overlap is surfaced explicitly under
    ``"_shared_across_components"`` (the sum of the extra appearances), so
    the identity ``sum(components) + _root - _shared_across_components ==
    total`` always holds -- shared capacity is reported, never silently
    attributed to whichever component happens to be iterated first.

    The returned dict holds only plain ``int`` values and is JSON-serializable.
    """
    report: dict[str, int] = {}
    appearances: dict[int, tuple[int, int]] = {}  # id -> (numel, n_components)
    for name, child in model.named_children():
        report[name] = count_parameters(child, trainable_only=trainable_only)
        component_seen: set[int] = set()
        for param in child.parameters():
            if id(param) in component_seen:
                continue
            component_seen.add(id(param))
            if trainable_only and not param.requires_grad:
                continue
            numel, count = appearances.get(id(param), (param.numel(), 0))
            appearances[id(param)] = (numel, count + 1)

    shared_extra = sum(
        numel * (count - 1) for numel, count in appearances.values() if count > 1
    )
    if shared_extra:
        report["_shared_across_components"] = shared_extra

    root = sum(
        param.numel()
        for param in model.parameters(recurse=False)
        if not (trainable_only and not param.requires_grad)
    )
    if root:
        report["_root"] = root

    report["total"] = count_parameters(model, trainable_only=trainable_only)
    return report


def _component_count(model: object, name: str, trainable_only: bool) -> int:
    """Parameter count of ``model``'s ``name`` component, or an actionable error."""
    component = getattr(model, name, None)
    if component is None:
        raise ValueError(
            f"cannot compare backbones: component {name!r} is missing on "
            f"{type(model).__name__}"
        )
    return count_parameters(component, trainable_only=trainable_only)


def _within_tolerance(count_a: int, count_b: int, rel_tol: float) -> bool:
    if count_a == count_b:
        return True
    if rel_tol <= 0.0:
        return False
    denom = max(abs(count_a), abs(count_b))
    if denom == 0:
        return True
    return abs(count_a - count_b) <= rel_tol * denom


def assert_matched_backbones(
    model_a: object,
    model_b: object,
    components: Iterable[str] = _DEFAULT_MATCHED_COMPONENTS,
    rel_tol: float = 0.0,
    *,
    trainable_only: bool = False,
) -> dict:
    """Assert two models share a parameter-matched backbone.

    For each name in ``components`` (default: the raw/JEPA shared trunk --
    tokenizers, encoder, action-encoder, predictor; decoder and any EMA copy
    excluded), the two models' component parameter counts must agree to within
    ``rel_tol`` (a relative tolerance; ``0.0`` requires exact equality). The
    ``tokenizers`` component may be a ``ModuleDict`` or a plain mapping.

    Raises:
        ValueError: naming the FIRST mismatching component and both counts, or
            reporting a component missing on either model.

    Returns:
        ``{component: shared_count}`` for all compared components (useful for
        logging the matched trunk size).
    """
    matched: dict[str, int] = {}
    for name in components:
        count_a = _component_count(model_a, name, trainable_only)
        count_b = _component_count(model_b, name, trainable_only)
        if not _within_tolerance(count_a, count_b, rel_tol):
            raise ValueError(
                f"backbone mismatch in component {name!r}: model_a has "
                f"{count_a} parameters but model_b has {count_b} "
                f"(rel_tol={rel_tol}). The raw baseline and JEPA must share an "
                "identical trunk; check that both were built from the same "
                "component configuration."
            )
        matched[name] = count_a
    return matched


def token_throughput_summary(
    n_tokens: int, wall_seconds: float, world_size: int = 1
) -> dict:
    """Summarise token throughput for a run.

    Args:
        n_tokens: total tokens processed across all ranks.
        wall_seconds: wall-clock duration in seconds (must be > 0).
        world_size: number of ranks the work was spread across (must be >= 1).

    Returns:
        A JSON-serializable dict with the inputs plus ``tokens_per_s`` (aggregate
        rate) and ``tokens_per_s_per_rank`` (aggregate rate divided by
        ``world_size``).
    """
    if wall_seconds <= 0.0:
        raise ValueError(f"wall_seconds must be > 0, got {wall_seconds}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    tokens_per_s = n_tokens / wall_seconds
    return {
        "n_tokens": int(n_tokens),
        "wall_seconds": float(wall_seconds),
        "world_size": int(world_size),
        "tokens_per_s": tokens_per_s,
        "tokens_per_s_per_rank": tokens_per_s / world_size,
    }
