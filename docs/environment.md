# Environment Management

Fusion-JEPA's engineering spec calls for a plain `uv.lock` for dependency
management. This repo deliberately uses [pixi](https://pixi.sh) instead,
carried over from the FusionAIHub fork it builds on. Pixi wraps `uv` for
PyPI resolution and adds conda-forge (and, for the `fdp` environment,
`ga-fdp`) channels for packages that don't ship as pure-Python wheels —
most importantly the ROCm PyTorch stack that has been verified working on
OLCF Frontier. A bare `uv.lock` workflow has not been tested there, so
`pixi.lock` remains the source of truth for reproducible environments in
this repo; this is a conscious deviation from the spec, not an oversight.

## Environments

Defined in `pyproject.toml` under `[tool.pixi.environments]`:

- `default` — CUDA build (`cu124` PyTorch wheels), no `ga-fdp` packages.
- `fdp` — adds `toksearch` / `toksearch_d3d` from the `ga-fdp` conda
  channel for direct DIII-D MDSplus access. Linux-64 only.
- `frontier` — ROCm build (`rocm7.1` PyTorch wheels) for AMD MI250X GPUs
  on OLCF Frontier. Linux-64 only.

## Common commands

```bash
export PATH="$HOME/.pixi/bin:$PATH"   # if pixi isn't already on PATH

pixi install                          # solve (if needed) + install the default environment
pixi shell-hook -e frontier --frozen  # print the frontier env's shell hook without re-solving
pixi run test-unit                    # run the unit test suite (pytest tests/unit/)
pixi run lint                         # run ruff over the repo
```

Use `pixi run -e <env> <command>` (or `pixi shell -e <env>`) to target a
specific environment, e.g. `pixi run -e frontier python -m pytest tests/unit/ -v`.
