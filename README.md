# ANE / GPU / CPU: Apple Neural Engine kernel and placement research

Research prototype for running transformer work on the Apple Neural Engine (ANE) of an M4 Max alongside MLX on the GPU, measured against CPU and GPU baselines.

## Layout

- `docs/` — plan, findings and development guideline. Start with [the prototype plan](docs/agentic-offload-prototype-plan.md).
- `prototype/` — scheduler prototype, native ANE probes (`prototype/ane/`), R3 model evaluation (`prototype/ane/r3/`) and R4 concurrent placement (`prototype/r4/`). Setup notes in [prototype/README.md](prototype/README.md).
- `experiments/ane-latent/` — the isolated L-series experiment on activation-aware FFN down-projection factorization. See its [README](experiments/ane-latent/README.md) and [current state](experiments/ane-latent/CURRENT-STATE.md).
- `tests/` — unit tests; hardware tests skip when the native build is absent.

## Not included

Run outputs (`results/`), model caches, downloaded weights and fixtures, the virtual environment, native builds and the vendored `ane.cpp` checkout are excluded. Rebuild the vendor tree with `prototype/ane/vendor.sh`. The Agent.xpu paper discussed in the docs is at https://arxiv.org/abs/2506.24045v2.

## Status

Research code, not a product. Measurements are local to one M4 Max running macOS 26.5 and depend on private ANE runtime interfaces that may change.

## License

Released under the [MIT License](LICENSE). You may use, modify and redistribute this work, including commercially, provided the copyright notice and licence text are kept, crediting Sertan Avdan. If you use this work in research, please cite this repository.

Third-party material is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
