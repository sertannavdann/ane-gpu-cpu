# Plan (rev 2): Metal-accelerated inference on the M4 Max — host-native LLM tiers + Docker CPU fallback

Revised 2026-09-18. Changes from rev 1 are marked **[rev2]**. Unchanged items are kept verbatim so this file can replace the original.

## Context

`llm_service` runs `llama-cpp-python` inside Docker Desktop's Linux VM (8 GB RAM). Docker on macOS cannot expose Metal or the Neural Engine to containers, and `ModelManager._load()` never passes `n_gpu_layers`, so every token is generated on CPU while the M4 Max's 40-core GPU sits idle. The same compose file must keep working on Linux / cloud / Windows hosts.

**[rev2] Premise re-verified (Sept 2026).** Neither Docker Desktop nor `apple/container` passes Metal to a Linux guest; the M5 generation did not change this. Docker's own answer is Docker Model Runner (DMR), which runs `llama.cpp` or `vllm-metal` *natively on the host* and exposes it to containers over TCP — i.e. exactly the topology this plan implements. The host-native decision stands. Two things have moved and are folded in below: (a) MLX has become the default fast path for dense models ≤14B on Apple Silicon (Ollama 0.19 switched to it; MLX leads llama.cpp by ~20–87% under 14B and converges above ~27B where bandwidth dominates), and (b) burst tok/s is now known to be a misleading metric on Apple Silicon — sustained decode and tokens-per-joule are what to record.

This implements steps 1 (cleanup) and 4 (Metal for llama.cpp) of `docs/LLM_SERVICE_REDESIGN.md`, **[rev2] plus a thin engine interface so step 5 (MLX) is an additive backend rather than a rewrite.** Core ML / Core AI (ANE) remains out of scope (step 6) — the ANE is a prefill/small-fixed-model engine, not a decode engine, and macOS 27's Core AI is too new to build a tier on.

Facts established during exploration (unchanged from rev 1):
- `llm_service/llama/llama-cli` is not tracked (deleted in 5c34440); only a local stray file. `rm -rf` it.
- Compose v5.0.2 errors when `depends_on` names a service in an inactive profile. Long-form `depends_on: {llm_service: {condition: service_started, required: false}}` works (tested).
- `orchestrator/capability_map.py:get_lidm_endpoints()` prefers `config/routing_config.json` tier endpoints over `LLM_HEAVY_HOST`/`LLM_STANDARD_HOST`. Env vars alone will NOT redirect the pool.
- Host `llm_pb2_grpc.py` uses `from . import llm_pb2`, so the host process must run as `python -m llm_service.llm_service` from repo root.
- Host: Python 3.12.9, uv, cmake, full Xcode, grpcurl+jq, no `grpc_health_probe`, no `llama_cpp`. 10 P + 4 E cores, 36 GB. Models: 0.5B, 3B, 14B-Q4_K_M GGUFs in `llm_service/models.nosync/`.
- `ui_service/src/app/api/settings/route.ts` rewrites the whole `.env` on save; must confirm it round-trips unknown keys like `COMPOSE_FILE`.

## Key decisions

1. **Compose switching** = committed override file selected via `COMPOSE_FILE` in `.env`. Unchanged.
2. **Device config**: `LLM_DEVICE` (`auto|cpu|gpu|metal|cuda`) + `LLM_GPU_LAYERS` (int, -1=all). Unchanged.
   **[rev2]** Add `LLM_ENGINE` (`llamacpp|mlx`, default `llamacpp`). `GetActiveModelResponse` gains `string engine = 9` alongside `device = 8`, so status output reads e.g. `engine=llamacpp device=metal` or `engine=mlx device=metal`.
3. **Mac tier→model mapping**: heavy = Qwen2.5-14B-Instruct-Q4_K_M.gguf, ctx 16384, :50051; standard = qwen2.5-3b-instruct-q5_k_m.gguf, ctx 8192, :50061. ~15 GB GPU-resident, under the default wired limit.
   **[rev2]** Keep GGUF as the shipping format for rev 2 so nothing is blocked on model conversion. The MLX backend (step 4b) reads `mlx-community/*` safetensors; the 3B standard tier is the first candidate to move because that is where MLX's lead is largest and the download is small.
4. **Threads**: supervisor exports `LLM_THREADS=$(sysctl -n hw.perflevel0.physicalcpu)` unless the tier env sets it; Docker default stays 4. **[rev2]** Note in docs that with full Metal offload `n_threads` only affects sampling and prefill tokenization; do not tune it looking for decode gains.
5. **Health**: `NOT_SERVING` until the model is loaded (`start_period: 180s`). Unchanged.
6. **CUDA path** (future): Dockerfile `ARG CMAKE_ARGS`. Unchanged.
7. **[rev2] Metal runtime settings for llama.cpp** (heavy tier defaults, in `config/llm/heavy.env`):
   - `LLM_FLASH_ATTN=1` → `flash_attn=True`. Required for KV-cache quantization and reduces memory traffic at 16k ctx.
   - `LLM_KV_CACHE_TYPE=q8_0` → `type_k=type_v=GGML_TYPE_Q8_0`. Halves KV footprint on the 14B at 16k ctx with negligible quality loss; leave the 3B at f16.
   - `LLM_N_BATCH=2048`, `LLM_N_UBATCH=512`. Prefill throughput on the 40-core GPU is batch-bound; the llama-cpp-python default (512/512) under-fills it.
   - `LLM_MLOCK=1` on the host tiers so the wired weights are never paged during idle stretches.
   - Speculative decoding is deferred (see "Not in scope, revisited"). The 0.5B GGUF on disk is the obvious draft model when it is picked up.
8. **[rev2] Engine interface.** `llm_service/engines/{base,llamacpp,mlx}.py` with `load(spec, cfg)`, `stream(prompt, params) -> Iterator[Token]`, `info() -> {engine, device, ctx, model}`. `ModelManager` owns an `Engine`, not a `Llama`. This is the one structural change; it costs ~150 lines and is what makes the next runtime shift a config change. The MLX engine is *optional* at import (`try: import mlx_lm`), so the Docker CPU image never needs it.

## Implementation steps

### Step 0 — Baseline **[rev2: expanded]**

- Upgrade `scripts/test_inference_speed.py` → rename `scripts/llm_bench.py`: argparse `--addr` (default `localhost:50051`), `--runs`, `--prompt-tokens {128,1024}`, `--sustain SECONDS` (default 0).
  - Per run: TTFT, prefill tok/s (prompt_tokens / TTFT), decode tok/s (rolling 5 s window), chunk count.
  - `--sustain 300`: re-prompt loop for N seconds; report decode tok/s at t=0, t=60, t=300 and *retained %* (t=300 / t=0). GPU runtimes on Apple Silicon can lose half their burst rate inside a minute under thermal budget; the sustained number is the one that matters for an always-on tier.
  - `--power`: if run as root on macOS, spawn `powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000` for the duration, parse mW, and report **tokens per joule** (decode tokens / GPU+CPU energy) and average GPU W. This is the efficiency metric; it is also the number that will later justify or reject an ANE tier.
  - Print one summary line; `--json` dumps everything for `docs/BENCHMARKS.md`.
- Run against the running Docker container; record as the "before" row (power will be host-wide, note that).

### Step 1 — Cleanup (no behaviour change)

Unchanged from rev 1: `rm -rf llm_service/llama/`; `N_CTX=4096 → LLM_CTX_SIZE=4096`; `openai_wrapper.py:91 response.text → response.token`; Makefile `lidm-status` loop over heavy/standard/ultra printing `modelName tier device engine ctx` via jq (`.engine // "n/a"`), no hard-coded model names.

### Step 2 — Service changes

- `shared/proto/llm.proto`: `string device = 8; string engine = 9;` in `GetActiveModelResponse`; `make proto-gen-llm`.
- `llm_service/config.py`: fields `device="auto"`, `n_gpu_layers=0`, `tier=""`, **[rev2]** `engine="llamacpp"`, `flash_attn=False`, `kv_cache_type="f16"`, `n_batch=512`, `n_ubatch=512`, `mlock=False`; `from_env()` reads the matching `LLM_*` vars; pure helper `resolve_gpu_layers(device, explicit)`; `validate()` rejects unknown devices/engines/kv types; `log_config()` prints all of it.
- **[rev2]** `llm_service/engines/base.py`, `llamacpp.py`, `mlx.py` per decision 8. `llamacpp.py` passes `n_gpu_layers`, `flash_attn`, `type_k/type_v`, `n_batch`, `n_ubatch`, `use_mlock`; sets `effective_device` (`metal` on darwin with `llama_supports_gpu_offload()` and layers≠0, `cuda` elsewhere under the same condition, else `cpu`). `mlx.py` wraps `mlx_lm.load` / `mlx_lm.stream_generate`; device is always `metal`; it refuses to load a `.gguf` path with a clear error.
- `llm_service/llm_service.py`: `ModelManager` instantiates the engine from `CONFIG.engine`; health servicer per rev 1 (`NOT_SERVING` → `SERVING` after preload, exit 1 on failure); `GetActiveModel` returns `device`, `engine`, `tier=CONFIG.tier or spec.tier`; delete dead `switch_model`.
- `llm_service/Dockerfile`: `ARG CMAKE_ARGS=""` / `ENV CMAKE_ARGS=$CMAKE_ARGS` before pip install. **[rev2]** `requirements.txt` stays llama.cpp-only; `requirements-mlx.txt` (mlx, mlx-lm) is host-only.
- `docker-compose.yaml`: unchanged from rev 1 (both LLM containers `LLM_DEVICE=cpu`, `start_period: 180s`; orchestrator long-form `depends_on` with `required: false`; `${VAR:-default}` env).
- `orchestrator/routing_config.py` + `config/routing_config.json`: unchanged from rev 1 (`${LLM_HEAVY_HOST:-…}` expansion; fix `Q4_K → Q4_K_M`).
- Rebuild `llm_service` image once; confirm the CPU path still works (`device: "cpu"`, `engine: "llamacpp"`).

### Step 3 — Compose override + .env

Unchanged from rev 1 (`docker-compose.host-llm.yaml` with `profiles: [cpu-fallback]` on both LLM services and `host.docker.internal` env on orchestrator; `COMPOSE_FILE=` in `.env`; `.env.example` block; verify with `docker compose config`; check the settings-route serializer keeps `COMPOSE_FILE`).

### Step 4 — Host runtime (llama.cpp / Metal)

- `config/llm/heavy.env` and `config/llm/standard.env` (tracked): `LLM_PORT`, `LLM_MODEL_PATH`, `LLM_CTX_SIZE`, `LLM_DEVICE=gpu`, `LLM_GPU_LAYERS=-1`, `LLM_TIER`, `LLM_HOST=[::]`, `LLM_MAX_TOKENS=2048`, **[rev2]** `LLM_ENGINE=llamacpp`, `LLM_FLASH_ATTN=1`, `LLM_MLOCK=1`, and on heavy only `LLM_KV_CACHE_TYPE=q8_0`, `LLM_N_BATCH=2048`, `LLM_N_UBATCH=512`.
- `scripts/llm_supervisor.sh` (`start|stop|restart|status|logs|wait [tier…]`): unchanged from rev 1 (pidfiles in `logs/llm/`, `lsof` port check with the "held by Docker" hint, `nohup … python -m llm_service.llm_service`, health-poll `wait`, never source root `.env`). **[rev2]** `status` also prints `engine`.
- Makefile: `llm-venv` (source build with `CMAKE_ARGS="-DGGML_METAL=on"`; **[rev2]** pin `llama-cpp-python` to a version whose vendored `llama.cpp` is ≥ the `b7xxx` range — check `pip show` after install and record the vendored commit in `BENCHMARKS.md`, since this package lags upstream Metal kernel work by weeks to months), `llm-up`, `llm-down`, `llm-restart`, `llm-status`, `llm-logs`, `llm-bench` (**[rev2]** `SUSTAIN?=0 POWER?=0`), `lidm-up` delegation, `status` host-tier section, `proto-gen-llm` venv preference.
- Run: `make llm-venv`, `docker stop llm_service`, `make llm-up`, `make llm-status`.

### Step 4b — MLX engine **[rev2, was "out of scope"]**

- `make llm-venv-mlx`: `uv pip install -r llm_service/requirements-mlx.txt` into the same venv.
- `config/llm/standard-mlx.env`: `LLM_ENGINE=mlx`, `LLM_MODEL_PATH=mlx-community/Qwen2.5-3B-Instruct-4bit` (HF id; `mlx_lm` caches it under `~/.cache/huggingface`), same port/ctx/tier as `standard.env`.
- `scripts/llm_supervisor.sh start standard-mlx` is just another tier file; no supervisor change.
- Bench both 3B variants back-to-back with `--sustain 300 --power`; whichever wins on *sustained* tok/s **and** tokens/J becomes the default `standard.env`. Repeat for the 14B (`mlx-community/Qwen2.5-14B-Instruct-4bit`) only if the 3B result is decisive; above ~14B the two engines converge and the download is not free.
- Known MLX caveat to record: `mlx_lm` does full prefill before emitting the first token, so TTFT on 1024-token prompts may be worse than llama.cpp even when decode is faster. The bench's two prompt sizes exist to expose this.

### Step 5 — Tests and docs

- `tests/unit/test_llm_config.py`: device→layers resolution, explicit layers win, invalid device/engine/kv type raises, `LLM_TIER` override.
- **[rev2]** `tests/unit/test_engines.py`: engine factory picks by name; `mlx` engine import failure surfaces a clear error rather than an `ImportError` at request time; `llamacpp` engine passes the Metal kwargs (mock `Llama`).
- `tests/unit/test_routing_config.py`: `${LLM_HEAVY_HOST:-…}` expansion with and without env.
- Fix `test_model_registry.py::test_known_model_qwen14b` key if still failing.
- `tests/integration/conftest.py`: fall back to `logs/llm/heavy.log` when `docker logs llm_service` is empty.
- `docs/BENCHMARKS.md` (new). **[rev2]** Columns: config, engine, model, quant, ctx, prompt tokens, TTFT, prefill tok/s, decode tok/s (burst), decode tok/s (t=300 s), retained %, avg GPU W, tokens/J, peak RSS. Rows: Docker CPU (before), llama.cpp/Metal heavy, llama.cpp/Metal standard, MLX standard, (MLX heavy if run). Include the `ggml_metal_init` / `offloaded N/N layers` lines and the vendored llama.cpp commit.
- `docs/OPERATIONS.md`: "Host-native LLM tiers on macOS" section (venv, `.env` line, `make llm-up`, reverting with `--profile cpu-fallback`, the Linux `models` symlink note). **[rev2]** Add a short "Why not in-container GPU" paragraph: no Metal passthrough in Docker Desktop or `apple/container` as of macOS 26/27; the only in-container GPU path is Podman + libkrun/krunkit Vulkan→Metal remoting (~80% of native decode, llama.cpp Vulkan backend, not Docker Desktop) — documented as an alternative, not supported by this repo. Mention Docker Model Runner as the equivalent host-native pattern if someone wants an OpenAI-compatible endpoint without our gRPC service.
- `docs/LLM_SERVICE_REDESIGN.md`: mark steps 1, 4 done and 5 started; correct F2 (stub not tracked); add the routing_config precedence finding; **[rev2]** add the engine-interface decision and the "sustained + tokens/J" benchmark policy.

## Not in scope, revisited **[rev2]**

- **Speculative decoding** (0.5B draft → 14B target): supported in `llama-server` (`-md`) but not cleanly exposed in `llama-cpp-python`'s streaming API. Revisit when/if the host tier moves to a `llama-server` proxy (below). Expect ~1.5–2× decode on code/structured output, less on prose.
- **`llama-server` instead of `llama-cpp-python`**: `brew install llama.cpp` gives a C++ binary with the freshest Metal kernels and no 5–10 min source build; the gRPC node would become a thin proxy over its HTTP/SSE stream. Cleaner and faster to track upstream, but a larger diff than rev 2 wants. The engine interface (decision 8) makes this a third engine, `llamaserver.py`, later.
- **ANE / Core AI utility tier**: only worth it for a small fixed model doing routing/extraction with short outputs, on power/thermal grounds rather than tok/s. Gate on the tokens/J numbers from Step 0/4b.
- **`vllm-metal` / `vllm-mlx`**: relevant only if the orchestrator starts issuing concurrent requests to one tier (continuous batching). Single-stream today; not needed.

## Pitfalls to respect

- Port 50051 is bound by the running container; switching `COMPOSE_FILE` does not stop a now-profiled container, hence the explicit `docker stop` in `llm-up`.
- After editing `routing_config.json`, recreate orchestrator (or `POST /admin/routing-config/reload`).
- Keep `LLM_HOST=[::]` on the host; macOS firewall may prompt to allow python incoming connections.
- First host run with `LLM_VERBOSE=1` to capture Metal init lines; afterwards rely on the `device`/`engine` fields.
- **[rev2]** `flash_attn=True` + `q8_0` KV is required together; llama.cpp silently falls back to f16 KV without FA. Assert the effective KV type in the load log.
- **[rev2]** `powermetrics` needs root; `make llm-bench POWER=1` should `sudo -v` up front rather than fail mid-run.
- **[rev2]** Do not compare burst numbers across sessions or after a macOS update; only `--sustain` rows are comparable, and only within the same OS build.

## Verification

1. `pytest tests/unit` green (incl. `test_engines.py`).
2. `docker compose config --services` omits `llm_service`; config shows `host.docker.internal:50051`; no `depends_on` error.
3. `make llm-up` returns only after both tiers are `SERVING`; `logs/llm/heavy.log` shows `ggml_metal_init: found device: Apple M4 Max`, `offloaded N/N layers to GPU`, and the effective KV cache type; `GetActiveModel` on :50051 shows `device: "metal", engine: "llamacpp", tier: "heavy"`.
4. `make status` shows containers healthy, no `llm_service` row, host tiers green; `make lidm-status` prints real model names and engines.
5. `make llm-bench SUSTAIN=300 POWER=1` on :50051 and :50061 vs the Step-0 baseline; all columns into `docs/BENCHMARKS.md`. Acceptance: heavy tier retains ≥70% of burst decode at t=300 s (if it doesn't, the box is thermally limited and the 14B/16k config is too aggressive — drop to 12k ctx or a smaller quant before touching anything else).
6. Step 4b: MLX standard tier reaches `SERVING`, `engine: "mlx"`, and its bench row exists; default `standard.env` set to the winner with the reason recorded.
7. End-to-end: `make query Q="..."` answers via the orchestrator; `docker logs orchestrator` shows `LIDM endpoints … host.docker.internal:50051`.
8. Regression: `make llm-down && docker compose --profile cpu-fallback up -d llm_service` brings the CPU container back healthy with `device: "cpu"`; with `COMPOSE_FILE` removed from a scratch `.env`, `docker compose config --services` includes `llm_service` again.
