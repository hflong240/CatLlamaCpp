# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Review [AGENTS.md](AGENTS.md) before beginning any work. It defines the contribution rules
(human-authored code, no AI-written commits/PRs, no `git push`/`gh pr create` on the user's behalf, ASCII
only - no `—`/`→`/`×`/`…`, concise comments). Those rules override anything below when they conflict.

## What this fork is

This is the **CatEngine** fork of `llama.cpp`. All upstream functionality is unchanged; the fork adds two
things on top:

1. **On-demand MoE expert streaming from disk** - run Mixture-of-Experts models whose routed-expert weights
   are far larger than VRAM/RAM by streaming only the selected experts on the fly (three-tier cache:
   SSD/mmap -> RAM page cache -> VRAM). Opt-in, off by default.
2. **Hunyuan-v3 (`hy_v3`) architecture** - decoder graph in [src/models/hy-v3.cpp](src/models/hy-v3.cpp).

When touching fork code, keep it isolated from upstream paths (it is not proposed upstream) and preserve the
byte-identical guarantee of the non-async streaming modes.

## Build

CMake-based. Common configurations:

```sh
cmake -B build                       # CPU only
cmake -B build -DGGML_CUDA=ON        # NVIDIA CUDA (the fork's primary target)
cmake -B build -DGGML_VULKAN=ON      # Vulkan
cmake --build build --config Release -j
```

Binaries land in `build/bin/`. For debug: `-DCMAKE_BUILD_TYPE=Debug` (single-config generators) or
`--config Debug` (multi-config like Visual Studio). See [docs/build.md](docs/build.md) for every backend
(Metal, HIP, SYCL, MUSA, CANN, OpenCL, etc.). `CMakePresets.json` defines named presets
(e.g. `x64-windows-llvm-release`, `arm64-windows-llvm-release`).

## Test

Tests are gated behind `-DLLAMA_BUILD_TESTS=ON` and run with ctest:

```sh
cmake -B build -DLLAMA_BUILD_TESTS=ON && cmake --build build
ctest --test-dir build --output-on-failure          # all tests
ctest --test-dir build -R test-moe-stream -V         # a single test by name/regex
```

Test sources are `tests/test-*.cpp` (registered via `llama_build_and_test` in `tests/CMakeLists.txt`).
The fork's streaming logic is covered by [tests/test-moe-stream.cpp](tests/test-moe-stream.cpp) - run it
after any change to the MoE-streaming code.

## Running MoE streaming (fork feature)

All modes are opt-in via CLI flags; behavior is further tuned by environment variables. Streaming forces the
expert tensors to mmap-backed CPU memory so only selected/cached experts occupy VRAM.

| Flag | Meaning |
|------|---------|
| `--moe-stream` | Compaction: copy only the experts selected this step to VRAM. Byte-identical, slowest, smallest VRAM. |
| `--moe-stream-async` | Persistent VRAM expert cache reused across tokens. Byte-identical by default. Best VRAM/speed balance. |
| `--moe-stream-cache N[MB\|GB]` | Implies `--moe-stream`, records a resident byte budget. |

Env knobs (see README for the full table): `LLAMA_MOE_CACHE_CAP=N` (experts/layer resident),
`LLAMA_MOE_CACHE_GETROWS=1` (full-resident + GPU `get_rows` remap, fastest correct path),
`LLAMA_MOE_ASYNC=1` (**experimental** drop-on-miss loader - diverges from a non-streamed run; only safe on
robust MoEs), `LLAMA_MOE_IO_THREADS=N` (prefill disk-read threads, raise on NVMe).

Important runtime notes:
- The persistent-cache / async fast paths only engage on single-token decode (`n_tokens <= 1`). With
  `llama-server`, use `-np 1`; continuous batching (`-np N`, N>1) makes each step `n_tokens > 1` and falls
  back to slow per-step compaction. Treat server use as experimental.
- The fork's `-fit` device-fitting (see below) runs a separate measurement pass; pass `-fit off` when
  benchmarking streaming in isolation.

## Architecture

Layered, from lowest to highest:

- **`ggml/`** - the tensor library and compute backends (CPU, CUDA, Vulkan, Metal, ...). Backend-agnostic
  op graph; backends are selectable at runtime (`--device`, `--list-devices`) and can be built as loadable
  DLLs with `-DGGML_BACKEND_DL=ON`. The MoE-streaming code deliberately reuses the stock `ggml_mul_mat_id`
  and `get_rows` kernels rather than adding custom CUDA kernels.
- **`src/`** - the `llama` library (public C API in [include/llama.h](include/llama.h)). Key units:
  - `llama-model*.cpp` / `llama-arch.cpp` - model definitions and the per-architecture registry.
  - `src/models/*.cpp` - one file per architecture's decoder graph (e.g. `hy-v3.cpp`).
  - `llama-graph.cpp` - graph construction; `build_moe_ffn` here is the hook point for expert streaming.
  - `llama-moe-stream.{h,cpp}` - **fork-only** streaming cache. Three cooperating pieces: an LRU byte-budget
    host cache (`llama_moe_stream_cache`), expert *compaction* (build a small tensor of just the selected
    experts + remap routing ids, keeping stock kernels valid), and *persistent VRAM caches*
    (`llama_moe_cache_build*` / `llama_moe_layer_cache_*`) that keep hot experts resident across tokens with
    a GPU-side slot-table `get_rows` remap and a background loader. Read the header comments first - they
    document the invariants (byte-identical vs drop-on-miss, per-tensor vs per-layer residency).
  - `llama-kv-cache*.cpp`, `llama-memory*.cpp` - KV cache and memory management (incl. hybrid/recurrent/iswa).
  - `llama-sampler.cpp`, `llama-grammar.cpp`, `llama-vocab.cpp`, `llama-chat.cpp` - sampling, grammar/GBNF,
    tokenizer, chat templates.
- **`common/`** - shared helpers linked by tools: `arg.cpp` (CLI arg parsing - where flags like `-fit`,
  `--moe-stream*` are defined), `common.cpp`, and `fit.{h,cpp}` (**fork** `-fit` device-memory fitting that
  spills whole layers to system RAM; independent of and combinable with expert streaming).
- **`tools/`** - end-user programs, each its own `main.cpp`: `server` (OpenAI-compatible HTTP server, the
  largest sub-project), `cli` and `completion` (interactive/one-shot inference), `llama-bench`,
  `perplexity`, `quantize`, `imatrix`, `mtmd` (multimodal), `fit-params` (**fork** `-fit` helper), plus
  `parser`, `ui`, `rpc`, `tts`, `gguf-split`, etc.
- **`examples/`** - smaller, self-contained sample programs.
- **`gguf-py/`** and the `convert_*.py` scripts - Python side for creating/converting GGUF model files
  (config in `pyproject.toml`; `requirements/`).

### Adding or modifying an architecture

Register it in `llama-arch.{h,cpp}`, add hparam/tensor loading in `llama-model.cpp`, and implement the
decoder graph in a new `src/models/<name>.cpp`. Follow the pattern of an existing model (the fork's
`hy-v3.cpp` is a MoE reference). See [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md).

## Chat templates and output parsing

Model output parsing uses a PEG parser and a higher-level auto-parser rather than ad-hoc regex. If working
on chat/tool-call parsing, read [docs/development/parsing.md](docs/development/parsing.md),
[docs/autoparser.md](docs/autoparser.md), and [common/jinja/README.md](common/jinja/README.md). Related
tests: `tests/test-chat*.cpp`, `tests/peg-parser/`.
