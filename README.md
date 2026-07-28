# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

> [!NOTE]
> This is the **CatEngine** fork of `llama.cpp`. In addition to everything upstream provides, it adds
> **on-demand MoE expert streaming from disk** (run Mixture-of-Experts models whose expert weights are
> far larger than your VRAM/RAM) and support for the **Hunyuan-v3 (`hy_v3`)** architecture. See
> [CatEngine fork additions](#catengine-fork-additions) below. All upstream functionality is unchanged.

## CatEngine fork additions

This fork lets you run very large Mixture-of-Experts (MoE) models on modest hardware by keeping the
routed expert weights **out of VRAM** and pulling only the experts each token actually needs, on the fly.
As a milestone, it runs a **170 GB / 299B-parameter Hunyuan-v3 MoE on a single 32 GB GPU** by streaming
experts from an SSD.

### Technical background

In a MoE layer only a handful of experts (e.g. 8 of 192) are active per token, yet upstream keeps the
entire `[n_embd, n_ff, n_expert]` expert tensor resident. For large models that tensor dwarfs the rest of
the weights, which is what forces huge VRAM/RAM requirements. This fork treats the experts as a
three-tier cache instead:

```
SSD (mmap, full weights)  ->  RAM (OS page cache)  ->  VRAM (routing-driven expert cache)
```

Each MoE step, the router's selection drives which experts are materialized on the compute device. The
implementation lives in [src/llama-moe-stream.cpp](src/llama-moe-stream.cpp) and hooks into
`build_moe_ffn` in [src/llama-graph.cpp](src/llama-graph.cpp). It stays compatible with the stock
`ggml_mul_mat_id` kernels (no custom CUDA kernels): experts are either **compacted** into a small tensor
of just the selected experts with routing ids remapped, or served from a **persistent VRAM cache** whose
slot table is remapped on the GPU via `get_rows`. Prefill reads are parallelized and prefetched across a
disk-read thread pool. The design is inspired by SSD expert-streaming engines (three-tier expert cache,
routing-driven residency) adapted to ggml's single-3D-expert-tensor layout.

This is separate from the fork's `-fit` device-memory fitting (which spills whole layers to system RAM).
Both decide where the expert weights live, but by opposite strategies - `-fit` pre-plans a static layer
split up front, while streaming pins all experts to CPU and fills VRAM dynamically at runtime - so they
should not run together. Use `-fit off` with streaming (see the note under Usage).

### What's new

- **MoE expert streaming** (opt-in, off by default): stream routed experts from disk instead of keeping
  them resident. Two flags (`--moe-stream`, `--moe-stream-async`), see [Usage](#moe-streaming-usage).
- **Auto-sized cache tiers**: both the VRAM expert cache (`LLAMA_MOE_CACHE_CAP`) and the host-RAM pool
  (`LLAMA_MOE_RAM_CAP`) size themselves from free VRAM / available system RAM at startup - you normally set
  neither. They fill the device and host uniformly across all MoE layers (see the sizing note below).
- **Fused CUDA MoE-FFN op** (opt-in, `LLAMA_MOE_FUSED=1`, CUDA only): a single CUDA-native op does the
  per-layer expert sync-load + gate/up + SwiGLU + down + weighted sum, replacing a per-layer CPU op (and its
  scheduler split, 160 splits/token -> 4). Quality-identical to the default path, but measured decode is
  **slower** on the disk/H2D-bound streaming path (the fused op serializes the expert load with compute,
  losing the implicit load//compute overlap the split provided), so it is **off by default**. Kept for CUDA
  setups where the split, not the load, dominates.
- **Startup RAM-pool prefill** (on by default with `--moe-stream-async`): fills the host-RAM expert pool at
  full NVMe bandwidth during the prompt-eval window instead of letting decode warm it token by token, so the
  decode rate reflects the warm steady state from the first token. Set `LLAMA_MOE_PREFILL=0` to disable.
- **Parallel, prefetched disk I/O** for the prefill expert reads (thread pool + OS prefetch + offset-sorted
  reads), tuned for high-throughput NVMe.
- **Hunyuan-v3 (`hy_v3`) architecture support** ([src/models/hy-v3.cpp](src/models/hy-v3.cpp)): sigmoid
  gating with expert-probability bias, shared experts, Q/K-norm GQA, YaRN rope. Base decoder graph only
  (no MTP/speculative head). Reference: [ggml-org/llama.cpp#25395](https://github.com/ggml-org/llama.cpp/pull/25395).

### MoE streaming usage

Streaming is opt-in and off by default. Enabling it forces the routed-expert tensors to mmap-backed CPU
memory, so only the experts selected or cached each step occupy VRAM. There are two flags:

| Flag | What it does | Correctness |
|------|--------------|-------------|
| `--moe-stream` | **Compaction.** Each step copies only the experts selected that step to VRAM and runs the stock kernels over that small tensor. | Byte-identical to a non-streamed run. Largest VRAM saving, lowest throughput (experts cross PCIe every token). |
| `--moe-stream-async` | **Persistent VRAM expert cache.** Hot experts stay resident and reused across tokens. Enables the recommended async-streaming config by default (per-layer cache, off-page-cache reads, top-2 sync budget, RAM-pool prefill) - the per-layer streaming cache needed when a model exceeds VRAM *and* RAM. | Diverges from a non-streamed run (opt-in via this flag). For byte-identical output use `--moe-stream` or run non-streamed. Best balance of VRAM vs speed. Set any `LLAMA_MOE_*=0` to opt out of a piece. |

`LLAMA_MOE_CACHE_CAP` (VRAM experts/layer) and `LLAMA_MOE_RAM_CAP` (host-RAM experts/layer) are both
**auto-sized** - you normally set neither. For a model that fits in VRAM, or one only modestly larger,
plain `--moe-stream-async` is all you need and stays byte-identical.

#### Recommended: a model larger than VRAM *and* RAM

For the headline case - a MoE whose expert weights dwarf both VRAM and system RAM (e.g. Hunyuan-v3 Q4,
~170 GB, on a 24 GB GPU + 64 GB RAM) - enable the per-layer streaming cache with a small synchronous
expert budget. The VRAM cache and host-RAM pool auto-size to fill the device and host:

```bat
llama-completion -m hy3-q4.gguf -ngl 99 --moe-stream-async -fit off -no-cnv -p "..."
```

`--moe-stream-async` enables the recommended async-streaming configuration by default: the per-layer
streaming cache + background loader, off-page-cache expert reads, a top-2 synchronous expert budget, a
coverage early-stop that trims that budget when the resident experts already cover most of the routed
weight, and a one-shot startup RAM-pool prefill. `CACHE_CAP` fills VRAM and `RAM_CAP` fills host RAM
automatically (both printed at startup as `auto CACHE_CAP=N` / `auto RAM_CAP=N`). The top-2 `SYNC_BUDGET`
is what keeps output coherent on hard prompts: it guarantees each layer's two highest-weight experts are
resident before the matmul, while the rest reuse the previous (still-valid) expert. `SYNC_COVER=0.2` then
skips the second of those two loads on steps whose cached experts already cover enough weight, for roughly
+30% decode on average at coherent quality. (The async loader is non-deterministic, so a run may
occasionally degenerate into a repeat loop on reasoning-heavy prompts; re-running or `LLAMA_MOE_SYNC_COVER=0`
clears it.)

To opt out of any single piece, set that switch to `0` (e.g. `LLAMA_MOE_NOMMAP=0`, `LLAMA_MOE_SYNC_COVER=0`
to restore the pure fixed top-2 budget, or `LLAMA_MOE_SYNC_BUDGET=0` to let decode run pure-GPU and much
faster - though the stale reuse then **degrades quality on hard prompts**). An explicit value always wins
over the default. To try the fused CUDA op, add `LLAMA_MOE_FUSED=1` (off by default - see below).

Measured on an RTX 4090D (24 GB) + 64 GB RAM, coherent and non-degenerate through long generations:

| Model | Fits in | Decode | Prefill |
|-------|---------|--------|---------|
| Hunyuan-v3 **IQ2** (~92 GB) | neither VRAM nor RAM | **~13 tok/s** | ~20-40 tok/s |
| Hunyuan-v3 **Q4** (~170 GB) | neither VRAM nor RAM | **~3.8 tok/s** | ~10-15 tok/s |

Q4 is slower than IQ2 because each expert slab is ~1.8x the bytes, so fewer experts fit resident (both
tiers hold fewer) and the working set overflows VRAM+RAM further - decode is bounded by NVMe read
bandwidth. The table figures are the correct-quality path; the pure-GPU stale path is much faster but
degrades on hard prompts. The shipped default adds the coverage early-stop (`SYNC_COVER=0.2`); in paired
A/B tests (same prompt and seed, coverage off vs on) it gave a further **~+30% decode on IQ2 and ~+45% on
Q4** at coherent quality - Q4 gains more because each skipped expert sync is a saved NVMe read, its
dominant cost. Set `LLAMA_MOE_SYNC_COVER=0` for the fixed-budget baseline.

> [!IMPORTANT]
> **Use `-fit off` with streaming.** `-fit` and expert streaming both manage where the weights live, and
> they work against each other. `-fit` runs measurement passes that load the model with no weights resident;
> the VRAM cache auto-sizer (`auto CACHE_CAP`) can measure free VRAM during such a pass and freeze an
> oversized cap that later spills to system memory at runtime (a large decode slowdown). And when `-fit`
> tries to spill layers, it aborts on the expert CPU-override streaming has already set (a harmless warning;
> fit just does nothing). Streaming's own auto `CACHE_CAP` / `RAM_CAP` already fill the device and host, so
> `-fit` is redundant here - pass `-fit off`.

> [!IMPORTANT]
> **`llama-server`: run with `-np 1`.** The persistent VRAM cache and async fast paths only engage on
> single-token decode steps (`n_tokens <= 1`). `llama-server`'s continuous batching processes all parallel
> slots in one step, so with `-np N` (N > 1) - or whenever multiple requests are batched together - each
> step has `n_tokens > 1` and **bypasses the fast cache path**, falling back to per-step compaction. That
> is still correct but much slower, and with the async cache active it re-runs the synchronous prompt warm
> every step (very slow). Use `-np 1` so decode stays single-token; be aware concurrent clients still get
> batched and lose the fast path. Server use of these features is otherwise validated only for single-stream
> decode (`llama-cli`/`llama-completion`) - treat it as experimental.

> [!IMPORTANT]
> **`SYNC_BUDGET` governs decode only - prompt processing (prefill) uses a different, lossy path.**
> The top-N `SYNC_BUDGET`/`SYNC_COVER` selection engages only on single-token steps (`n_tokens <= 1`).
> A multi-token prefill batch (`n_tokens > 1`) instead loads each layer's working set bounded by
> `CACHE_CAP` and **drops the rest to a zeroed expert** (no stale reuse across a batch). When a model is
> far larger than VRAM so `CACHE_CAP` is a small fraction of the experts a batch touches, a prompt can be
> processed with most of each token's experts zeroed - the output stays fluent but can misread the prompt
> (e.g. ignoring injected instructions or tool definitions), and raising `SYNC_BUDGET` does **not** help
> because it never applies to prefill. Measured on Hunyuan-v3 IQ2 (192 experts, ~40 resident, 24 GB card):
> prompt perplexity 6.66 with the default `-ub`/large batch vs 3.07 lossless - a 2.17x gap.
>
> **Workaround: `-ub 1`.** With a micro-batch of one, every prefill token takes the same `n_tokens <= 1`
> path as decode, so `SYNC_BUDGET`/`SYNC_COVER` apply to prefill too and misses reuse a real (stale)
> expert instead of zero - prompt quality returns to the decode-path level (measured PPL back in the
> lossless band). The cost is speed: `-ub 1` serializes prefill token-by-token (measured ~7 tok/s prompt
> processing on the hy3-IQ2 case above, vs faster-but-lossy large batches), so it suits agent/tool use
> where following the prompt matters more than prompt-eval latency. This gap is specific to models whose
> per-batch expert working set exceeds the resident cache; a MoE that fits (or nearly fits) in VRAM does
> not zero experts and large `-ub` stays correct and faster. The byte-identical `--moe-stream` (compaction)
> mode is lossless at any `-ub`.

### Tuning (environment variables)

**Core** (the knobs you actually reach for):

| Variable | Default | Effect |
|----------|---------|--------|
| `LLAMA_MOE_ASYNC` | on with `--moe-stream-async` | The per-layer streaming cache + background loader - required to run a model that exceeds VRAM *and* RAM. On by default with `--moe-stream-async`; set `LLAMA_MOE_ASYNC=0` to fall back to a plain resident cache. |
| `LLAMA_MOE_CACHE_CAP=N` | auto | Experts-per-layer kept resident in the VRAM cache. **Unset = auto:** sized from free VRAM so the cache fills the device without spilling to system memory (adapts to `-c` context size and card). Set `N` to override; trades VRAM for quality. |
| `LLAMA_MOE_SYNC_BUDGET=N` | `2` with `--moe-stream-async` | On decode, synchronously load each layer's `N` highest-weight missing experts (the true top-N) before its matmul, reusing the previous (real, stale) expert for the rest. This is what makes decode **correct on hard prompts**; `N=2` is the quality knee for Hunyuan-v3. Set `0` to disable (decode runs pure-GPU and much faster, but stale reuse degrades quality on hard prompts). |
| `LLAMA_MOE_SYNC_COVER=F` | `0.2` with `--moe-stream-async` | Coverage early-stop layered on top of `SYNC_BUDGET`: skip a top-N miss's synchronous load once the step's already-resident experts cover fraction `F` of the routed gate weight (the low-weight tail then reuses its stale expert). Since Hunyuan-v3's top-2 gate weight sums to only ~0.34, `F=0.2` lets a well-covered step load one expert instead of two - **~+30% decode on IQ2, ~+45% on Q4 (paired A/B, same prompt/seed), coherent output**. A sweep found `0.15-0.19` is a seed-fragile cliff (repeat loops), so `0.2` is the safe knee. Tuned on Hunyuan-v3; other MoE architectures may want a different `F`. Set `0` to disable (pure fixed `SYNC_BUDGET`). Decode only. |
| `LLAMA_MOE_RAM_CAP=N` | auto | Experts-per-layer held in a locked host-RAM pool (three-tier: VRAM cache -> RAM pool -> disk). A VRAM miss reads from RAM (~25 GB/s) instead of faulting the model file from disk. **Unset = auto:** sized from available system RAM to fill the host, leaving headroom for the OS + mmap working set. Set `0` to disable the tier, or `N` to override. |
| `LLAMA_MOE_NOMMAP` | on with `--moe-stream-async` | Read experts from the model file by offset (own file handles) instead of the mmap pointer, bypassing the OS page cache so host RAM stays under the locked pool's control rather than an unbounded page cache that thrashes when the model is larger than RAM. Set `LLAMA_MOE_NOMMAP=0` to use the mmap path. |
| `LLAMA_MOE_FUSED=1` | off | Use the fused CUDA MoE-FFN op on the async decode path: one CUDA-native op does the expert sync-load + gate/up + SwiGLU + down + weighted sum, instead of a per-layer CPU remap op that forces a scheduler split (160 splits/token -> 4). Quality-identical, but measured decode is **~20% slower** on this disk/H2D-bound path (the fused op serializes the load with compute, losing the split's load//compute overlap), so it is **off by default**. Decode only; prefill, non-hy3 layers, and non-CUDA backends fall back automatically. |
| `LLAMA_MOE_PREFILL` | on with `--moe-stream-async` | Fill the host-RAM expert pool at full NVMe bandwidth once, during the prompt-eval window, instead of letting decode warm it token by token. The decode rate then reflects the warm steady state from the first token. Set `LLAMA_MOE_PREFILL=0` to disable. |
| `LLAMA_MOE_PREFILL_VRAM` | on with `--moe-stream-async` | After the RAM pool is warm, also fill each layer's VRAM slot cache with its top-scoring experts from the RAM pool (~25 GB/s) before decode, instead of warming VRAM one miss at a time over the first tokens. Quality-safe (VRAM is a subset of the RAM set; a wrong pick is just LRU-evicted). Set `LLAMA_MOE_PREFILL_VRAM=0` to disable. |

**Auto-sizing** (only if the auto `CACHE_CAP` / `RAM_CAP` needs nudging):

| Variable | Default | Effect |
|----------|---------|--------|
| `LLAMA_MOE_VRAM_FRAC=F` | `0.97` | Fraction of free VRAM (minus the reserve) the auto VRAM cache may use. Lower it if you see a system-memory spill. Ignored when `LLAMA_MOE_CACHE_CAP` is set. |
| `LLAMA_MOE_VRAM_RESERVE_MB=N` | `512` | VRAM (MiB) the auto cache keeps free as a fragmentation margin. KV/compute are already excluded from the measured free, so this is small; raise only if a spill persists. |
| `LLAMA_MOE_RAM_FRAC=F` | `0.90` | Fraction of (available RAM - reserve) the auto RAM pool may use. Ignored when `LLAMA_MOE_RAM_CAP` is set. |
| `LLAMA_MOE_RAM_RESERVE_MB=N` | `5120` | System RAM (MiB) the auto pool keeps free for the OS + the growing mmap working set. Raise it if you see paging or a runtime slowdown as the pool fills. |

**Loader / performance** (sensible defaults; rarely changed):

| Variable | Default | Effect |
|----------|---------|--------|
| `LLAMA_MOE_IO_THREADS=N` | `16` | Parallel disk-read threads for expert loads. Raise on fast NVMe. |
| `LLAMA_MOE_SYNC_THRESHOLD=F` | `0.5` | With `LLAMA_MOE_ASYNC=1`: on **prefill**, a step sync-loads its experts when the fraction of not-yet-resident routed experts exceeds `F` (else drops misses). Bounds the cold first step's disk reads to `CACHE_CAP` experts/layer. |
| `LLAMA_MOE_RAM_FILL=N` | `1024` | Experts the background loader promotes into the RAM pool per tick (across all layers). Its disk reads run off the lock, so this is fill speed, not stall - keep it high so a large pool fills before the sync path misses to disk. |
| `LLAMA_MOE_WEIGHTED_EVICT=0` | on | Loader protects recurring high-weight (top-1/top-2) experts from eviction so the stable core stays resident. Set `0` for plain age-LRU. |
| `LLAMA_MOE_SENTINELS=N` | `1` | With `LLAMA_MOE_ASYNC=1`: zero "sentinel" slots per layer for dropped experts (each costs one expert slab of VRAM/layer). `1` suffices in practice. |
| `LLAMA_MOE_PINNED_STAGE=0` | on | Stage no-mmap expert reads through a single shared page-locked host buffer so the H2D runs at full PCIe rate. On by default; `0` uses a plain pageable buffer. Byte-identical either way. |

**Diagnostics:**

| Variable | Default | Effect |
|----------|---------|--------|
| `LLAMA_MOE_DIAG=1` | off | Print decode miss-rate, per-token load/lock-wait timing, and top-2 residency. |
| `GGML_MOE_SPLIT_DIAG=1` | off | Per-token breakdown of scheduler split time (CPU-split vs GPU-split copy/sync and compute). |

> [!IMPORTANT]
> **`LLAMA_MOE_CACHE_CAP` and `LLAMA_MOE_RAM_CAP` are auto-sized by default** - you normally set neither.
> The VRAM cache uses about `(CACHE_CAP + LLAMA_MOE_SENTINELS) x n_moe_layers x per_expert_slab_bytes`; if
> it overflows the device the driver does **not** error - it silently spills to system RAM (CUDA "system
> memory fallback"), and decode becomes PCIe-bound and can be ~10x slower. The auto sizing measures free
> VRAM / available RAM once at the first MoE layer (before any cache allocates, so the reading is the true
> ceiling) and applies the **same** cap to every layer, filling the device and host uniformly - it adapts
> to `-c` context size, card, and quantization (a bigger per-expert slab yields a smaller cap). The chosen
> values print at startup (`auto CACHE_CAP=N -> ~X GiB`, `auto RAM_CAP=N -> ~Y GiB`). If you see a VRAM
> spill, lower `LLAMA_MOE_VRAM_FRAC`; if the RAM pool starves the OS, raise `LLAMA_MOE_RAM_RESERVE_MB`. For
> reference, Hunyuan-v3 Q4 has an ~11.7 MiB per-expert slab (~0.9 GiB per cap unit across its 80 MoE
> layers), so `CACHE_CAP=16` (~21.5 GB) fills a 24 GB card; IQ2's ~6.4 MiB slab lets ~36 experts/layer fit.


### Performance and caveats

- **A MoE that fits in VRAM** (used to validate correctness): `--moe-stream` and `--moe-stream-async` are
  both **byte-identical** to a non-streamed run. Async keeps hot experts resident and approaches
  non-streamed throughput; `--moe-stream` uses the least VRAM but moves experts over PCIe every token.
- **Hunyuan-v3 on an RTX 4090D (24 GB) + 64 GB RAM** (model exceeds VRAM *and* RAM), correct-quality
  (`SYNC_BUDGET=2`) path: **IQ2 (~92 GB) ~13 tok/s decode, Q4 (~170 GB) ~3.8 tok/s decode**, coherent and
  non-degenerate through long generations. Decode is far below the pure-GPU stale path because correct
  quality forces each layer's true top-2 experts resident before its matmul, and it is bounded by NVMe
  read bandwidth: the per-token working set overflows VRAM+RAM, so misses fault from disk. A smaller
  quantization (more experts fit resident) or more VRAM/RAM is the lever, not the execution model.

Caveats:

- **Prefill is disk-bound; decode is loader/PCIe/disk-bound.** The I/O path (parallel + prefetched +
  offset-sorted) is built for NVMe, where scattered reads run near full bandwidth; on a SATA SSD prefill
  saturates at the drive's scattered-read rate. Raise `LLAMA_MOE_IO_THREADS` on fast NVMe. Decode rate
  also **slides with generation length** as a longer output routes to a wider expert set (more disk misses).
- The async streaming cache reuses a **stale** expert on a miss rather than blocking. With the default
  top-2 `SYNC_BUDGET` this stays coherent on **robust** production MoEs; with `LLAMA_MOE_SYNC_BUDGET=0`, or
  on **fragile** merged models that cannot tolerate any stale expert, output can degrade. Verify quality on
  your model before relying on it.
- The fused CUDA MoE-FFN op (`LLAMA_MOE_FUSED=1`) is quality-identical to the default path and eliminates
  the per-layer scheduler split, but measured decode is slower on this disk/H2D-bound path, so it is off by
  default. Enable it only where the split, not the expert load, dominates. Skipped on non-CUDA backends.
- Expert LoRA is not applied on the streamed path.
- These features are a fork-local addition and are **not** proposed upstream.

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)
- WebGPU support is now available in the browser, see a blog/demo introducing it [here](https://reeselevine.github.io/llamas-on-the-web/).

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7) (incl. Hunyuan-v3 `hy_v3`, CatEngine fork)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)
- [x] [Mellum models](https://huggingface.co/JetBrains/models?search=mellum)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Ruby: [docusealco/rllama](https://github.com/docusealco/rllama)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from Hugging Face Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
- [LLMKube](https://github.com/defilantech/llmkube) - Kubernetes operator for llama.cpp with multi-GPU and Apple Silicon Metal
  support"
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. The `MODEL_ENDPOINT` must point to a Hugging Face compatible API endpoint.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
