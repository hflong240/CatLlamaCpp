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
It runs models whose routed experts are several times the size of VRAM *and* system RAM combined: the
current primary target is **DeepSeek-V4-Flash** (78 GiB of routed experts alone, on a 24 GB GPU + 64 GB
RAM), and it also runs a **170 GB / 299B-parameter Hunyuan-v3 MoE on a single 32 GB GPU**.

Prompt processing is **lossless** here, not just fluent: the prefill expert-group sweep (on by default)
evaluates every routed expert a batch selects, so nothing is dropped or substituted. See
[What's new](#whats-new).

### Technical background

In a MoE layer only a handful of experts (6 of 256 for DeepSeek-V4-Flash, 8 of 192 for Hunyuan-v3) are
active per token, yet upstream keeps the
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

- **DeepSeek-V4-Flash (`deepseek4`) architecture support**: 43 MoE layers, 256 experts with 6 active,
  hyper-connections, a lightning indexer, and static token-id expert routing on the first layers. Ported
  from [ggml-org/llama.cpp#24162](https://github.com/ggml-org/llama.cpp/pull/24162). This is the fork's
  main test target; the streaming defaults below are tuned on it.
- **Chunked lightning-indexer scoring** (`deepseek4`, on by default): the indexer scores every cached key
  against every query head, so its intermediate is `[n_lid, n_tokens, indexer_n_head]` - and it is
  materialized **twice**, as the `mul_mat` result and as its transposed copy, because the per-head weighting
  has to reduce over the head axis. That is `2 * n_lid * indexer_n_head * 4` bytes **per token** (2 MiB/token
  at `n_lid` 4096 and 64 heads), and `n_lid` is the indexer's own KV size, i.e. `n_ctx/4`. The term therefore
  grows as `n_ubatch * n_ctx` and dominated the reserved compute buffer: **34 GiB** measured at
  `-c 131072 -ub 2048`. That is the real reason long-context runs needed a hand-tuned VRAM reserve or simply
  ran out of memory - it was never the expert cache.
  The fix needs no new kernel. Each token's indexer scores depend only on its own query row and on all
  `n_lid` key rows, so the token axis is scored in passes and the per-pass masked scores are concatenated
  along the token axis, with a single `ggml_top_k` at the end. The result is still a set of **global** row
  indices, so nothing has to be remapped and every downstream mask keeps its shape. Peak becomes
  `pass_size * n_lid * indexer_n_head * 8` bytes instead of the whole ubatch. Measured compute buffer at
  `-ub 2048`: **34013 -> 2885 MiB** at `-c 131072`, **5149 -> 1170 MiB** at `-c 16384`. Byte-identical
  (greedy MD5 over a 411-token prompt spanning 4 passes, 4 runs, loader frozen), and decode is untouched -
  a single-token step is a single pass, and the `bs=1` graph stays node-for-node identical. Tune with
  `LLAMA_DSV4_LID_CHUNK_MB` / `LLAMA_DSV4_LID_MIN_CHUNK`; `LLAMA_DSV4_LID_CHUNK_MB=0` restores the old
  single-pass graph.
- **Lossless prefill via the expert-group sweep** (`LLAMA_MOE_PREFILL_SWEEP`, **on by default** with
  `--moe-stream-async`): prompt processing no longer rations a batch against the VRAM cache and drop the
  rest. The expert *index* space is partitioned into `ceil(n_expert / capacity)` static groups and the MoE
  FFN runs once per group, so every selected expert is evaluated and the per-group sum reproduces the full
  layer output exactly. This replaces the old `-ub 1` workaround: measured on DeepSeek-V4-Flash IQ2
  (1195-token prompt, 24 GB card), prefill **75.9 tok/s lossless vs 13.4 tok/s** for `-ub 1`
  (**5.7x**) at the same decode rate, where `-ub 1` still only got 4.18 of 6 routing positions right per
  token. Set `LLAMA_MOE_PREFILL_SWEEP=0` for the old behaviour.
- **Capacity auto-sizing fixed to the measured throughput knee**: filling VRAM to the last byte is
  *slower*, not faster. The driver pages part of the working set once total committed VRAM crosses a
  threshold, and this is invisible in every VRAM counter - it shows up only as throughput, on **both**
  prefill and decode (~1.5x on DeepSeek-V4-Flash IQ2 past the knee), while extra cache below the knee buys
  nothing. The auto sizer therefore holds back an absolute amount rather than a thin fragmentation margin.
- **Capacity sized against the *measured* compute buffer** (on by default): the capacity has to be chosen
  during graph build, before any compute buffer exists, so on its own it can only hold back a flat guess
  (`LLAMA_MOE_VRAM_RESERVE_MB`, 3072). No flat value can be right for a buffer that scales with
  `n_ubatch * n_ctx`: at `-c 131072` the buffer measured 4.3 GiB at `-ub 256` and 34 GiB at `-ub 2048`, so
  lowering `-ub` only shrank the overshoot linearly and a hand-tuned reserve was unavoidable.
  Instead, the first reserve pass now runs with the flat guess purely to *build* the graph - the buffer size
  does not depend on the capacity, since the slot pools live in their own allocation - then the real
  device-side size is read back, the caches are dropped, and the capacity is recomputed against it. No
  estimation coefficients are involved; the only remaining constant is what free VRAM still does not
  account for (`LLAMA_MOE_VRAM_MARGIN_MB`). This splits the old unportable 3072 into a part that varies with
  the configuration (now measured) and a part that does not (now the only knob). Measured on
  DeepSeek-V4-Flash IQ2 / 24 GB: capacity is left **unchanged** wherever the flat guess was already right
  (`-c 16384 -ub 2048` -> 36; `-c 131072 -ub 512/256` -> 33) and tightened only where it was not
  (`-c 131072 -ub 2048`: the 2885 MiB buffer left the flat 3072 with a 187 MiB margin, so 33 was
  overcommitted -> 28). Throughput is unchanged there (both sit below the knee, as the sizing note predicts);
  what it buys is 1.6 GiB less committed VRAM, which is the difference between running and not running on a
  card that was previously OOMing. Set `LLAMA_MOE_AUTOCAP_RECAP=0` for the flat-guess behaviour.
- **Sentinel GEMM skip** (`LLAMA_MOE_SENTINEL_SKIP`, on by default, CUDA): during the sweep most routing
  positions point at permanently-zero "sentinel" expert slots, and their GEMM tiles are now skipped rather
  than multiplied against zeros. Provably bit-identical (verified byte-for-byte on a greedy run), and it
  matters more the smaller the VRAM cache is.
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
| `--moe-stream-async` | **Persistent VRAM expert cache.** Hot experts stay resident and reused across tokens. Enables the recommended async-streaming config by default (per-layer cache, off-page-cache reads, top-2 sync budget, RAM-pool prefill, **lossless prefill sweep**) - the per-layer streaming cache needed when a model exceeds VRAM *and* RAM. | Diverges from a non-streamed run (opt-in via this flag). For byte-identical output use `--moe-stream` or run non-streamed. Best balance of VRAM vs speed. Set any `LLAMA_MOE_*=0` to opt out of a piece. |

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
| DeepSeek-V4-Flash **IQ2** (78 GiB routed experts) | neither VRAM nor RAM | **~14.5 tok/s** | **~76 tok/s (lossless)** |
| Hunyuan-v3 **IQ2** (~92 GB) | neither VRAM nor RAM | **~13 tok/s** | ~20-40 tok/s |
| Hunyuan-v3 **Q4** (~170 GB) | neither VRAM nor RAM | **~3.8 tok/s** | ~10-15 tok/s |

The DeepSeek-V4-Flash row is the shipped default configuration (`--moe-stream-async -fit off -ub 2048`),
N=3 medians on a 1195-token prompt with a 320-token generation. Its prefill is both lossless and ~5.7x the
old `-ub 1` path; the Hunyuan-v3 prefill figures predate the expert-group sweep.

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
> **`SYNC_BUDGET` governs decode only.** The top-N `SYNC_BUDGET`/`SYNC_COVER` selection engages only on
> single-token steps (`n_tokens <= 1`). Prefill takes a different path, and since the expert-group sweep that
> path is **lossless** (see below), so raising `SYNC_BUDGET` neither helps nor hurts prompt processing.

> [!NOTE]
> **Prefill is lossless by default (the expert-group sweep), and `-ub 1` is obsolete.**
> Historically a multi-token prefill batch loaded each layer's working set bounded by `CACHE_CAP` and
> **dropped the rest to a zeroed expert**. On a model far larger than VRAM that meant most of each token's
> experts were zeroed: output stayed fluent but could misread the prompt (ignoring injected instructions or
> tool definitions), and the only fix was `-ub 1`, which serialized prefill token by token.
>
> `LLAMA_MOE_PREFILL_SWEEP` (on by default) removes the loss instead of working around it: the expert index
> space is split into `ceil(n_expert / capacity)` static groups, the MoE FFN is evaluated once per group with
> out-of-group positions routed to a genuine zero, and the group outputs are summed - reproducing the full
> layer exactly with nothing dropped. Expert *bytes read* are unchanged (each expert still reaches VRAM at
> most once per step); the cost is `n_groups` FFN passes, which measurement shows is a small share of prefill
> because prefill is bound by moving expert weights, not by arithmetic.
>
> Measured on DeepSeek-V4-Flash IQ2 (1195-token prompt, 320-token generation, 24 GB card + 64 GB RAM, N=3
> medians):
>
> | Prefill path | Prompt eval | Decode | Routing positions correct |
> |---|---|---|---|
> | sweep (default, `-ub 2048`) | **75.9 tok/s** | 14.45 tok/s | **6 of 6** |
> | `-ub 1` (`LLAMA_MOE_PREFILL_SWEEP=0`) | 13.4 tok/s | 14.39 tok/s | 4.18 of 6 |
>
> So the sweep is ~5.7x faster than the old lossless-ish workaround *and* strictly more correct, at the same
> decode rate. It engages only when `capacity < n_expert` (a model that fits needs no grouping) and only on
> multi-token batches. Note it raises the per-layer zero-sentinel count to `n_expert_used`, which costs a few
> expert slabs of VRAM per layer and therefore a slightly smaller auto capacity; if you specifically want
> `-ub 1`, set `LLAMA_MOE_PREFILL_SWEEP=0` so the sentinel count drops back to 1.
>
> The byte-identical `--moe-stream` (compaction) mode remains lossless at any `-ub`.

> [!TIP]
> **Long prompts are bound by expert bytes, not compute.** Every ubatch step must bring each expert it
> selects into VRAM once, and on DeepSeek-V4-Flash a 2048-token step selects nearly all 256 experts in every
> layer - about 59 GiB of expert weights per step, moved at PCIe and NVMe speed (measured: 77% of prompt-eval
> wall time is inside the expert loader). That cost is **per step**, so a 15k-token prompt at `-ub 2048` pays
> it 8 times. A larger `-ub` is therefore the main lever on long-prompt latency, and a warm RAM pool
> (`LLAMA_MOE_RAM_CAP`, auto) is what keeps half of those reads off the disk.

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
| `LLAMA_MOE_PREFILL_SWEEP` | on with `--moe-stream-async` | **Lossless prefill.** Partition the expert index space into `ceil(n_expert / capacity)` static groups and evaluate the MoE FFN once per group, so a prompt batch never has to drop experts it selected. Replaces the old `-ub 1` workaround (measured 5.7x faster *and* strictly more correct on DeepSeek-V4-Flash IQ2). Engages only when `capacity < n_expert` and only on multi-token batches, so decode is untouched. Costs `n_expert_used` sentinel slots per layer (see `LLAMA_MOE_SENTINELS`). Set `0` to restore capacity-rationed, lossy prefill. |
| `LLAMA_MOE_PREFILL` | on with `--moe-stream-async` | Fill the host-RAM expert pool at full NVMe bandwidth once, during the prompt-eval window, instead of letting decode warm it token by token. The decode rate then reflects the warm steady state from the first token. Set `LLAMA_MOE_PREFILL=0` to disable. |
| `LLAMA_MOE_PREFILL_VRAM` | on with `--moe-stream-async` | After the RAM pool is warm, also fill each layer's VRAM slot cache with its top-scoring experts from the RAM pool (~25 GB/s) before decode, instead of warming VRAM one miss at a time over the first tokens. Quality-safe (VRAM is a subset of the RAM set; a wrong pick is just LRU-evicted). Set `LLAMA_MOE_PREFILL_VRAM=0` to disable. |

**Auto-sizing** (only if the auto `CACHE_CAP` / `RAM_CAP` needs nudging):

| Variable | Default | Effect |
|----------|---------|--------|
| `LLAMA_MOE_VRAM_FRAC=F` | `0.97` | Fraction of (free VRAM minus the reserve) the auto VRAM cache may use. Deliberately near 1.0: the real allowance is the absolute reserve below, because the amount of VRAM that must stay free does **not** scale with card size. Ignored when `LLAMA_MOE_CACHE_CAP` is set. |
| `LLAMA_MOE_VRAM_RESERVE_MB=N` | `3072` | VRAM (MiB) the auto cache leaves free **before the compute buffer has been measured**, i.e. the first reserve pass only. It is not a fragmentation margin: past a total-committed-VRAM threshold the driver pages part of the working set, costing ~1.5x on **both** prefill and decode, and no VRAM counter reveals it (see the sizing note below). Setting it explicitly pins the reserve for *both* passes, disabling the measured path below. |
| `LLAMA_MOE_AUTOCAP_RECAP=0` | on | After the first reserve pass, read the real device-side compute-buffer size, drop the expert caches, and recompute the capacity against it (see [What's new](#whats-new)). Zero estimation coefficients - the capacity follows a measured number, so it adapts to `-c`, `-ub`, KV quantisation and architecture on its own. Costs one extra reserve pass plus one extra VRAM prewarm at startup. Set `0` to keep the flat-guess capacity. |
| `LLAMA_MOE_VRAM_MARGIN_MB=N` | `1856` | With the measured path above: VRAM (MiB) held back **in addition to** the compute buffer, which by then is already excluded from the free figure. This is the portable half of the old flat 3072 - the part that does not vary with `-c` / `-ub`. It is not just fragmentation: the driver starts paging before free VRAM reaches zero. The default is reverse-engineered to leave the capacity unchanged wherever the flat guess was already right, so it is a **no-regression value, not a swept optimum** - `1536` measured one capacity step past the knee (decode 7.69/7.67/7.15 vs 10.01/8.28/8.17 tok/s, non-overlapping). Re-sweep it per card against tok/s; the VRAM audit cannot find the knee for you. |
| `LLAMA_MOE_RAM_FRAC=F` | `0.97` | Hard ceiling on the auto RAM pool as a fraction of available RAM. Only binds when `LLAMA_MOE_RAM_RESERVE_MB` is set too low to be safe on its own. Ignored when `LLAMA_MOE_RAM_CAP` is set. |
| `LLAMA_MOE_RAM_RESERVE_MB=N` | `5120` | System RAM (MiB) the auto pool leaves free, for the OS + the growing mmap working set. **Exact:** set `N` and that much available RAM stays free. Raise it if other software gets squeezed, or if you see paging / a runtime slowdown. |

**Loader / performance** (sensible defaults; rarely changed):

| Variable | Default | Effect |
|----------|---------|--------|
| `LLAMA_MOE_IO_THREADS=N` | `16` | Parallel disk-read threads for expert loads. Raise on fast NVMe. |
| `LLAMA_MOE_SYNC_THRESHOLD=F` | `0.5` | With `LLAMA_MOE_ASYNC=1`: on **prefill**, a step sync-loads its experts when the fraction of not-yet-resident routed experts exceeds `F` (else drops misses). Bounds the cold first step's disk reads to `CACHE_CAP` experts/layer. |
| `LLAMA_MOE_RAM_FILL=N` | `1024` | Experts the background loader promotes into the RAM pool per tick (across all layers). Its disk reads run off the lock, so this is fill speed, not stall - keep it high so a large pool fills before the sync path misses to disk. |
| `LLAMA_MOE_WEIGHTED_EVICT=0` | on | Loader protects recurring high-weight (top-1/top-2) experts from eviction so the stable core stays resident. Set `0` for plain age-LRU. |
| `LLAMA_MOE_SENTINELS=N` | `n_expert_used` with the sweep, else `1` | With `LLAMA_MOE_ASYNC=1`: zero "sentinel" slots per layer that dropped or out-of-group experts route to (each costs one expert slab of VRAM/layer, so it reduces the auto capacity). The prefill sweep needs one per routing position, since a token can have all of them outside the current group; without the sweep, `1` suffices. |
| `LLAMA_MOE_SENTINEL_SKIP=0` | on (CUDA) | Skip the GEMM tiles of the zero sentinel slots instead of multiplying against zeroed weights. Bit-identical (the slabs are zeroed once at cache creation and no code path can ever write a slot above the capacity), and worth more the more groups the sweep uses: measured +0.6% at capacity 36 (8 groups) and +9.3% at capacity 8 (32 groups) on DeepSeek-V4-Flash IQ2. Implemented via `op_params` on `ggml_mul_mat_id`; other backends ignore it and compute the same zeros. |
| `LLAMA_MOE_PREFILL_DRAIN=1` | off | Block the first decode token until the background loader has worked off its backlog. Measured **negative on both axes** (decode slower *and* seconds of added wall clock) because its "loader went idle" condition is never reached while the RAM pool is still chasing a target it cannot hold, so it always runs to its internal deadline. Off by default; kept only for A/B. |
| `LLAMA_MOE_PINNED_STAGE=0` | on | Stage no-mmap expert reads through a single shared page-locked host buffer so the H2D runs at full PCIe rate. On by default; `0` uses a plain pageable buffer. Byte-identical either way. |

**Diagnostics:**

| Variable | Default | Effect |
|----------|---------|--------|
| `LLAMA_MOE_DIAG=1` | off | Print decode miss-rate, per-token load/lock-wait timing and top-2 residency, plus the per-ubatch-step prefill breakdown (loader wall time, disk vs H2D, experts loaded, RAM-pool hit rate) - the numbers that show whether prefill is bound by bytes or by compute. |
| `LLAMA_MOE_LAYERDBG=1` | off | Per-layer decode residency profile (hit / stale-substitution percentage per MoE layer). Reveals layers that are structurally harder than the aggregate suggests - on DeepSeek-V4-Flash the statically token-id-routed first layers run far colder than the rest, because their locality follows the output token distribution rather than the hidden state. |
| `LLAMA_MOE_SWEEPDBG=1` | off | Verify the prefill sweep's invariant: over the `n_groups` passes of a step every routing position must resolve to its real expert exactly once, so the reported HIT rate must equal `1/n_groups`. |
| `GGML_MOE_SPLIT_DIAG=1` | off | Per-token breakdown of scheduler split time (CPU-split vs GPU-split copy/sync and compute). |

**Compute buffer** (`deepseek4`; these size the reserved graph buffer, not the expert cache):

| Variable | Default | Effect |
|----------|---------|--------|
| `LLAMA_DSV4_LID_CHUNK_MB=N` | `256` | Byte budget for one pass of the chunked lightning-indexer scoring (see [What's new](#whats-new)). The pass size is derived as `N MiB / (2 * n_lid * indexer_n_head * 4)`, i.e. a **budget rather than a token count**, because `n_lid` is `n_ctx/4` - a fixed token count would let the peak grow with the context instead of holding it constant. Set `0` to score the whole ubatch in one pass (the pre-chunking graph). Lowering it below the default buys almost nothing: at `-c 16384 -ub 2048`, 64 MiB saved a further 8 MiB of compute buffer over 256 MiB while adding ~14000 graph nodes, because by then the indexer is no longer the peak. |
| `LLAMA_DSV4_LID_MIN_CHUNK=N` | `64` | Floor on the pass size in tokens, which wins over the budget above. Needed because the budget alone makes the pass **count** grow with `n_lid`, and each pass costs ~7 graph nodes per MoE layer, while `graph_max_nodes` falls back to `32 * n_tensors` below roughly 1000 tokens - so a small ubatch has the *least* node headroom exactly where the budget wants the *most* passes. Without this floor, `-c 131072 -ub 2048` asked for 128 passes and exhausted the graph context's object pool. With it, a very long context raises the peak instead of failing to build. |

One CLI flag belongs to the same accounting: **`--n-outputs-max N`** (`llama-cli` / `llama-completion`,
default `0` = `-b`). The reserved graph holds an `[n_vocab, N]` logits tensor, so `1` shrinks it to a single
row when only the last token is ever sampled - 1009 MiB to 0.5 MiB on DeepSeek-V4-Flash at `-ub 2048`. The
*net* compute-buffer saving is far smaller than the tensor (109 MiB measured) because the buffer is a
**peak**, not a sum, and the logits do not overlap whatever else is peaking. The library raises the value to
`--parallel` if set lower, since `output_reserve` floors its request at `n_seq_max`.

> [!IMPORTANT]
> **`LLAMA_MOE_CACHE_CAP` and `LLAMA_MOE_RAM_CAP` are auto-sized by default** - you normally set neither.
> The VRAM cache uses about `(CACHE_CAP + LLAMA_MOE_SENTINELS) x n_moe_layers x per_expert_slab_bytes`. The
> auto sizer measures free VRAM once at the first MoE layer and applies the **same** cap to every layer,
> filling the device uniformly; it adapts to `-c`, card and quantization (a bigger per-expert slab yields a
> smaller cap). The chosen values print at startup (`auto CACHE_CAP=N -> ~X GiB`, `auto RAM_CAP=N -> ~Y GiB`).
>
> **Bigger is not better, and you cannot see the limit in any VRAM counter.** Once total committed VRAM
> crosses a threshold the driver keeps part of the working set out of dedicated VRAM and pages it over PCIe.
> Measured on DeepSeek-V4-Flash IQ2 (24 GB card, 43 MoE layers, 7.19 MiB per expert, sweep on so each layer
> holds `cap + 6` slabs), prefill / decode tok/s against total slots per layer:
>
> | slots/layer | 38 | **42** | 44 | 46 | 47 | 50 |
> |---|---|---|---|---|---|---|
> | prefill | 63.8 | **76.1** | 75.9 | 47.3 | 47.6 | 50.3 |
> | decode | 14.44 | **14.61** | 12.21 | 10.86 | 9.35 | 10.41 |
>
> The knee is sharp between 44 and 46 slots and costs roughly 1.5x on **both** axes past it, while extra
> cache below the knee buys nothing. With the measured path on (the default), what puts the capacity under
> the knee is `LLAMA_MOE_VRAM_MARGIN_MB` on top of the measured compute buffer, not the flat
> `LLAMA_MOE_VRAM_RESERVE_MB=3072` that produced this table - the margin default was reverse-engineered to
> land on the same capacity here, and one step past it (margin 1536) was measurably slower on decode.
>
> Do **not** try to find that knee with a free-VRAM reading: the figure reported after allocation sits near
> 0.00 GiB at *any* cache size, because whatever headroom the sizer leaves is absorbed afterwards (raising the
> reserve by ~1 GiB left the reported free VRAM unchanged and simply grew the unaccounted portion). It is
> insensitive at the ~1 GiB scale, not useless: freeing the ~4 GiB the indexer chunking recovered did move it
> (0.00 -> 2.04 GiB at the same settings). Sweep
> `LLAMA_MOE_CACHE_CAP` against measured tok/s instead - the knee is sharp and shows on prefill as well as
> decode, so a couple of runs per value is enough. `tools/dev/bench_moe.sh` does the repetition, standby-list
> purge and median reporting.
>
> When you sweep, make sure the prompt spans **several** `-ub`-sized steps. A prompt short enough to fit one
> step measures startup and RAM-pool prefill, not prefill throughput: on a 411-token prompt at `-ub 2048` the
> *same* configuration returned 13.4 to 20.4 tok/s across runs, a 52% spread that swamps any real effect,
> while a 6894-token prompt put the same runs at 70-92 tok/s - the range this table was measured in.
>
> For reference: DeepSeek-V4-Flash IQ2 has a ~7.19 MiB per-expert slab (~309 MiB per cap unit across its 43
> MoE layers); Hunyuan-v3 Q4 has ~11.7 MiB (~0.9 GiB per cap unit across 80 layers) and IQ2 ~6.4 MiB. If the
> RAM pool starves the OS, raise `LLAMA_MOE_RAM_RESERVE_MB`.

> [!NOTE]
> The RAM reserve is sampled **once**, before the context, the KV cache and `llama-server`'s context
> checkpoints exist - whatever is allocated later eats into the headroom it left. Checkpoints are the big
> one: for an architecture whose KV cannot be rolled back partially (`deepseek4`), `llama-server` keeps up
> to `--ctx-checkpoints` (default 32) per-slot KV snapshots on the **host heap**. `--cache-ram` does not
> bound them: it budgets the *archived* prompts in the prompt cache, while the live slot's ring sits outside
> that budget - with `--cache-ram 0` there is no archive at all, yet the ring still fills. `hy_v3` allocates
> none, since its KV supports partial removal. Each snapshot logs its own size (`created context checkpoint
> N of M ... size = X MiB`), so read that rather than guessing: when host RAM is tight, lower
> `--ctx-checkpoints` instead of raising the reserve. Setting it to `0` trades that RAM for a full re-prefill
> whenever the prompt diverges from what is cached.


### Baking the system-prompt KV cache (skip cold-start prefill)

With expert streaming, the correct-quality path runs at `-ub 1` (see the caveats below), so the very first
prompt evaluation of a session is slow: a long, fixed system prompt (agentic tool definitions, injected
rules) is prefilled one token at a time. Measured on an RTX 4090D (IQ2, `-ub 1`), an ~900-token system
prompt takes **~169 s** to prefill from cold. That prefill is pure compute over a *fixed* prefix, so it can
be done once, saved, and restored on every later session for **~0 ms**.

This reuses the stock session/prompt-cache mechanism; no fork-specific flag is needed. The saved file holds
only the tokens and the raw K/V cache bytes - no expert weights and nothing streaming-specific - so it is
independent of where experts live (VRAM/RAM/disk). It must be baked at `-ub 1`: a larger ubatch prefills
with most experts dropped (see caveats) and would persist a low-quality KV.

`llama-completion` (bake once, then load read-only):

```sh
# Format the system turn exactly as the model's chat template does (Hunyuan-v3 shown), so the baked
# tokens match the runtime tokens byte-for-byte; -no-cnv avoids conversation mode pausing for input.
SYS="$(cat system.txt)"
FULL="<|startoftext|>${SYS}<|extra_4|>"

# Bake: prefill the fixed prefix at -ub 1 and write the KV to disk (one-time, slow).
llama-completion -m model.gguf --moe-stream-async -fit off -c 4096 -ub 1 -no-cnv \
  -p "$FULL" -n 1 --prompt-cache sys.kv --prompt-cache-all

# Reuse: load the baked KV; the shared prefix is skipped (prompt eval ~0 ms), only new tokens are evaluated.
llama-completion -m model.gguf --moe-stream-async -fit off -c 4096 -ub 1 -no-cnv \
  -p "${FULL}<|startoftext|>your question here<|extra_0|>" --prompt-cache sys.kv --prompt-cache-ro
```

`llama-server` (per-slot KV save/restore over the HTTP API):

```sh
# Start with a directory for slot state.
llama-server -m model.gguf --moe-stream-async -fit off -c 4096 -ub 1 -np 1 --slot-save-path ./kvcache

# After the first request has prefilled the system prompt into slot 0, persist that slot once:
curl -X POST 'localhost:8080/slots/0?action=save'    -d '{"filename":"sys.kv"}'

# On every later server start, restore the slot to skip the system-prompt prefill; subsequent requests
# reuse the system prefix automatically via the server's longest-common-prefix KV reuse.
curl -X POST 'localhost:8080/slots/0?action=restore' -d '{"filename":"sys.kv"}'
```

Caveats specific to baking:
- Baking skips the *prefill compute* but not the one-time host-RAM expert-pool warm (`LLAMA_MOE_PREFILL`,
  ~7 s), which runs during model load on both the bake and the load path.
- The saved file is rejected on a mismatch of llama.cpp session version, model architecture, KV cache type
  (`--cache-type-k`/`-v`), or context geometry, so re-bake after rebuilding the fork or changing those.
  A different *expert* quantization is not rejected (the KV is independent of it).

### Compacting the KV cache mid-session (evict token ranges)

A long agentic session grows its context to tens of thousands of tokens. The usual fix - summarize the
history into a shorter text and send it as a new prompt - forces a full re-prefill of that summary, which
under expert streaming runs at single-digit tokens/s (tens of minutes for a 20k-token summary). This fork
adds a server endpoint that instead **deletes chosen token-position ranges directly from a slot's KV cache**,
with **zero re-prefill**: the surviving tokens keep their computed K/V, only their positions are compacted
(the engine re-applies RoPE to the shifted keys automatically). No expert weights are touched.

This is not the built-in context-shift (which only drops the *oldest* tokens, FIFO). Here you choose exactly
which ranges to drop, so you can keep the task goal at the start, keep the recent turns at the end, and evict
finished middle work (a file you already read, a stale tool output). Deciding *which* ranges to drop is the
caller's job (e.g. score each past turn with a small model); the server only executes the deletion.

Endpoint (needs `-np 1`; no `--slot-save-path` required, it is a pure in-memory op):

```
POST /slots/{id}?action=evict
body: {"ranges": [[p0,p1], ...]}   # half-open [p0,p1) KV token-position ranges to delete
resp: {"id_slot": 0, "n_evicted": N}
```

Ranges must be within `[0, n_tokens]`, non-overlapping, and each `0 <= p0 < p1`. After the call the slot's
token list and KV are both compacted, so the next request re-uses the surviving prefix with no re-prefill.

Important - what to send on the *next* request: the server matches a new prompt against the slot's kept
tokens by longest common prefix. So the new prompt's leading text must equal the **surviving** token
sequence (the concatenation of the kept ranges, in order), followed by your new turn. Only the appended
new turn is prefilled; the kept span is reused. If the new prompt diverges earlier than the kept length,
everything after the divergence is re-prefilled - so build the next prompt from the same pieces you kept.

End-to-end example (start the server, fill a slot, evict the middle, continue):

```sh
# 1) Start the server. --chat-template chatml sidesteps a PEG-parser error on Hunyuan-v3's built-in template;
#    --slots exposes /slots so you can read a slot's current token count.
llama-server -m model.gguf --moe-stream-async -fit off -c 8192 -np 1 --slots --chat-template chatml &

# 2) Send a prompt that fills the slot's KV (cache_prompt keeps it resident for reuse).
curl -s localhost:8080/completion -d '{"prompt":"<goal at start> ... <lots of middle work> ...","n_predict":1,"cache_prompt":true}'

# 3) See how many tokens the slot holds now.
curl -s localhost:8080/slots | python -c 'import json,sys; print(json.load(sys.stdin)[0]["n_prompt_tokens"])'
#   -> e.g. 1917

# 4) Drop the middle range (keep the head goal + the tail). Positions are token indices from step 3.
curl -s -X POST 'localhost:8080/slots/0?action=evict' -d '{"ranges":[[50,1900]]}'
#   -> {"id_slot":0,"n_evicted":1850}     (slot now holds 67 tokens)

# 5) Continue. Re-send the surviving prefix + your new turn with cache_prompt:true; the kept KV is reused
#    (response "timings.prompt_n" counts only the NEW tokens, proving the evicted span was not re-prefilled).
curl -s localhost:8080/completion -d '{"prompt":"<goal at start> ... <new turn>","n_predict":128,"cache_prompt":true}'
```

Determining the token ranges to pass:
- Track them in your application: as you append each turn, record the `[start,end)` token span it occupies.
  Positions are 0-based, in the order tokens entered the slot, contiguous.
- Or compute a span's length with the `/tokenize` endpoint (POST `{"content":"..."}` returns the token
  array; its length is the token count), and read the slot's current total from `/slots`.
- Use the **model's own tokenizer** for positions. A separate small scoring model can decide *which turns*
  are unimportant, but the byte offsets it sees do not map to the big model's token positions.

Caveats:
- **Text-only.** A multimodal (mtmd) slot is rejected (returns an error, does not crash): image chunks span
  multiple tokens and the token-list rebuild assumes 1 token == 1 position.
- The slot's KV **checkpoints** (position-keyed snapshots used for fast context restore) are dropped by an
  evict, since they would no longer match the compacted cache. Speculative-decoding checkpoints too.
- Evict runs only when the slot is idle; a request in flight defers it (it does not interrupt a decode).
- Quality of the result depends on *what* you evict - keeping the wrong ranges loses information the model
  needed. The mechanism itself preserves the retained tokens' K/V exactly (verified: dropping ~880 filler
  tokens between a head instruction and the tail, the model still recalled the head's pass key).

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
