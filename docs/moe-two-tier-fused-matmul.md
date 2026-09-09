# Two-tier fused `mul_mat_id` (mixed-quantization expert dispatch)

Fork-only design note. Covers a new CUDA op that computes ONE `mul_mat_id` whose expert slabs are drawn
from two different quantization types in a single dispatch, and what that op buys on each of the two
paths it touches (prefill expert-group sweep, two-tier decode).

Status: design. Nothing here is implemented yet. Every number below is arithmetic derived from the code
structure and from per-expert byte sizes read out of the GGUFs - none of it is a measured speedup.

## 1. Why

The binding axis is prefill, not decode. Single-model streaming prefill already runs at tens to low
hundreds of tokens/s; a mixed-precision mode that trades quality for decode speed but makes prefill
slower moves the whole product into a range that is not usable interactively. So the target for the
two-tier mode is not "prefill degrades only a little" - it is **prefill strictly faster than the
single-model baseline at the same VRAM budget**.

Today the two-tier mode misses that target for one structural reason, and the reason is not a tuning
problem:

- `src/llama-graph.cpp` computes the prefill sweep's pass count from the HIGH tier's capacity alone:
  `n_groups = ceil(n_expert / cap_hi)`. The low tier's slots are invisible to it.
- `src/llama-graph.cpp` computes two-tier decode as two full FFN chains summed under a 0/1 mask,
  because (quoting the current comment) "a ggml tensor has one type and one expert stride, so this
  two-pass sum is the only way to express mixed precision at all".

Both statements are true of `ggml_mul_mat_id` as it exists. Neither is true of the hardware. A GEMM
does not care that two groups of its output columns read weight blocks in different formats; only
ggml's one-type-per-tensor API and MMQ's compile-time type template make it look that way. The op
below removes the API restriction without touching the kernels' inner loops.

## 2. What the current code costs

### 2.1 Prefill sweep

`src/llama-graph.cpp`, the `moe_prefill_sweep` block: one full-width FFN pass per static expert-index
group, `n_groups` of them, summed. Within one ubatch step the expert bytes read are invariant in
`n_groups` (each group loads only its own in-group selected experts, and the groups are disjoint).
What scales linearly with `n_groups`:

- GPU: the `dst` memset in `mmq.cu`, the src1 q8_1 quantization walking all `ne11_flat` entries, the
  SwiGLU / clamp / weight-mul chain, and the `ggml_add` accumulation.
- CPU: the per-(layer, group) flatten / need-scan / dst-write loops in
  `llama_moe_layer_remap_group_cb`, one scheduler split, one device sync, one `g_moe_layer_mutex`
  acquisition, and `parallel_load`'s per-call fixed overhead.

Expert GEMM tiles do NOT scale: the sentinel-skip path collapses `n_real` to the capacity, so out-of-
group positions cost nothing.

### 2.2 Two-tier decode

`src/llama-graph.cpp`, the `moe_ids_lo && moe_mask` block. Per MoE layer this emits, instead of one
FFN chain: two chains (6 `mul_mat_id` instead of 3 where gate/up are separate), two SwiGLU chains, a
`ggml_mul` and a `ggml_sub` to split the router weights by the mask, and a `ggml_add` to sum.

The cost that matters most is not the node count. It is that **the pass which does not own a routed
position still computes that position's full expert GEMM**, at a hard-zero weight. The current comment
says so explicitly. So decode evaluates `n_used` high-tier experts AND `n_used` low-tier experts per
layer, when only `n_used` experts total are actually contributing.

## 3. The three code facts that make the op cheap to build

Read out of the CUDA backend, not assumed:

1. **`ggml_cuda_mul_mat_q`'s `mul_mat_id` branch is four separable linear steps**
   (`ggml/src/ggml-cuda/mmq.cu`): optional `dst` memset, `mm_ids_helper` producing
   `ids_src1` / `ids_dst` / `expert_bounds`, one `quantize_mmq_q8_1_cuda` of src1, then one
   `ggml_cuda_mul_mat_q_switch_type` GEMM launch. Only the last step is type-specialized.

2. **`expert_bounds` is a prefix-sum over expert channels.** The MMQ kernel reads
   `col_low = expert_bounds[zt]`, `col_high = expert_bounds[zt + 1]` and early-outs on
   `col_diff == 0`. Both the src1 offset (`offset_y`) and the dst indirection
   (`ids_dst[col_low + ...]`) are ABSOLUTE column numbers into the shared compacted buffer. This is
   the load-bearing property: a launch can be handed a SLICE of `expert_bounds` and it will read the
   correct sub-range of an unmodified, shared `ids_dst` / q8_1 buffer.

3. **Q4_K and IQ1_S share `MMQ_Q8_1_DS_LAYOUT_DS4`** (`ggml/src/ggml-cuda/mmq.cuh`), and
   `quantize_mmq_q8_1_cuda` uses `type_src0` for nothing except selecting that layout
   (`ggml/src/ggml-cuda/quantize.cu`). So for this model pair the quantized src1 buffer the high tier
   needs is byte-identical to the one the low tier needs. The single most expensive shared setup step
   is naturally shareable, with no new kernel and no new layout logic.

On the decode path the equivalent fact is even simpler: `quantize_row_q8_1_cuda` asserts `!ids` and
quantizes the whole of src1 independently of routing, so MMVQ's src1 buffer is trivially shared.

## 4. The op

### 4.1 Signature

A new fork-private op, following the `GGML_OP_MOE_FFN` / `GGML_OP_HC_SINKHORN` precedent rather than
extending upstream `GGML_OP_MUL_MAT_ID` (which would perturb every backend's `supports_op` and the
CPU forward path):

```c
    // fork: one mul_mat_id whose expert slabs live in TWO tensors of different quantization types.
    // Each routed position is computed exactly once, against whichever tier `tier` assigns it.
    GGML_API struct ggml_tensor * ggml_mul_mat_id_2t(
            struct ggml_context * ctx,
            struct ggml_tensor  * as_hi,  // [n_embd, n_ff, n_slots_hi] high-precision expert slabs
            struct ggml_tensor  * as_lo,  // [n_embd, n_ff, n_slots_lo] low-precision expert slabs
            struct ggml_tensor  * b,      // [n_embd, n_used, n_tokens] input activations
            struct ggml_tensor  * ids,    // [n_used, n_tokens] i32, slot id within the OWNING tier
          struct ggml_tensor  * tier);  // [n_used, n_tokens] i32, 1 = as_hi owns it, 0 = as_lo
```

Result shape and type are exactly those of `ggml_mul_mat_id(ctx, as_hi, b, ids)`:
`[n_ff, n_used, n_tokens]` f32. `as_lo` must match `as_hi` in `ne[0]` and `ne[1]`; `ne[2]` (slot
count) may differ, and the types must differ or the caller should have used plain `mul_mat_id`.

`op_params[0] = n_real_hi`, `op_params[1] = n_real_lo` carry the existing sentinel-skip contract
per tier (channels at or above `n_real_*` in that tier are permanently-zero sentinel slabs).

Registration checklist, mirroring the two existing fork ops: enum in `ggml/include/ggml.h`, name in
`ggml/src/ggml.c`'s op-name table, constructor in `ggml/src/ggml.c`, abort stubs in
`ggml/src/ggml-cpu/ggml-cpu.c` (two sites) and `ggml/src/ggml-cpu/ggml-cpu.cpp`, compute dispatch and
`supports_op` in `ggml/src/ggml-cuda/ggml-cuda.cu`. `GGML_MAX_SRC` is 10, so five sources is fine.

### 4.2 Unified channel space

The whole implementation rests on one representational choice. Concatenate the two tiers' expert
channels into a single index space of width `n_slots_hi + n_slots_lo`: channel `c` means high-tier
slot `c` when `c < n_slots_hi`, and low-tier slot `c - n_slots_hi` otherwise. The graph-side callback
emits `ids` already biased into this space, so `tier` is redundant on the CUDA side and exists only to
keep the op self-describing and to let a CPU/other-backend fallback exist later.

### 4.3 CUDA, batched path (MMQ) - the prefill sweep case

1. `cudaMemsetAsync(dst)` once, as today, when either tier declares sentinels.
2. **One** `mm_ids_helper` launch over `n_slots_hi + n_slots_lo` channels, producing one `ids_src1`,
   one `ids_dst`, and one `expert_bounds` of length `n_slots_hi + n_slots_lo + 1`. The helper needs a
   segmented `n_real` (see 4.5) but is otherwise unchanged.
3. **One** `quantize_mmq_q8_1_cuda`, valid for both tiers when
   `mmq_get_q8_1_ds_layout(hi) == mmq_get_q8_1_ds_layout(lo)`. If the layouts differ, fall back to two
   quantizations into two buffers - correct, just no longer free. (unsloth "UD" dynamic quants vary the
   type per tensor, so this check must be per call, not per model.)
4. **Two** `ggml_cuda_mul_mat_q_switch_type` launches sharing steps 1-3:
   - high: `x = as_hi->data`, `type_x = as_hi->type`, `expert_bounds` as-is,
     `nchannels_x = nchannels_y = n_slots_hi`.
   - low: `x = as_lo->data`, `type_x = as_lo->type`, `expert_bounds + n_slots_hi`,
     `nchannels_x = nchannels_y = n_slots_lo`.

   Both launches receive the same `ids_dst`, the same q8_1 buffer, and the same `dst`. Because
   `expert_bounds` is a prefix sum and the kernel indexes src1 and dst by absolute column, the pointer
   offset is the entire adaptation. No kernel source changes.

Two launches instead of one is the only duplicated GPU work, and it duplicates launch overhead, not
arithmetic: each launch's tiles cover a disjoint set of output columns.

### 4.4 CUDA, vector path (MMVQ) - the decode case

Decode has `n_tokens == 1`, so `ne2 <= MMVQ_MAX_BATCH_SIZE` and `ggml_cuda_mul_mat_id` takes the MMVQ
branch, which has no compaction, no `expert_bounds`, and a routing-independent src1 quantization.

Here the shared setup is free, but skipping the other tier's columns needs a kernel change, because
MMVQ's grid assigns one block per `channel_dst` and reads `channel_x = ids[channel_dst]` directly.
Add an optional tier pointer and a compile-time flag to `mul_mat_vec_q`:

```c
    if constexpr (has_tier) {
        if (tier[channel_dst] != want_tier) {
       return;
        }
    }
```

`has_tier == false` (every existing caller) compiles the branch away entirely. Then issue the same two
launches over the same q8_1 buffer, one per tier, each early-outing on the channels it does not own.

### 4.5 Segmented sentinel skip

> **M2 audit result: this kernel change is UNNECESSARY and was NOT implemented.** The shipped M2 op
> runs ONE `mm_ids_helper` over the full unified space with `n_real == ne02_u` (scan everything, no
> skip), producing a genuine global prefix sum. The per-tier sentinel bounds `n_real_hi` / `n_real_lo`
> are applied instead at the two `mul_mat_q` launches via `nchannels` (the high launch iterates
> `[0, n_real_hi)`, the low launch's sliced `expert_bounds + n_slots_hi` iterates `[0, n_real_lo)`).
> Sentinel slots sit at the tail of each tier's range, so a reduced `nchannels` drops exactly their
> tiles; dropped positions' dst rows fall back to the pre-zeroed dst. This is byte-identical to the
> masked-sum reference (verified in `tests/test-mul-mat-id-2t.cpp`) with zero kernel edits. The
> original segmented-helper sketch below is kept only as the design record.

`mm_ids_helper`'s existing `n_real` says "channels at or above this index hold no rows". In the
unified space there are two such thresholds. The helper's `skip` / `cmp` computation becomes:

```c
    const bool is_lo = expert >= n_slots_hi;
    const int  local = is_lo ? expert - n_slots_hi : expert;
    const int  nreal = is_lo ? n_real_lo : n_real_hi;
    const bool skip  = local >= nreal;
```

`cmp` (the "count positions strictly below this" bound used to build the prefix sum) must become the
global channel id of the owning segment's real upper bound, i.e. `n_real_hi` for a skipped high
channel and `n_slots_hi + n_real_lo` for a skipped low one. Passing `n_slots_hi == n_experts` and
`n_real_lo == 0` reproduces today's behaviour exactly, which is the regression test for this change.

### 4.6 Graph side

- `llama_moe_tiered_pack_cb` (`src/llama-moe-stream.cpp`) already emits three stacked rows in ONE CPU
  op: high ids, low ids, and the 0/1 tier selector. It collapses to two rows - one unified id row
  (low ids biased by `n_slots_hi`) and the tier row - so the op costs no new CPU work and one less
  row of writes.
- The decode block in `src/llama-graph.cpp` collapses from two `build_experts` calls plus
  `ggml_mul` / `ggml_sub` / `ggml_add` to a single `build_experts` call. `moe_mask`,
  `llama_moe_layer_cache_tier_weights`, and the `tier_w` get_rows disappear from the graph.
- The sweep block's `n_groups` becomes `ceil(n_expert / (cap_hi + cap_lo))`. The `dep` serialization
  edge between groups must be preserved unchanged - it is what keeps group `g+1`'s slot recycling from
  racing group `g`'s in-flight matmuls.
- `mm_id_exps` grows a branch that emits `ggml_mul_mat_id_2t` when a low cache is present and both
  tiers resolve to device slabs, and otherwise emits exactly what it emits today.

## 5. What it buys

### 5.1 Prefill, on a 24 GiB card (slab budget ~271.6 MiB/layer)

Per-expert bytes: high Q4_K_XL 3.060 MiB, low IQ1_S 1.546 MiB.

| configuration | slots | sweep passes | expert bytes/layer/step |
|---|---|---|---|
| single-model Q4 baseline | 88 hi | 6 | 1567 MiB |
| two-tier today (high-only tiling) | 64 hi + 49 lo | 8 | 1567 MiB |
| two-tier fused | 64 hi + 49 lo | **5** | **1270 MiB** |

Pass count: `ceil(512 / 113) = 5`. Bytes: with per-group tier assignment by expert score, four full
groups contribute 64 high + 49 low and the last (60 experts) fits entirely in high slots, giving
316 high + 196 low = `316*3.060 + 196*1.546` = 1270 MiB.

Note this corrects an earlier estimate of 888 MiB, which wrongly assumed only the globally hottest 64
experts would ever be read at high precision. Tier assignment is per group, so the high share is
`cap_hi / (cap_hi + cap_lo)` = 57%, not `cap_hi / n_expert` = 12.5%.

Both axes beat the single-model baseline: 5 passes vs 6, and 1270 MiB vs 1567 MiB. Weighting by the
~77% of prompt-eval wall the README attributes to the expert loader gives a paper figure of about
1.2x against the single-model baseline and about 1.3x against two-tier today. These are arithmetic,
not measurements, and the loader share is itself a previously measured average - treat both as an
order-of-magnitude claim about direction, not a predicted number.

On a 5090 (slab ~364 MiB/layer) the same arithmetic gives `cap_hi 103 + cap_lo 87 = 190` slots ->
3 passes, against that card's single-model baseline of `364/3.060 = 119` -> 5 passes. The advantage
widens with VRAM, which is the behaviour the current defaults fail to deliver.

### 5.2 Decode

Honest accounting, worst part first:

- **Expert GEMM roughly halves.** Today both passes compute every routed position. With the high tier
  serving ~55-58% of positions (per `LLAMA_MOE_TIERDBG`) and low-tier arithmetic costing about half of
  high-tier per position, work goes from `n_used * 1 + n_used * 0.5` to about
  `0.57 * n_used * 1 + 0.43 * n_used * 0.5`, i.e. to ~0.52x.
- **Node count roughly halves**, from about 20 MoE nodes per layer to about 10. At the fork's
  previously measured 0.77-0.94 us per ggml node, 48 layers x ~10 nodes is ~0.4 ms/token.
- **Expert bytes loaded do not change.** Each routed position already reads from exactly one tier.
- **Split count does not change.** Both passes are GPU-side; the CPU splits come from the remap
  callback, which is already one op per layer.

Decode is load-bound and split-bound, and this op touches neither. Against a ~56 ms/token steady
state, the node saving alone is under 1%; the GEMM halving is worth more but expert GEMM is a minority
of decode wall. **Expect single-digit percent.** The op is worth building for prefill; decode
improvement is a real but secondary benefit, and the graph simplification (deleting the mask/sub/add
machinery and one of the two `tier_w` lookups) is arguably worth as much as the speed.

## 6. Correctness

- **Decode becomes strictly MORE correct, not less.** Today every position is computed twice and one
  copy is zeroed by the mask. Fused, it is computed once against the same tier the mask would have
  selected. Same weights, same arithmetic, one fewer rounding of a zero-weighted term.
- **Prefill loses its lossless guarantee, deliberately.** Today the sweep runs entirely on the high
  tier, so prefill is bit-exact against a non-streamed run and the two-tier mode's quality cost is
  confined to generation. Fused, ~38% of expert positions in prefill are evaluated at IQ1_S. This is
  the trade the mixed-precision mode is for, but it is a behaviour change and must be gated by an
  env knob that restores high-only sweep tiling.
- **This invalidates the existing quality harness for this change.** The KLD harness requires `-ub 1`,
  which forces every position onto the decode path, so the sweep never runs and the resulting table
  contains zero prefill information. Measuring the prefill quality cost needs a different instrument -
  the natural one is a KLD run at the production `-ub` with `-b == -c`, compared against the same
  `-ub` on the single-model baseline. That harness does not exist yet and building it is a
  prerequisite for defaulting this on.
- The sentinel contract is preserved per tier and is checked by the `n_slots_hi == n_experts,
  n_real_lo == 0` degenerate case reproducing current behaviour bit-for-bit.
- `ggml_cuda_should_use_mmq` and the MMVQ batch-size gate are evaluated on `src0->type`. The two tiers
  can disagree. The op must decide once, from the high tier, and refuse (falling back to today's two
  passes) if the low tier's type cannot take the same path.

## 7. Staging

1. **M1 - op skeleton, high tier only.** Register the op; implement it as "ignore `as_lo`, behave
   exactly like `mul_mat_id` on `as_hi`". Wire nothing. Verifies the registration checklist and gives
   a bit-identical baseline to diff against.
2. **M2 - MMQ two-launch. [DONE]** Unified channel space, ONE `mm_ids_helper` (full-scan `n_real`,
   per-tier sentinel bounds applied at the launches via `nchannels` - see the 4.5 audit note; the
   segmented-helper kernel change was found unnecessary), shared q8_1 and `expert_bounds` slicing.
   Verified by `tests/test-mul-mat-id-2t.cpp`: fused output is byte-identical to the two-pass
   masked-sum reference (Q4_K high + IQ1_S low, 4096 elements, on CUDA).
3. **M3 - decode wiring.** Collapse the two-pass block, add the MMVQ tier early-out. Verifiable
   without any timing: with `LLAMA_MOE_NOLOADER=1`, `--temp 0` and a fixed seed, the fused decode must
   produce a completion identical to the two-pass decode, since section 6 argues the two are the same
   arithmetic.
4. **M4 - sweep wiring.** `n_groups` from the combined capacity, per-group tier assignment by expert
   score, env knob to restore high-only tiling. Not defaultable until the prefill quality harness of
   section 6 exists.
5. **M5 - capacity policy.** `llama_moe_auto_capacity`'s snap-to-group-boundary logic currently snaps
`cap_hi` alone against `ceil(n_expert / g)`. It must snap `cap_hi + cap_lo` instead, which also
   removes the `cap_lo >= n_used` guard that the measured cap_lo cliff (~59 slots, not 10) shows is
   wrong by roughly 5x.

Each stage after M1 has a bit-identity test that needs no benchmark, which is the point of the
ordering: everything except M4's quality question can be settled by MD5, not by timing.

## 8. Risks

- **UD dynamic quants break the shared-quantization assumption per tensor, not per model.** The
  layout check must be per call with a two-buffer fallback, or a single layer whose high tensor is
  Q5_K and low is IQ1_S will silently read mis-scaled src1.
- **Two launches can lose to one when the per-launch fixed cost dominates**, i.e. at small
  `n_slots_lo`. If `cap_lo` is small enough that the low launch covers a handful of columns, emitting
  it at all may cost more than folding those positions into the high tier. Worth a threshold.
- **`GGML_OP_MOE_FFN` is the cautionary precedent**: it removed real work and still lost ~20% end to
  end, because fusing the whole FFN serialized the expert load against compute and destroyed an
  implicit overlap. This op is deliberately narrower - it fuses one matmul, not the chain - but the
  same failure mode (an op boundary that removes a scheduling seam) is the thing to watch for on the
  decode path, and it is why M3 is separable from M4.
- CUDA graph capture: `GGML_OP_MOE_FFN` disables CUDA graphs. This op should not need to, since it
  issues only kernel launches with build-time-fixed parameters, but that must be confirmed rather
  than assumed.
