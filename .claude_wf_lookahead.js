export const meta = {
  name: 'moe-lookahead-loader-prefetch-design',
  description: 'Design and validate wiring the existing one-layer lookahead routing probe into the background loader as a RAM-tier prefetch, to cut the decode callback disk reads (the dominant ~200ms/token cost) without quality risk',
  phases: [
    { title: 'Investigate' },
    { title: 'Synthesize' },
  ],
}

const CTX = [
"Repo E:\\\\Coding\\\\CatLllamaCpp (llama.cpp fork), hy3 Q4 170GB, 192 experts/8 used/80 MoE layers, 4090D 24GB + 64GB RAM, Windows/CUDA.",
"",
"CURRENT STATE (measured, correct-quality decode path = LLAMA_MOE_SYNC_BUDGET=2 + LLAMA_MOE_ASYNC + NOMMAP + CACHE_CAP=16 + RAM_CAP=64):",
"- decode 3.11 tok/s (321ms/token), coherent quality through long generations.",
"- Per-token cost breakdown (corrected): callback's OWN synchronous disk+H2D of the budget=2 top-2 misses ~170-200ms = DOMINANT; GPU compute de-fused by 80 splits ~36ms extra; lock-wait residual ~38ms; base compute ~19ms.",
"- The callback (llama_moe_layer_remap_cb, a ggml map_custom1 CPU op, one per MoE layer on decode) runs AFTER that layer's router produced selected_experts; for the true top-2 experts that are NOT resident it fread+H2D them from disk/RAM BEFORE the layer's expert matmul (correctness: this token's real top-2 must be resident before its matmul). budget=2 + stale-reuse for the other 6 positions gives correct quality.",
"- Expert source tiers (llama_moe_read_expert / llama_moe_layer_src): locked RAM pool (LLAMA_MOE_RAM_CAP experts/layer, ~25GB/s) -> no-mmap file fread (disk, ~ms) -> mmap. A callback top-2 miss that IS in the RAM pool is fast (~0.5ms); one that must hit disk is ~4ms. Measured top-2 residency (VRAM) ~46-74%; the rest miss to RAM-or-disk.",
"- Background loader (llama_moe_loader_main): polls each layer's published sel_buf every 100us, weight-aware-evicts a hot core into the VRAM cache, and fills the locked RAM pool (llama_moe_loader_fill_ram) toward highest expert_score. Just refactored to do its disk reads OUTSIDE g_moe_layer_mutex (3-phase: collect under lock -> fread unlocked -> H2D+publish under lock).",
"",
"THE ALREADY-BUILT LOOKAHEAD PROBE (not yet wired to anything): src/models/hy-v3.cpp ~line 156, gated by LLAMA_MOE_LOOKAHEAD_PROBE, computes a PURE-GPU prediction of layer il+1's routing by running layer il+1's gate (model.layers[il+1].ffn_gate_inp) on layer il's current normalized hidden 'cur', then sigmoid + exp_probs_b + argsort_top_k, tagged 'ffn_moe_topk_pred'. Currently a DEAD subgraph (ggml_build_forward_expand keeps it but nothing consumes it). MEASURED accuracy (real hidden, one layer ahead): full-8 overlap ~40%, top-2 exact ~18%, vs correct next-layer routing. (Cross-token / stale-full-prepass prediction was ~7% and is dead; this one-layer-ahead-from-real-hidden 40% is the survivor.)",
"",
"THE IDEA TO DESIGN: wire the lookahead prediction into the loader as a PREFETCH HINT (NOT into compute - compute still uses the true router, so zero quality risk, mispredict just wastes a background read). While layer il computes, the predicted top-K experts for layer il+1 are published to a host-visible buffer; the loader reads it and prioritizes prefetching those experts into the RAM pool (and/or VRAM cache) so that when layer il+1's callback runs ~microseconds later, more of its true top-2 misses are already RAM-resident (0.5ms) instead of disk (4ms). Goal: cut the ~200ms/token callback disk read.",
"",
"KEY UNCERTAINTY to resolve: the RAM pool is ALREADY filled by the loader toward popularity (expert_score). Does next-layer-specific lookahead prefetch add RAM hits BEYOND what the popularity-based pool already captures? The relevant metric is NOT the 18% top-2 exact-match; it is: of the callback's true top-2 misses that currently hit DISK, how many would a lookahead prefetch (predicted top-K for K=8 or larger, issued one layer ahead) have placed in RAM in time? If the popularity pool already holds them, lookahead adds nothing; if the disk-missing top-2 are exactly the churny non-popular ones lookahead can foresee, it helps a lot.",
].join("\n")

const SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string' },
          evidence: { type: 'string' },
          confidence: { type: 'string', enum: ['high','medium','low'] },
        },
        required: ['claim','evidence','confidence'],
      },
    },
  },
  required: ['summary','findings'],
}

phase('Investigate')

const [timing, wiring, measure] = await parallel([
  () => agent(CTX + "\n\nYOUR TASK (TIMING FEASIBILITY - is there time to prefetch il+1 during il?): Read src/models/hy-v3.cpp layer loop and src/llama-moe-stream.cpp loader (100us poll). Answer:\n1. The lookahead for il+1 is computed on the GPU during il's graph. But the loader reads published buffers on a 100us poll and its RAM fill (fill_ram) does disk reads. Between when layer il's lookahead prediction is available (host-visible) and when layer il+1's callback needs those experts, how much wall-time is there? On decode, consecutive layers il and il+1 are ~4ms apart (321ms/80). Is 4ms enough for the loader to fread a predicted expert (~2-4ms disk) into RAM before il+1's callback? Reason about whether one-layer-ahead is enough lead time, or whether we need to predict il+2/il+3 (deeper lookahead) to have time - and that deeper prediction is less accurate.\n2. Is the prefetch better targeted at the RAM pool (fill_ram, so the callback's fread hits RAM) or directly at the VRAM cache (so the expert is already resident and the callback does NOTHING)? The latter would also eliminate the split for that layer if ALL top-2 became resident. Assess both.\n3. The loader currently polls all 80 layers' sel_buf each 100us pass. If it also processes 80 lookahead-prediction buffers, does the extra work slow its cycle? Quantify the added per-pass cost.\nReturn structured findings.", { label: 'timing', phase: 'Investigate', schema: SCHEMA }),

  () => agent(CTX + "\n\nYOUR TASK (WIRING - concrete implementation path): Design the minimal code to publish the lookahead prediction and consume it in the loader. Read src/models/hy-v3.cpp (the probe ~156, and llama_moe_layer_cache_publish call ~1594 in llama-graph.cpp), src/llama-moe-stream.cpp (sel_buf allocation, llama_moe_layer_cache_publish, loader_main, fill_ram, the layer cache struct).\n1. How to publish the predicted ids: add a pred_sel_buf device tensor to the layer cache (mirror sel_buf), and a publish call analogous to llama_moe_layer_cache_publish that copies the ffn_moe_topk_pred tensor into it via ggml_cpy. But the prediction for il+1 is computed in il's graph - which layer's cache does it belong to? It predicts il+1's routing, so it should land in il+1's cache's pred buffer. Work out the cache-keying: at build time for layer il, is layer il+1's cache object available (created lazily)? Cite llama_moe_layer_cache_get and the creation timing.\n2. How the loader consumes it: in loader_main or fill_ram, read each cache's pred_sel_buf (host-visible), and prioritize those experts in the RAM fill (before/above the popularity-based fill). Sketch the change to fill_ram's target-selection (currently picks highest expert_score not-yet-in-RAM; add: prefer pred-listed experts first).\n3. Correctness: the prediction only drives prefetch, never compute (compute uses the true selected_experts through slot_table/stale_table). Confirm no path lets a predicted id reach the matmul. Confirm the extra GPU op (the lookahead gate matmul+argsort per layer) does NOT itself open a scheduler split (it is pure GPU - get_rows/argsort/mul_mat are CUDA ops), so it does not add split tax.\n4. Cost of the lookahead GPU op itself: an extra [n_embd x n_expert] matmul + sigmoid + argsort per layer per token. Quantify vs the 19ms base compute - is it negligible?\nReturn structured findings.", { label: 'wiring', phase: 'Investigate', schema: SCHEMA }),

  () => agent(CTX + "\n\nYOUR TASK (THE DECISIVE CHEAP MEASUREMENT - before building the full prefetch): The key uncertainty is whether lookahead prefetch adds RAM hits BEYOND the popularity-based RAM pool. Design the cheapest measurement to answer this WITHOUT building the full loader wiring. Read src/llama-moe-stream.cpp (the callback's load path, llama_moe_read_expert, the g_diag_ram_hit counter that already exists but may be unused, the RAM pool ram_slot).\n1. Propose adding a per-token diagnostic in the CALLBACK's sync-load: when it loads a true top-2 miss, count (a) how many were RAM-resident (fast) vs disk (slow), i.e. actually populate the existing g_diag_ram_hit. This tells us the CURRENT RAM-hit rate of the callback's misses under the popularity pool alone. If it's already high (say >80%), lookahead can't help much; if low (<50%), there's room. Cite where in the callback to add it (the load_e loop / moe_layer_load_into_slot).\n2. Propose a SECOND cheap measurement: instrument the existing dead lookahead probe to, per layer per token, compare its predicted top-K against the NEXT token's actual sel_buf for that same layer AND against what's currently RAM-resident, to estimate 'of the callback's disk-missing top-2, how many did lookahead foresee AND were not already in RAM'. This is the true upside. Can this be computed with the existing ffn_moe_topk_pred dump + a RAM-residency snapshot, offline from logs?\n3. Give the go/no-go thresholds: what callback RAM-hit rate and what 'lookahead-foreseeable disk-miss fraction' would justify building the full prefetch wiring vs dropping it.\nReturn structured findings.", { label: 'measure', phase: 'Investigate', schema: SCHEMA }),
])

phase('Synthesize')

const VERDICT = {
  type: 'object',
  properties: {
    is_it_worth_building: { type: 'string', description: 'Direct verdict: does lookahead-prefetch-into-loader have a real shot at cutting the ~200ms callback disk read, or is it likely subsumed by the popularity RAM pool?' },
    timing_verdict: { type: 'string', description: 'Is one-layer lead time (~4ms) enough for the loader to prefetch, or is deeper (less accurate) lookahead needed?' },
    decisive_measurement: { type: 'string', description: 'The single cheapest measurement to run FIRST (ideally just populating g_diag_ram_hit + a log-diff), with exact go/no-go thresholds.' },
    build_plan: { type: 'string', description: 'If the measurement passes: the concrete minimal wiring (pred buffer, publish, loader consume), files/functions.' },
    quality_safety: { type: 'string', description: 'Confirm the prefetch-only design cannot affect output (prediction never reaches compute).' },
    recommendation: { type: 'string', description: 'measure-first / build / drop, honest, given decode is at 3.11 and Pulsar-parity ~5.3 would need the callback disk read roughly halved.' },
  },
  required: ['is_it_worth_building','timing_verdict','decisive_measurement','build_plan','quality_safety','recommendation'],
}

const verdict = await agent(CTX + "\n\nThree investigators examined wiring the lookahead probe into the loader as a RAM prefetch. Findings:\n\n## TIMING FEASIBILITY\n" + JSON.stringify(timing, null, 2) + "\n\n## WIRING\n" + JSON.stringify(wiring, null, 2) + "\n\n## DECISIVE MEASUREMENT\n" + JSON.stringify(measure, null, 2) + "\n\nYOUR TASK (SYNTHESIZE): The dominant decode cost is the callback's ~200ms/token synchronous disk read of budget=2 top-2 misses. The proposed lever: use the already-built one-layer lookahead (40% full-8 accuracy) to prefetch next-layer experts into the RAM pool so the callback hits RAM not disk. Decide honestly whether this is worth building, gate it behind the cheapest possible measurement (the key unknown = does lookahead add RAM hits beyond the popularity pool already there), and give the go/no-go thresholds + minimal build plan. We have shipped a real 1.7x loader-lock fix and a 2.2x prefill win this session; do not oversell a marginal gain. Ground everything. Return the structured verdict.", { label: 'synthesis', phase: 'Synthesize', schema: VERDICT })

return { timing, wiring, measure, verdict }
