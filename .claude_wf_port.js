export const meta = {
  name: 'pulsar-port-two-questions',
  description: 'Answer two questions at code level: (1) why not port Pulsar per-kernel launch (esp. MoE-section-only), (2) why have we ported NONE of Pulsar latency-hiding, and can we port its CopyStream/prefetch overlap into our ggml callback',
  phases: [
    { title: 'Investigate' },
    { title: 'Synthesize' },
  ],
}

const CTX = [
"TWO codebases. PULSAR E:\\\\Coding\\\\pulsar (Rust/CUDA, the FAST engine, hy3 5.3 tok/s decode on 16GB+32GB IQ2, correct quality). CatLlamaCpp E:\\\\Coding\\\\CatLllamaCpp (llama.cpp fork, hy3 Q4 170GB, our engine, decode ~3.1 tok/s correct-quality on 24GB+64GB).",
"",
"USER'S TWO QUESTIONS (answer at CODE LEVEL, no hand-waving, no 'structurally hard'):",
"Q1. Pulsar decode is a straight sequence of hand-launched CUDA kernels (no fused graph). Why can't we port that - specifically, why can't we drop to manual per-kernel launch for JUST the MoE expert section of each hy3 layer while keeping the rest on ggml? Give the concrete blocker with file/function names and the exact cost.",
"Q2. Pulsar has latency-hiding (cross-layer prefetch on a background thread + a CopyStream side-stream that overlaps H2D with default-stream compute). We have ported NONE of it - our expert load is always synchronous-blocking inside the map_custom callback. Why not, and CAN we port the CopyStream-overlap and/or the cross-layer prefetch into our design? Give the concrete path or the concrete blocker.",
"",
"OUR CURRENT DECODE (correct-quality path, LLAMA_MOE_SYNC_BUDGET=2 + ASYNC + NOMMAP + CACHE_CAP=16 + RAM_CAP=64 + RAM_FILL high):",
"- decode ~3.1 tok/s. Per token: the per-MoE-layer map_custom1 CPU callback (llama_moe_layer_remap_cb in src/llama-moe-stream.cpp) synchronously loads this layer's true top-2 misses (fread + H2D) BEFORE returning, because the downstream ggml_mul_mat_id reads those expert slots right after. ~48 experts/token synced, ~65% RAM-hit / 35% disk.",
"- We DID ship this session: (a) loader background thread does its RAM-fill disk reads outside the global mutex; (b) pinned host staging buffers; (c) async-H2D batching of the RAM/mmap-direct copies (set_async + one synchronize) - gave +9%. But the CALLBACK's load is still on the critical path, fully blocking the GPU: the GPU sits idle during the callback's fread+H2D because the matmul cannot start until experts are resident.",
"- The fundamental structure: ggml builds ONE compute graph/token; ggml_backend_sched splits it at each CPU map_custom op; a CPU split forces ggml_backend_synchronize (full GPU drain, ggml-cuda.cu ~3290) so the GPU is idle while the callback runs. So there is NO compute running to overlap the load against, WITHIN the current layer.",
"",
"PULSAR MECHANISMS (found in crates/kernels/src/lib.rs, crates/engine/src/lib.rs, crates/stream/src/lib.rs):",
"- CopyStream (kernels/lib.rs:525-536+): a side CUDA stream + event; cudaMemcpyAsync from PINNED host memory overlaps default-stream kernels; cudaStreamWaitEvent to order. Comment: 'copy_async from PINNED sources overlaps default-stream kernels'.",
"- Cross-layer prefetch (engine/lib.rs, PULSAR_NO_PREFETCH): a background thread runs layer N+1's router on layer N's hidden and prefetches predicted experts while layer N computes. Measured +21% (1.72->2.08) on the C predecessor.",
"- Pulsar decode is hand-launched kernels on the default stream (NO cudaGraph), so its per-layer host readback + demand load is its baseline, and it CAN interleave a side-stream H2D with the next kernels because there is no graph barrier.",
"",
"KEY ASYMMETRY to analyze: in Pulsar, because there is no fused graph, after layer N's router the engine can (i) launch layer N's attention/other kernels on the default stream AND (ii) simultaneously H2D the experts on the CopyStream, so load overlaps compute. In ggml, the map_custom CPU op forces a full sync so nothing overlaps. The question is whether we can create overlap WITHOUT going fully manual - e.g. by moving the expert load OUT of the map_custom callback and onto a CopyStream that runs during the layer's OWN attention/norm compute (which happens BEFORE the router selects experts... but the experts needed depend on the router output, chicken-and-egg), OR during the PREVIOUS layer's compute via cross-layer prefetch.",
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
          evidence: { type: 'string', description: 'exact file:line / function names in either repo' },
          confidence: { type: 'string', enum: ['high','medium','low'] },
        },
        required: ['claim','evidence','confidence'],
      },
    },
  },
  required: ['summary','findings'],
}

phase('Investigate')

const [q1, q2copy, q2prefetch] = await parallel([
  () => agent(CTX + "\n\nYOUR TASK (Q1 - manual per-kernel launch for the MoE section only): Answer at code level whether we can hand-launch just the MoE expert matmuls outside ggml. Read CatLlamaCpp src/llama-graph.cpp build_moe_ffn (the mm_id_exps lambda, ggml_mul_mat_id calls ~1809/1833), src/models/hy-v3.cpp, ggml/src/ggml-cuda/ggml-cuda.cu (is ggml_cuda_mul_mat_id callable standalone? what state does it need?), and how the layer's hidden-state tensors are allocated (ggml-managed device buffers) and whether raw CUDA can read/write them.\n1. Concretely: at layer il, after the router produces selected_experts (a ggml tensor), could we skip building the ggml mul_mat_id nodes and instead, from the map_custom callback (which runs on CPU with the routing known), hand-launch the expert matmul CUDA kernels directly (reading the hidden state from its ggml device buffer, writing the FFN output back)? What breaks? Name the exact ggml-cuda internals needed (ggml_cuda_mul_mat_id signature/staticness, the ggml_backend_cuda_context, the stream, the quant dequant path per type).\n2. If the MoE matmul is hand-launched inside the callback on the SAME stream, does the ggml scheduler even allow it (the callback is a CPU op; the next ggml node expects the FFN output tensor already computed)? Could we make the map_custom callback ITSELF do the whole MoE FFN (load experts + matmul + write output) so ggml sees one CPU op that produces the FFN output, no separate mul_mat_id GPU split? Would that ADD or REMOVE splits?\n3. The honest cost: enumerate what must be reimplemented (mul_mat_id for each quant type Q4_K etc, the MMVQ/dp4a dispatch, bias/scale). Is there a MINIMAL version (hy3 uses one quant type for experts) that is a few hundred lines, not a full engine? Give a concrete LOC estimate and the top 3 hazards.\nReturn structured findings.", { label: 'q1-manual', phase: 'Investigate', schema: SCHEMA }),

  () => agent(CTX + "\n\nYOUR TASK (Q2a - port Pulsar CopyStream H2D-overlap into our callback): Read Pulsar crates/kernels/src/lib.rs CopyStream (~525+, copy_async, the event ordering) and how engine/lib.rs uses it to overlap expert H2D with compute. Then read CatLlamaCpp src/llama-moe-stream.cpp (llama_moe_layer_parallel_load, the callback, the async-H2D batch we added) and ggml/src/ggml-cuda internals.\n1. Our async-H2D we added uses ggml_backend_tensor_set_async then ONE ggml_backend_synchronize before the callback returns - so it still fully blocks before the matmul. Pulsar's trick is the H2D runs on a SIDE stream and the CONSUMER kernel does cudaStreamWaitEvent so only the matmul waits, not the host, and other work proceeds. In our ggml setting, is there ANY compute that could proceed while the expert H2D is in flight? The problem: within layer il, the ONLY consumer of the loaded experts is il's own mul_mat_id, and there is nothing else to run. So overlap within-layer is impossible regardless of CopyStream. Confirm or refute this precisely.\n2. Therefore the ONLY way CopyStream-style overlap helps us is CROSS-layer: prefetch il+1's experts on a side stream WHILE il's matmul runs on the default stream. That needs il+1's expert identities during il - i.e. the lookahead prediction. Trace whether ggml lets us issue a side-stream H2D into the il+1 cache's device slots during il's graph execution (from the map_custom callback of il, or from the background loader thread). Is the background loader already the right place (it runs on its own thread, could own a CopyStream)? What stops the loader TODAY from overlapping - is it the synchronous ggml_backend_tensor_set (per-call cudaStreamSynchronize) it uses?\n3. Concrete port: give the loader a CopyStream (cudaMemcpyAsync from its pinned staging, no per-copy sync), so its RAM-fill and any prefetch H2D overlap the GPU's decode compute instead of contending. Does this require CUDA-specific code in llama-moe-stream (breaking backend-agnosticism), or is there a ggml async primitive that already uses a copy stream? Assess ggml_backend_tensor_set_async's actual stream behavior in ggml-cuda (does it use a separate copy stream or the compute stream?).\nReturn structured findings.", { label: 'q2-copystream', phase: 'Investigate', schema: SCHEMA }),

  () => agent(CTX + "\n\nYOUR TASK (Q2b - port Pulsar cross-layer prefetch as a PREFETCH-ONLY hint, honestly): We previously dismissed lookahead because predicted routing overlaps true top-2 only ~7% (cross-token) / 40% (one-layer full-8). But Pulsar USES one-layer prefetch for +21% and it is prefetch-ONLY (never drives compute, mispredict just wastes a background read). We have a BUILT but DEAD lookahead probe (src/models/hy-v3.cpp, LLAMA_MOE_LOOKAHEAD_PROBE, predicts il+1 routing from il hidden, 40% full-8). Read Pulsar engine/lib.rs prefetch (how it decides what to prefetch, where it puts it - host cache or VRAM, how the next layer consumes it) and CatLlamaCpp loader.\n1. Reconcile honestly: earlier analysis said one-layer prefetch has no lead time on OUR path because the callback holds the mutex and the GPU is idle during the callback (no compute to overlap). But Pulsar gets +21%. What is DIFFERENT that gives Pulsar the overlap? (Answer likely: Pulsar's GPU runs layer N's OTHER kernels on the default stream while the prefetch H2Ds N+1 on a side stream - it has compute to overlap because it is not blocked on a full-graph sync. We are blocked.) So does porting prefetch REQUIRE first breaking the full-sync (i.e. moving expert load off the map_custom critical path)? State the dependency chain.\n2. Given our decode is bound by the callback's synchronous load of THIS layer's true misses, would prefetching il+1 into the RAM pool (during il) actually reduce il+1's callback disk reads? Only if the prefetched experts are the ones il+1 actually needs (40% full-8 hit) AND they were not already RAM-resident. The measurement we ran showed callback RAM-hit is ~65% at RAM_FILL high. So lookahead could only help the ~35% disk misses, and only 40% of those are foreseeable = ~14% of misses. Quantify the realistic tok/s gain.\n3. The higher-value variant: prefetch il+1's predicted experts directly into the VRAM cache during il (on a side stream), so il+1's callback finds them resident and does NO load. This needs the loader to write VRAM slots off the critical path via a copy stream. Assess feasibility + the eviction risk (mispredict evicts hot experts from 16 slots).\nReturn structured findings.", { label: 'q2-prefetch', phase: 'Investigate', schema: SCHEMA }),
])

phase('Synthesize')

const VERDICT = {
  type: 'object',
  properties: {
    q1_answer: { type: 'string', description: 'Confident code-level answer to WHY NOT (or CAN) port manual per-kernel launch for the MoE section. Concrete blocker or concrete path + LOC/risk.' },
    q2_answer: { type: 'string', description: 'Confident code-level answer to WHY we ported no latency-hiding, and whether CopyStream-overlap / cross-layer prefetch CAN be ported. The dependency chain (what must change first).' },
    the_real_blocker: { type: 'string', description: 'The single root reason our decode has zero load-compute overlap while Pulsar has it, stated crisply.' },
    portable_plan: {
      type: 'array',
      description: 'Ordered, concrete things we CAN port, best ROI first, each with the mechanism and honest expected gain.',
      items: {
        type: 'object',
        properties: {
          item: { type: 'string' },
          mechanism: { type: 'string' },
          expected_gain: { type: 'string' },
          effort: { type: 'string' },
          risk: { type: 'string' },
        },
        required: ['item','mechanism','expected_gain','effort','risk'],
      },
    },
    recommendation: { type: 'string', description: 'What to build first to actually port a Pulsar latency-hiding idea (not align its test case). Honest.' },
  },
  required: ['q1_answer','q2_answer','the_real_blocker','portable_plan','recommendation'],
}

const verdict = await agent(CTX + "\n\nThree investigators dug into the user's two questions. Findings:\n\n## Q1 MANUAL PER-KERNEL (MoE section only)\n" + JSON.stringify(q1, null, 2) + "\n\n## Q2a COPYSTREAM H2D-OVERLAP\n" + JSON.stringify(q2copy, null, 2) + "\n\n## Q2b CROSS-LAYER PREFETCH\n" + JSON.stringify(q2prefetch, null, 2) + "\n\nYOUR TASK (SYNTHESIZE - give the user the confident code-level answers they've been missing): The user is right that we ported NO Pulsar latency-hiding and only patched our own bugs. Answer both questions definitively with code grounding. Identify the ONE root blocker (our ggml full-graph-sync at the map_custom split = GPU idle during load = nothing to overlap). Then give a concrete, ordered plan of what we CAN actually port from Pulsar (CopyStream on the loader, cross-layer prefetch into RAM/VRAM, or the more radical MoE-section-manual-launch), with honest expected gains grounded in our measured numbers (3.1 tok/s, ~48 experts/token, 65% RAM-hit, callback fully blocking). Do NOT recommend aligning the Q2 quant test case. Recommend what to BUILD to port a real overlap mechanism. Return the structured verdict.", { label: 'synthesis', phase: 'Synthesize', schema: VERDICT })

return { q1, q2copy, q2prefetch, verdict }
