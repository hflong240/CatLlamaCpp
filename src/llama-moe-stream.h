#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <unordered_map>

// Env-switch truthiness with a caller-supplied default. Used so a switch can be defaulted ON (when the
// user passed --moe-stream-async) yet still be turned OFF with NAME=0. Unset -> default_on; "0"/"false"/
// "off" (case-insensitive) -> false; any other value -> true. Centralizes the semantics for every
// recommended-config switch so a bare presence check (which can never honor NAME=0) is not reintroduced.
static inline bool moe_env_on(const char * name, bool default_on) {
    const char * v = getenv(name);
    if (!v) {
        return default_on;
    }
#ifdef _WIN32
    #define moe_strcasecmp _stricmp
#else
    #define moe_strcasecmp strcasecmp
#endif
    if ((v[0] == '0' && v[1] == '\0') || moe_strcasecmp(v, "false") == 0 || moe_strcasecmp(v, "off") == 0) {
        return false;
    }
#undef moe_strcasecmp
    return true;
}

//
// SSD streaming cache for MoE routed-expert weights.
//
// In a GGUF, all experts of a layer share one 3D tensor [ne0, ne1, n_expert];
// expert e is the contiguous byte range [base + e*stride, base + (e+1)*stride).
// This cache keeps only a bounded byte budget of experts resident and fetches
// the rest from a backing source (file/mmap) on demand. It is backend-agnostic
// host-side infrastructure; callers upload resident bytes to CPU/GPU buffers.
//

// projection index within a MoE layer
enum llama_moe_proj {
    LLAMA_MOE_PROJ_GATE = 0,
    LLAMA_MOE_PROJ_UP   = 1,
    LLAMA_MOE_PROJ_DOWN = 2,
    LLAMA_MOE_PROJ_COUNT,
};

// on-disk description of one (layer, proj) expert slab array
struct llama_moe_expert_slab {
    uint64_t file_offset;   // byte offset of expert 0 in the source
    uint64_t expert_stride; // bytes per expert (== ggml_nbytes(tensor) / n_expert)
    uint32_t n_expert;
};

static inline uint64_t llama_moe_expert_offset(const llama_moe_expert_slab & s, uint32_t e) {
    return s.file_offset + (uint64_t) e * s.expert_stride;
}

// pack (layer, proj, expert) into a stable 64-bit cache key
static inline uint64_t llama_moe_stream_key(uint32_t layer, llama_moe_proj proj, uint32_t expert) {
    return ((uint64_t) layer << 32) | ((uint64_t) proj << 24) | (uint64_t) expert;
}

// abstract byte source (the model file, mmap, etc.)
struct llama_moe_source {
    virtual ~llama_moe_source() = default;
    // read `bytes` from absolute `offset` into `dst`; return true on success
    virtual bool read(void * dst, uint64_t offset, uint64_t bytes) = 0;
};

// FILE*-backed source (portable fallback; 64-bit seeks)
struct llama_moe_file_source : llama_moe_source {
    FILE * fp = nullptr;
    explicit llama_moe_file_source(FILE * f) : fp(f) {}
    bool read(void * dst, uint64_t offset, uint64_t bytes) override;
};

// mmap-backed source (zero-copy; OS pages on demand)
struct llama_moe_mmap_source : llama_moe_source {
    const uint8_t * base = nullptr;
    uint64_t        size = 0;
    llama_moe_mmap_source(const void * b, uint64_t s) : base((const uint8_t *) b), size(s) {}
    bool read(void * dst, uint64_t offset, uint64_t bytes) override;
};

struct llama_moe_stream_stats {
    uint64_t hits          = 0;
    uint64_t misses        = 0;
    uint64_t bytes_fetched = 0;
    uint64_t evictions     = 0;
};

// byte-budgeted LRU cache of resident experts
struct llama_moe_stream_cache {
    explicit llama_moe_stream_cache(uint64_t budget_bytes) : budget(budget_bytes) {}
    ~llama_moe_stream_cache();

    // return a pointer to `bytes` resident bytes for `key`, fetching from
    // `src` at `offset` on miss. returns nullptr on read failure.
    const void * acquire(uint64_t key, uint64_t offset, uint64_t bytes, llama_moe_source & src);

    uint64_t                       resident_bytes() const { return total; }
    const llama_moe_stream_stats & get_stats()      const { return stats; }

private:
    struct entry {
        uint64_t key;
        void *   buf;
        uint64_t bytes;
    };

    uint64_t budget;
    uint64_t total = 0;
    std::list<entry> order;                                      // front = most recently used
    std::unordered_map<uint64_t, std::list<entry>::iterator> map;
    llama_moe_stream_stats stats;

    void evict_to_fit(uint64_t incoming);
};

//
// Expert compaction (route-3 core).
//
// A MoE layer stores all experts in one 3D tensor [ne0, ne1, n_expert] and the
// matmul kernels require it fully resident. To stream, we run the matmul over a
// small tensor holding only the experts actually selected this step, and remap
// the routing ids into that compact tensor's slots. This keeps the existing
// ggml_mul_mat_id kernels unchanged on every backend.
//

// Given routing ids [n_used, n_tokens] (row-major, pos = t*n_used + e), compute
// the unique experts in first-appearance order. Writes slot_of_pos[pos] in
// [0,k) and expert_of_slot[0..k). Returns the compact expert count k
// (<= n_used*n_tokens and <= n_expert). n_expert bounds a scratch table.
int llama_moe_compaction(const int32_t * ids,
                         int64_t         n_used,
                         int64_t         n_tokens,
                         int             n_expert,
                         int32_t       * slot_of_pos,
                         int32_t       * expert_of_slot);

// Decide whether a step's routed experts require a synchronous sync-load, given a per-expert
// selection frequency table `freq[n_expert]` and the cache's current residency `expert_slot`
// (slot per expert, or -1 if not resident). Fills the count of unique selected experts and of
// misses (non-resident uniques), and writes ALL selected expert ids into `out_ranked`
// (size >= n_expert) ordered by descending frequency then ascending id, with `out_n_ranked`
// == n_unique. The caller keeps the top `capacity` of `out_ranked` resident and drops the rest.
// Returns true when the miss fraction exceeds `threshold` (i.e. sync-load is warranted).
bool llama_moe_plan_remap(const int * freq,
                          int         n_expert,
                          const int32_t * expert_slot,
                          float       threshold,
                          int *       out_n_unique,
                          int *       out_n_miss,
                          int32_t *   out_ranked,
                          int *       out_n_ranked);


// ggml custom-op callbacks, registered by the graph builder.
struct ggml_tensor;

// ggml_custom_4d callback: dst[ne0,ne1,k] <- experts of src[1] selected by the
// ids in src[0] (raw byte copy, so any quantization type works unchanged).
void llama_moe_gather_cb(ggml_tensor * dst, int ith, int nth, void * userdata);

// ggml_map_custom1 callback: dst[n_used,n_tokens] i32 <- compact slot index of
// each routing position in src `a`.
void llama_moe_remap_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// fork: host sync callback for the fused CUDA GGML_OP_MOE_FFN op (see ggml_moe_ffn). Signature must match
// ggml_moe_ffn_sync_fn in ggml.h. llama_moe_layer_cache_sync_fn() returns the concrete callback pointer to
// hand to ggml_moe_ffn; the callback plans residency + sync-loads missing experts + fills the slot ids.
typedef void (*llama_moe_ffn_sync_fn_t)(void * cache, const int32_t * sel_host, int32_t * ids_host,
                                        int n_used, int n_tokens);
llama_moe_ffn_sync_fn_t llama_moe_layer_cache_sync_fn(void);

// fork: CPU expert lane callback (see ggml_moe_ffn_cpu_fn in ggml.h). llama_moe_layer_cache_cpu_fn()
// returns it; pass to ggml_moe_ffn to compute RAM-resident experts on the CPU instead of the GPU.
typedef int (*llama_moe_ffn_cpu_fn_t)(void * cache, const float * cur_host, float * weights_host,
                                      const int32_t * sel_host, int32_t * ids_host, float * partial_host,
                                      int n_used, int n_tokens, int n_embd, int n_ff, int gpu_skip_slot);
llama_moe_ffn_cpu_fn_t llama_moe_layer_cache_cpu_fn(void);

//
// Persistent VRAM expert cache (for --moe-stream-async, decode only).
//
// Keeps up to `capacity` experts of one [ne0,ne1,n_expert] weight tensor resident
// on the compute device across tokens. Experts reused between tokens stay resident
// (no reload). On a miss the expert is loaded into an LRU slot; when `async` the
// current (stale) slot content is used for this token and the real expert is loaded
// in the background for later tokens.
//
struct ggml_context;
struct ggml_backend_sched;
typedef struct ggml_backend_sched * ggml_backend_sched_t;

// Build the cache path for one expert weight tensor. Returns the persistent device
// cache tensor [ne0,ne1,capacity] to feed as mul_mat_id src0, and via `out_ids` the
// remapped ids [n_used,n_tokens] pointing into the cache. Returns nullptr on failure
// (caller should fall back to the non-cached path).
ggml_tensor * llama_moe_cache_build(ggml_context *       ctx0,
                                    ggml_backend_sched_t sched,
                                    ggml_tensor *        exps,
                                    ggml_tensor *        selected_experts,
                                    int                  capacity,
                                    bool                 async,
                                    ggml_tensor **       out_ids);

// Experimental ceiling test: full-resident cache + a GPU-side get_rows remap over a
// VRAM slot table (no per-layer CPU op). Measures the speed lever of moving the remap
// off the CPU. Returns the device cache tensor and, via out_ids, the remapped ids.
ggml_tensor * llama_moe_cache_build_getrows(ggml_context *       ctx0,
                                            ggml_backend_sched_t sched,
                                            ggml_tensor *        exps,
                                            ggml_tensor *        selected_experts,
                                            ggml_tensor **       out_ids);

// Async stale cache: small VRAM cache kept warm by a background loader from the latest
// selections (published into the graph via `gf`); the remap is a device get_rows over
// the slot table so nothing blocks. Misses use the current (stale) slot. `capacity`
// experts resident per pool (<=0 => full). Returns the device cache tensor + out_ids.
struct ggml_cgraph;
ggml_tensor * llama_moe_cache_build_async(ggml_context *       ctx0,
                                          ggml_cgraph *        gf,
                                          ggml_backend_sched_t sched,
                                          ggml_tensor *        exps,
                                          ggml_tensor *        selected_experts,
                                          int                  capacity,
                                          ggml_tensor **       out_ids);

// Stop the async loader thread (call before the model/CUDA context is destroyed).
void llama_moe_cache_shutdown(void);

// Token-boundary backpressure sync (decode only). Call once per generated token AFTER the graph has
// computed and the GPU is synced, BEFORE the next token's graph is built. Synchronously loads the
// top-`budget` (highest-weight) experts of every layer that the background loader has not kept
// resident, blocking the next token until they land - so generation drifts on stale experts by at most
// one token. Decode itself stays on the pure-GPU stale_table path (no per-layer CPU op); this is a
// once-per-token backstop that only stalls when the loader fell behind. Returns experts loaded (0 => no
// stall). No-op if budget<=0. Safe to call unconditionally on the single-token decode path.
int llama_moe_boundary_sync(int budget);

// Per-token freshness stride (LLAMA_MOE_SYNC_STRIDE=N, N<=1 = off, the default).
//
// Every N-th single-token decode step keeps the ordinary synchronous expert budget; the N-1 steps in
// between inspect no routing position at all and fall through to the existing per-position stale reuse
// (pos_slot), so synchronous loads per token drop by roughly 1/N. This scales a runtime count only - no
// node is added or removed and no graph is rebuilt, so graph reuse and CUDA-graph replay are unaffected
// and the "with bs=1" graph-splits count must stay identical to a non-strided run.
//
// llama_moe_stride_begin_step must be called exactly ONCE PER DECODE STEP, before the graph runs. The
// remap callback runs once per MoE layer, so it must never derive the phase itself. Arguments:
//   single_token - false for any prompt / re-prompt batch: disarms the stride AND restarts the phase.
//                  This is the only prefill guard that works (with -ub 1 every prompt ubatch looks like
//                  a single token inside the callback) and the only thing that makes the first generated
//                  token of turn 2 and later fresh, since c->pos_init is never cleared again.
//   tok          - this step's input token id, or -1.
//   stop_tok     - reasoning-close token id, or -1. With LLAMA_MOE_SYNC_STRIDE_THINK_ONLY=1 the stride
//                  is dropped for the rest of the turn once stop_tok has been fed back in, so only the
//                  reasoning span is strided and the final answer runs at full freshness.
//
// LLAMA_MOE_SYNC_STRIDE_KEEP_LAYERS=K keeps layers with il < K out of the stride (they always sync-load).
// Default 0 = stride every layer. K=3 on DeepSeek-V4-Flash matches its token-id hash-routed front layers,
// whose pos_slot donor is the slot the PREVIOUS token's table row used and so carries no locality; that
// was measured to cost nothing in speed either way, so it is left as an experiment rather than a default.
//
// Steps 0 and 1 of every decode span always keep the budget: the one-shot post-sweep VRAM refill runs
// between them and does not honour slot_pin, so a free-fly step 1 would reuse donors that pos_slot
// recorded before the refill moved them. A layer also refuses to free-fly when n_sentinel < n_used,
// where an unusable donor would fall back to a live WRONG expert at full gate weight.
void llama_moe_stride_begin_step(bool single_token, int32_t tok, int32_t stop_tok);

// True if this step is a free-fly step (no synchronous expert loading). Read by the remap callback.
bool llama_moe_stride_freefly(void);

// Register the on-disk location of an expert weight tensor so the layer cache can read experts
// directly from the model file (fread) instead of from the tensor's mmap-backed `data` pointer.
// This is the "no-mmap takeover" path: it lets the cache avoid touching the OS page cache for
// experts entirely (the page cache cannot be evicted per-expert on Windows, so leaving experts
// there thrashes RAM when the model is larger than RAM). `file_offset` is the absolute byte
// offset of expert 0 of `exps` in `path`. Call once per expert tensor after model load, before
// the first graph build. Only consulted when LLAMA_MOE_NOMMAP is set.
void llama_moe_register_expert_file(const ggml_tensor * exps, const char * path, uint64_t file_offset);

// fork: LOW-PRECISION EXPERT TIER (--moe-expert-gguf-low). Register the low-quantization TWIN of the
// routed-expert tensor `hi`: its ggml type, its 3D shape, and where its bytes start in the second GGUF.
// No memory is allocated for the twin and it is never mmapped - the streaming layer builds a parallel
// per-layer cache over a meta-only descriptor and freads experts from the file on demand. Call once per
// expert tensor at model load, before the first graph build.
// `type` is a ggml_type passed as int: this header only forward-declares the ggml structs it needs and
// an unscoped enum cannot be forward-declared, so keeping it an int avoids pulling in ggml.h here.
void llama_moe_register_low_tier(const ggml_tensor * hi, int type, const int64_t * ne,
                                 const char * path, uint64_t file_offset);

// True once any low-precision twin has been registered (i.e. the two-tier mode is available).
bool llama_moe_low_tier_enabled(void);

// Fill `out_lo[0..n_proj)` with the low-precision twins of `exps_list[0..n_proj)`. Returns n_proj on
// success, or 0 if the tier is off or any projection lacks a twin (all-or-nothing: a partially twinned
// layer would mix precisions inside a single FFN pass, which the two-pass sum cannot express).
int llama_moe_low_tier_list(ggml_tensor * const * exps_list, int n_proj, ggml_tensor ** out_lo);

//
// Per-layer async expert cache with drop-on-miss (the practical async path).
//
// Unlike the per-tensor pools above, one cache covers ALL projection tensors of a
// MoE layer (gate/up/down or fused gate_up/down) so an expert's residency is a single
// per-layer decision: an expert is "resident" only when loaded into every projection.
// The device remap is a get_rows over a shared slot table (no CPU op in the critical
// path); a background loader keeps the hot experts resident. On a miss the expert is
// NOT replaced by a wrong one - instead its routing weight is masked to zero (the
// caller multiplies the gate weights by `resident_flag`), so a missing expert is
// gently dropped rather than corrupting the output. This keeps generation coherent at
// small cache sizes while running at near-resident speed.
//
struct llama_moe_layer_cache;

// Create or look up the per-layer cache covering `exps_list[0..n_proj)`. `capacity`
// experts are kept resident per projection (<=0 => full n_expert). Returns nullptr on
// failure (caller falls back to the non-cached path).
llama_moe_layer_cache * llama_moe_layer_cache_get(ggml_backend_sched_t sched,
                                                  ggml_tensor * const * exps_list,
                                                  int                   n_proj,
                                                  ggml_tensor *         selected_experts,
                                                  int                   capacity);

// Look up an existing per-layer cache by its first projection tensor (no creation). Returns null if not
// yet created. Used by the cross-layer prefetch to reach layer L+1's cache from layer L's graph build.
llama_moe_layer_cache * llama_moe_layer_cache_lookup(const ggml_tensor * exps0);

// Auto-choose the resident experts-per-layer (CACHE_CAP) from free VRAM when the user did not set
// LLAMA_MOE_CACHE_CAP. `n_moe_layers` is the count of layers that build a streamed MoE cache (total
// layers minus the leading dense block). Returns a capacity in [n_used, n_expert], or 0 if it cannot
// measure (caller keeps its own default). Tune the VRAM fraction with LLAMA_MOE_VRAM_FRAC (default 0.80).
//
// fork two-tier mode: pass BOTH tiers' expert tensors in `exps_list` (the low-precision twins are
// recognised by having a registered high twin, so the order does not matter) and a non-null
// `cap_lo_out`. The single byte budget is then split between the tiers instead of being spent on one
// capacity shared by both - which, since a low slab is about half a high one, silently handed the high
// tier ~2/3 of the VRAM. `cap_hi_fixed > 0` means the caller already has an explicit high capacity
// (LLAMA_MOE_CACHE_CAP) and only wants the low tier sized against what that leaves.
int llama_moe_auto_capacity(ggml_backend_sched_t  sched,
                            ggml_tensor * const * exps_list,
                            int                   n_proj,
                            int                   n_expert,
                            int                   n_used,
                            int                   n_moe_layers,
                            int                   cap_hi_fixed,
                            int *                 cap_lo_out);

// Device slot table [1,n_expert] i32 (expert -> cache slot, sentinel `capacity` if missing).
ggml_tensor * llama_moe_layer_cache_slot_table(llama_moe_layer_cache * c);

// Device stale table [1,n_expert] i32 (expert -> a REAL settled slot in [0,capacity), never sentinel).
// The pure-GPU decode remap reads this so a non-resident expert reuses a real (stale) expert instead of
// zeroing the FFN branch. Kept consistent with slot_table by the loader and the boundary sync.
ggml_tensor * llama_moe_layer_cache_stale_table(llama_moe_layer_cache * c);

// Device cache tensor [ne0,ne1,capacity+1] for one projection (matched by its `exps`).
ggml_tensor * llama_moe_layer_cache_dev(llama_moe_layer_cache * c, const ggml_tensor * exps);

// True if this cache's resident device backend is CUDA. The fused GGML_OP_MOE_FFN op is a CUDA-only fork
// op (CPU aborts, no Vulkan/Metal/SYCL case), so the fused decode path must be gated on this: on a non-CUDA
// GPU build llama_moe_pick_device_backend still returns the (non-CPU) backend and builds the cache, but
// emitting the fused op there would crash the scheduler. Callers fall back to the per-projection path.
bool llama_moe_layer_cache_backend_is_cuda(llama_moe_layer_cache * c);

// fork: read-only gate-weight distribution probe (LLAMA_MOE_WEIGHT_DIAG). Attaches a no-op map_custom on
// the normalized `weights` [1, n_expert_used, n_tokens] that bins the per-token top-1/top-2 weight, to
// measure how often hy3's routing is "flat" (top-2 low). Its output is discarded (never consumed), so it
// cannot affect the model computation. No-op unless LLAMA_MOE_WEIGHT_DIAG is set.
void llama_moe_weight_diag_probe(ggml_context * ctx0, ggml_cgraph * gf, ggml_tensor * weights);

// Publish this step's selection into the loader-visible buffer (adds a cpy to `gf`).
void llama_moe_layer_cache_publish(llama_moe_layer_cache * c,
                                   ggml_context *          ctx0,
                                   ggml_cgraph *           gf,
                                   ggml_tensor *           selected_experts);

// fork: cross-layer lookahead prefetch. While building layer L's graph, publish a PREDICTED expert
// selection for layer L+1 into L+1's cache sel_buf, so the background loader can warm those experts
// during L's compute (hiding L+1's disk read). The prediction is L's hidden projected through L+1's
// gate_inp (cheap, on-GPU) - approximate, so pass a WIDER top-K than n_used (more candidates => higher
// chance the real top-2 are already warm). Purely speculative: a wrong prediction just wastes a load,
// never corrupts output (L+1 still runs its real selection + SYNC_BUDGET). No-op if next_c is null or
// pred's shape does not fit sel_buf. `pred` is [k, n_tokens] i32 (k may exceed sel_buf->ne[0]; only the
// leading sel_buf->ne[0] rows are published, so pass the argsort so the highest-weight candidates lead).
void llama_moe_layer_cache_prefetch(llama_moe_layer_cache * next_c,
                                    ggml_context *          ctx0,
                                    ggml_cgraph *           gf,
                                    ggml_tensor *           pred);

// Remap `selected_experts` [n_used,n_tokens] to cache-slot ids [n_used,n_tokens] i32 via a CPU
// op (map_custom1). Resident experts resolve to their slot; misses resolve to the zero sentinel
// slot (`capacity`, dropped) UNLESS this step's miss fraction exceeds `threshold`, in which case
// the missing experts are synchronously loaded into free/LRU slots first (bounded by capacity).
// Works for prefill (n_tokens>1) and decode (n_tokens<=1): the cold first step sync-loads the
// prompt's experts, later steps run near-resident and drop rare misses. The returned ids tensor
// is shared across all projections of the layer (a single frozen slot assignment per step). The
// slots it resolves are pinned so the background loader will not evict them before the matmul
// that consumes these ids has run. Returns nullptr if the cache/selection is unusable.
ggml_tensor * llama_moe_layer_cache_remap(llama_moe_layer_cache * c,
                                          ggml_context *          ctx0,
                                          ggml_tensor *           selected_experts,
                                          ggml_tensor *           weights,
                                          float                   threshold,
                                          int                     sync_budget);

// fork: TWO-TIER DECODE REMAP (--moe-expert-gguf-low). Same contract as llama_moe_layer_cache_remap for
// the high tier, but resolves the positions the high tier could NOT serve against a second cache holding
// a low-precision twin of the same experts - so an out-of-budget position gets the CORRECT expert at
// reduced precision instead of a wrong (stale) expert at full precision.
//
// Returns ONE i32 tensor packing everything the caller needs, so the two-tier path costs the same
// single CPU op (one ggml split) as the one-tier path. Two output layouts, selected by `fused`:
//
//   fused == false (masked two-pass sum) - [n_used, 3*n_tokens]:
//     rows [0*n_tokens, 1*n_tokens) - high-tier slot ids  -> ids for the high mul_mat_id pass
//     rows [1*n_tokens, 2*n_tokens) - low-tier slot ids   -> ids for the low  mul_mat_id pass
//     rows [2*n_tokens, 3*n_tokens) - 1 if the high tier served the position, else 0
//   The caller takes the first two blocks as contiguous 2D views and turns the third into the f32 gate
//   mask via ggml_get_rows over llama_moe_layer_cache_tier_weights(c_hi) (the constant {0,1} table):
//     w_hi = weights * mask; w_lo = weights - w_hi; out = FFN_hi(x)*w_hi + FFN_lo(x)*w_lo
//   which applies every routed position exactly once (one of the two weights is a hard zero).
//
//   fused == true (ggml_mul_mat_id_2t) - [n_used, 2*n_tokens]:
//     rows [0*n_tokens, 1*n_tokens) - UNIFIED slot ids: a high slot as-is, a low slot biased by the
//                                     high cache's slot count (c_hi->proj[0].dev->ne[2]), so one id
//                                     tensor indexes both tiers in the op's unified channel space.
//     rows [1*n_tokens, 2*n_tokens) - 1 if the high tier served the position, else 0: the op's `tier`
//                                     selector, kept self-describing (the CUDA path derives it from id).
//   The caller feeds both blocks to a single ggml_mul_mat_id_2t per projection, which computes each
//   position exactly once against its owning tier - no mask, no complement, no sum.
//
// Decode only (n_tokens == 1 in practice); prefill keeps the lossless expert-group sweep on the high
// tier. Returns nullptr if either cache is unusable, in which case the caller falls back to the
// single-tier path.
ggml_tensor * llama_moe_layer_cache_remap_tiered(llama_moe_layer_cache * c_hi,
                                                 llama_moe_layer_cache * c_lo,
                                                 ggml_context *          ctx0,
                                                 ggml_tensor *           selected_experts,
                                                 ggml_tensor *           weights,
                                                 float                   threshold,
                                                 int                     sync_budget,
                                                 bool                    fused);

// Constant device table [1,2] f32 = {0.0f, 1.0f} owned by a high-tier cache; get_rows over it turns the
// packed 0/1 tier selector into the f32 gate mask. Null unless the two-tier mode is active.
ggml_tensor * llama_moe_layer_cache_tier_weights(llama_moe_layer_cache * c);

// fork: PREFILL EXPERT-GROUP SWEEP (LLAMA_MOE_PREFILL_SWEEP=1, opt-in).
//
// The plain prefill remap above is bounded by `capacity`: once a step's unique experts exceed the
// resident slots it stops loading and the remainder drop to a zero sentinel, so their gate weight is
// simply lost (the router already normalized). For an MoE with many experts and a small top-k this
// bites immediately - a few hundred tokens light up essentially every expert - which is why a wide
// prefill ubatch degrades quality while ubatch 1 does not.
//
// The sweep removes the cap instead of rationing against it. The expert INDEX space is partitioned
// into fixed groups of at most `capacity` entries, and the whole MoE FFN is evaluated once per group.
// Within a group every expert the ubatch selected is guaranteed resident (the group cannot exceed
// capacity by construction), so there is no miss and nothing to drop. Positions routed outside the
// group resolve to a zero sentinel and contribute exactly 0, so summing the per-group outputs
// reproduces the full result losslessly. Cost is one extra FFN pass per group, not extra expert bytes:
// each expert is still read at most once per ubatch step.
//
// The partition is static (derived from n_expert and capacity, both run-constant), so graph topology
// stays fixed - no data-dependent shapes, no new kernels, mul_mat_id and its grouped-GEMM path are
// used unchanged.
//
// `dep` MUST be a tensor produced by the previous group's FFN output (any cheap 1-element derivative
// of it). It is read for its scheduling edge only, never for its value: it forces the scheduler to
// finish group g-1's device kernels before this callback runs, because group g reuses - and therefore
// overwrites - group g-1's slots. Pass nullptr for the first group.
bool llama_moe_prefill_sweep_enabled(void);

// Zero sentinel slots per layer cache. Single source of truth shared by llama_moe_auto_capacity
// (which must subtract these from the VRAM budget) and the cache allocation.
int llama_moe_sentinel_count(int n_used);

// fork: queue an opportunistic prefetch of `n_ids` expert ids for the MoE layer at index `il`. Non-blocking
// hint only - it appends to that layer cache's prefetch queue and returns; the background loader services it
// at the top of its next pass, in ascending layer order. Nothing downstream depends on the bytes arriving:
// until the loader publishes, the expert still reads as non-resident, so the remap callback applies its
// ordinary sync-budget / coverage / stale-reuse rules to it. A late prefetch is wasted, never wrong. H2D on
// the calling thread is not an option (ggml_backend_tensor_set synchronizes per call), hence the hand-off.
//
// The exploitable case is a layer whose routing does not depend on the hidden state. DeepSeek-V4 routes its
// first dsv4_hash_layer_count layers through a static token-id -> expert table, so the exact expert set is a
// pure function of the token id and is known before the graph is built - a prefetch oracle with no prediction
// error. Opt in with LLAMA_MOE_HASH_PREFETCH=1 (both this and the caller gate on it).
//
// MEASURED NEGATIVE on dsv4 IQ2 / cap=36, which is why it is off by default. Keep the mechanism, do not
// re-try it blindly. What was verified:
//  - the oracle is exact: the ids read here match the ids the graph's mul_mat_id receives, byte for byte
//    (LLAMA_MOE_PFDBG prints both sides)
//  - those layers really are the worst in the model: ~43-51% VRAM hit / ~47-55% stale versus ~64-70% /
//    ~22-27% everywhere else, in every configuration tried (with and without the prefill sweep, on
//    repetitive and on varied output), so the target was worth attacking
//  - the plumbing works: requests are queued and the loader publishes hundreds of them per generation
//  - and yet the hit rate of those layers does not move, across four variants: drain once per loader pass;
//    drain between every cache; pin the published slot so nothing can recycle it; and batch per layer
//    through parallel_load for 16-way reads plus a single device synchronize.
// The reason is horizon, not accuracy or eviction. Entry-time residency (see g_moe_pf_resident) matches the
// callback's hit rate almost exactly, so nothing is being lost between decode entry and the FFN - the
// prefetched bytes simply arrive after that layer's matmul has already read the slot table, and by the next
// token the ids have changed. Covering even one hash layer means moving ~n_used expert slabs inside the few
// ms between the token becoming known and layer 0 executing, which is tens of GB/s. This is the same wall
// the cross-layer prefetch hit, for a different reason: there the predictions were wrong, here they are
// perfect and there is no time to act on them.
void llama_moe_layer_prefetch(int il, const int32_t * ids, int n_ids);

// Resident slot count of this layer cache (excludes the zero sentinel slots).
int llama_moe_layer_cache_capacity(const llama_moe_layer_cache * c);

// Has this cache already served a single-token (decode) step? Used to tell a real prefill/warmup
// ubatch from a speculative-verify batch, which is a multi-token batch arriving mid-generation.
bool llama_moe_layer_cache_decoded(const llama_moe_layer_cache * c);

ggml_tensor * llama_moe_layer_cache_remap_group(llama_moe_layer_cache * c,
                                                ggml_context *          ctx0,
                                                ggml_tensor *           selected_experts,
                                                int                     group,
                                                ggml_tensor *           dep);

// fork: TWO-TIER prefill sweep (LLAMA_MOE_TIER_SWEEP, CUDA only). Like llama_moe_layer_cache_remap_group,
// but a group spans c_hi->capacity + c_lo->capacity experts, split across the two caches by selection
// frequency, and the returned tensor is a [n_used, 2*n_tokens] pack (row block 0 = unified slot ids, row
// block 1 = 0/1 tier selector) feeding one ggml_mul_mat_id_2t per projection. Returns null on refusal
// (bad shapes, mismatched n_expert, or n_sentinel < n_used), so the caller can fall back cleanly.
ggml_tensor * llama_moe_layer_cache_remap_group_tiered(llama_moe_layer_cache * c_hi,
                                                       llama_moe_layer_cache * c_lo,
                                                       ggml_context *          ctx0,
                                                       ggml_tensor *           selected_experts,
                                                       int                     group,
                                                       ggml_tensor *           dep);

// A sweep leaves every layer cache holding its LAST expert group - an index range unrelated to what
// decode will route to - so decode would run cold for the rest of the generation. These two restore
// the score-ranked working set once, at the first single-token step after a sweep. Call
// llama_moe_refill_vram_caches only after the device is synchronized: it overwrites slabs that the
// just-finished graph read.
bool llama_moe_sweep_refill_pending(void);
void llama_moe_refill_vram_caches(void);

// Dump every layer cache's resident expert set (LLAMA_MOE_RESDBG). Used to answer "does the sweep leave
// a different resident set than an ordinary prefill" by measurement instead of by comparing throughput.
void llama_moe_dump_residency(const char * tag);

// One-shot VRAM audit, run once after allocation: warns when the auto-sized expert cache has left too
// little device memory for the graph's transients (a silent shared-memory spill that costs throughput).
void llama_moe_vram_audit(void);

// Two-phase auto capacity: llama_context sets defer=true for its first pp graph reserve so the compute
// buffer can be measured with no expert caches allocated, publishes that size, then clears defer so the
// later reserve passes create the caches sized against the real number instead of a flat guess.
// NOTE: the defer=true variant is a dead end (no cache => build_moe_ffn takes the compaction path, whose
// transient is the full n_expert-wide expert tensor, so the measured buffer is larger than the real one).
// llama_moe_recap_from_compute_reserve below is the working form: measure with a flat-guess cache in
// place (same graph shape, and the buffer does not depend on the capacity), then rebuild against it.
void llama_moe_set_defer_caches(bool v);
void llama_moe_set_compute_reserve(size_t bytes);

// Publish the MEASURED device-side compute-buffer size and invalidate the expert caches so the next
// graph build re-sizes them against it instead of LLAMA_MOE_VRAM_RESERVE_MB's flat guess. Returns true
// if anything was invalidated, i.e. the caller must re-run its graph reserve. `sched` is the calling
// context's scheduler and scopes the invalidation to the caches THAT context built - the cache map is
// process-global, so without it a draft/embedding context created later would free the main model's
// caches out from under its live graphs. No-op when the capacity is pinned by hand
// (LLAMA_MOE_CACHE_CAP), when the per-tensor compaction pools are in use, when this context owns no
// caches, or when LLAMA_MOE_AUTOCAP_RECAP=0. Call during startup only, before llama_moe_prefill_once.
bool llama_moe_recap_from_compute_reserve(ggml_backend_sched_t sched, size_t compute_bytes);

// True while llama_moe_recap_from_compute_reserve is still going to fire for the next context that
// reserves a graph: auto capacity (no LLAMA_MOE_CACHE_CAP), recap enabled, nothing measured yet. Lets the
// caller recognise its first reserve as a MEASUREMENT pass before it runs. Cannot account for the
// compute-buffer size being 0 or for the context owning no caches, so a "pending" recap can still decline
// - which is why llama_moe_cache_defer_prewarm repairs rather than assumes.
bool llama_moe_recap_pending(void);

// While deferred, a newly built layer cache prewarms ONE slot instead of all `capacity` of them. The
// measurement pass's caches are dropped by the recap, and filling them first copies `capacity` experts per
// layer into VRAM for a cache nothing ever computes with (measured 14.1 GiB / 2.7s of startup on
// qwen3.8-flash-next). Slot 0 is still filled so stale_table keeps its "every expert maps to a real
// settled slot" invariant. Clearing the flag completes the prewarm of any deferred cache that survived, so
// a recap that declines after all is still correct - just as slow as before.
// LLAMA_MOE_DEFER_PREWARM=0 turns the deferral off (diagnostic: A/B it against the old behaviour).
void llama_moe_cache_defer_prewarm(bool defer);
