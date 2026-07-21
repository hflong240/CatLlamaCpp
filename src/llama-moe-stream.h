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

// Register the on-disk location of an expert weight tensor so the layer cache can read experts
// directly from the model file (fread) instead of from the tensor's mmap-backed `data` pointer.
// This is the "no-mmap takeover" path: it lets the cache avoid touching the OS page cache for
// experts entirely (the page cache cannot be evicted per-expert on Windows, so leaving experts
// there thrashes RAM when the model is larger than RAM). `file_offset` is the absolute byte
// offset of expert 0 of `exps` in `path`. Call once per expert tensor after model load, before
// the first graph build. Only consulted when LLAMA_MOE_NOMMAP is set.
void llama_moe_register_expert_file(const ggml_tensor * exps, const char * path, uint64_t file_offset);

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
int llama_moe_auto_capacity(ggml_backend_sched_t  sched,
                            ggml_tensor * const * exps_list,
                            int                   n_proj,
                            int                   n_expert,
                            int                   n_used,
                            int                   n_moe_layers);

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
