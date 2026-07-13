#pragma once

#include <cstdint>
#include <cstdio>
#include <list>
#include <unordered_map>

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

// ggml custom-op callbacks, registered by the graph builder.
struct ggml_tensor;

// ggml_custom_4d callback: dst[ne0,ne1,k] <- experts of src[1] selected by the
// ids in src[0] (raw byte copy, so any quantization type works unchanged).
void llama_moe_gather_cb(ggml_tensor * dst, int ith, int nth, void * userdata);

// ggml_map_custom1 callback: dst[n_used,n_tokens] i32 <- compact slot index of
// each routing position in src `a`.
void llama_moe_remap_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

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

// Device slot table [1,n_expert] i32 (expert -> cache slot, sentinel `capacity` if missing).
ggml_tensor * llama_moe_layer_cache_slot_table(llama_moe_layer_cache * c);

// Device cache tensor [ne0,ne1,capacity+1] for one projection (matched by its `exps`).
ggml_tensor * llama_moe_layer_cache_dev(llama_moe_layer_cache * c, const ggml_tensor * exps);

// Publish this step's selection into the loader-visible buffer (adds a cpy to `gf`).
void llama_moe_layer_cache_publish(llama_moe_layer_cache * c,
                                   ggml_context *          ctx0,
                                   ggml_cgraph *           gf,
                                   ggml_tensor *           selected_experts);

// Synchronously warm the cache from a prefill selection (loads the most-frequent experts
// in the compute stream). Call during prefill so decode starts with a hot, correct cache.
void llama_moe_layer_cache_warm(llama_moe_layer_cache * c,
                                ggml_context *          ctx0,
                                ggml_cgraph *           gf,
                                ggml_tensor *           selected_experts);
