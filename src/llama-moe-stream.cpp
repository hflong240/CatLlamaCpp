#include "llama-moe-stream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "llama-impl.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif

#ifdef _WIN32
#  define llama_moe_fseek64 _fseeki64
#else
#  define llama_moe_fseek64 fseeko
#endif

// Hint the OS to read a byte range of the mmap'd model file in one large asynchronous
// I/O instead of faulting it in a page at a time on first access. This removes most of
// the per-page fault overhead and lets a fast (NVMe) drive deliver its full sequential
// bandwidth for the expert-streaming reads; on a slow (SATA) drive it is close to a
// no-op since the drive itself is the bottleneck.
static inline void llama_moe_prefetch(const void * addr, size_t bytes) {
    if (!addr || bytes == 0) {
        return;
    }
#ifdef _WIN32
    WIN32_MEMORY_RANGE_ENTRY e;
    e.VirtualAddress = const_cast<void *>(addr);
    e.NumberOfBytes  = bytes;
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &e, 0);
#else
    (void) madvise(const_cast<void *>(addr), bytes, MADV_WILLNEED);
#endif
}

// Clamp a requested resident-cache capacity (experts per layer) into a sane range:
//   capacity <= 0        -> full (all n_expert experts)
//   capacity  > n_expert -> full (nothing to gain past the whole layer)
//   capacity  < n_used   -> raised to n_used (the per-token expert activation)
// A cache smaller than n_used cannot hold even a single token's selected experts, so the
// drop-on-miss path would lose experts every token (garbage) and the reload path would
// thrash. n_used is the hard floor; more is strongly recommended for a usable working set.
static int llama_moe_clamp_capacity(int capacity, int n_expert, int n_used) {
    if (capacity <= 0 || capacity > n_expert) {
        return n_expert;
    }
    if (capacity < n_used) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            LLAMA_LOG_WARN("MoE stream: cache capacity (%d) is below the per-token expert "
                           "activation (%d); raising to %d. Increase LLAMA_MOE_CACHE_CAP well "
                           "above %d for usable output.\n", capacity, n_used, n_used, n_used);
        }
        return n_used;
    }
    return capacity;
}

bool llama_moe_file_source::read(void * dst, uint64_t offset, uint64_t bytes) {
    if (!fp) {
        return false;
    }
    if (llama_moe_fseek64(fp, (int64_t) offset, SEEK_SET) != 0) {
        return false;
    }
    return fread(dst, 1, (size_t) bytes, fp) == (size_t) bytes;
}

bool llama_moe_mmap_source::read(void * dst, uint64_t offset, uint64_t bytes) {
    if (!base || offset + bytes < offset || offset + bytes > size) {
        return false;
    }
    memcpy(dst, base + offset, (size_t) bytes);
    return true;
}

llama_moe_stream_cache::~llama_moe_stream_cache() {
    for (auto & e : order) {
        free(e.buf);
    }
}

void llama_moe_stream_cache::evict_to_fit(uint64_t incoming) {
    while (total + incoming > budget && !order.empty()) {
        entry & victim = order.back();
        total -= victim.bytes;
        free(victim.buf);
        map.erase(victim.key);
        order.pop_back();
        stats.evictions++;
    }
}

const void * llama_moe_stream_cache::acquire(uint64_t key, uint64_t offset, uint64_t bytes, llama_moe_source & src) {
    auto it = map.find(key);
    if (it != map.end()) {
        order.splice(order.begin(), order, it->second); // promote to most-recently-used
        stats.hits++;
        return it->second->buf;
    }
    stats.misses++;

    // a single expert larger than the whole budget still gets served, but it
    // forces every other entry out first so the invariant total<=budget holds
    // for cached entries once it is later evicted.
    evict_to_fit(bytes < budget ? bytes : budget);

    void * buf = malloc((size_t) bytes);
    if (!buf) {
        return nullptr;
    }
    if (!src.read(buf, offset, bytes)) {
        free(buf);
        return nullptr;
    }
    stats.bytes_fetched += bytes;

    order.push_front(entry{ key, buf, bytes });
    map[key] = order.begin();
    total += bytes;
    return buf;
}

//
// expert compaction
//

int llama_moe_compaction(const int32_t * ids,
                         int64_t         n_used,
                         int64_t         n_tokens,
                         int             n_expert,
                         int32_t       * slot_of_pos,
                         int32_t       * expert_of_slot) {
    // slot_of_expert maps an expert id to its compact slot, or -1 if unseen.
    // sized to n_expert (ids are guaranteed < n_expert by the router).
    std::vector<int32_t> slot_of_expert(n_expert > 0 ? n_expert : 1, -1);

    int           k = 0;
    const int64_t n = n_used * n_tokens;
    for (int64_t p = 0; p < n; ++p) {
        const int32_t e = ids[p];
        int32_t s = (e >= 0 && e < n_expert) ? slot_of_expert[e] : -1;
        if (s < 0) {
            s = k;
            if (e >= 0 && e < n_expert) {
                slot_of_expert[e] = k;
            }
            expert_of_slot[k] = e;
            k++;
        }
        slot_of_pos[p] = s;
    }
    return k;
}

// read the [n_used, n_tokens] i32 ids of a tensor into a flat row-major buffer
// (pos = t*n_used + e), respecting tensor strides.
static void llama_moe_flatten_ids(const ggml_tensor * ids, std::vector<int32_t> & out) {
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    out.resize((size_t) (n_used * n_tokens));
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_used; ++e) {
            const char * src = (const char *) ids->data + t * ids->nb[1] + e * ids->nb[0];
            out[(size_t) (t * n_used + e)] = *(const int32_t *) src;
        }
    }
}

void llama_moe_gather_cb(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    (void) userdata;
    if (ith != 0) {
        return; // single-threaded: only the first task does the copy
    }

    const ggml_tensor * ids = dst->src[0];
    const ggml_tensor * w   = dst->src[1];

    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const int     n_expert = (int) w->ne[2];
    const int64_t n        = n_used * n_tokens;

    std::vector<int32_t> flat;
    llama_moe_flatten_ids(ids, flat);

    std::vector<int32_t> slot_of_pos((size_t) n);
    std::vector<int32_t> expert_of_slot((size_t) n);
    const int k = llama_moe_compaction(flat.data(), n_used, n_tokens, n_expert,
                                       slot_of_pos.data(), expert_of_slot.data());

    // one expert slab is nb[2] bytes; dst and src share ne0/ne1/type so strides match
    const size_t stride = (size_t) w->nb[2];
    // Copy the k selected expert slabs from the mmap'd file. Reads are (a) issued in
    // ascending file-offset order and (b) split into contiguous per-thread ranges, so each
    // thread walks the file forward instead of seeking randomly - on an SSD this turns a
    // scattered read pattern (~queue-depth-1 random) into several sequential streams, which
    // is markedly faster. dst slabs are disjoint, so the parallel writes do not conflict.
    {
        // order[] indexes slots sorted by their source expert offset
        std::vector<int> order((size_t) k);
        for (int j = 0; j < k; ++j) {
            order[(size_t) j] = j;
        }
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return expert_of_slot[a] < expert_of_slot[b]; });

        // hint the OS to stream the slabs in (large async reads) in that same order
        for (int idx = 0; idx < k; ++idx) {
            const int j = order[(size_t) idx];
            llama_moe_prefetch((const char *) w->data + (size_t) expert_of_slot[j] * stride, stride);
        }

        const char * env = getenv("LLAMA_MOE_IO_THREADS");
        int n_threads = env ? atoi(env) : 16;
        if (n_threads < 1) { n_threads = 1; }
        if (n_threads > k) { n_threads = k; }

        auto copy_range = [&](int lo, int hi) {
            for (int idx = lo; idx < hi; ++idx) {
                const int j = order[(size_t) idx];
                memcpy((char *) dst->data + (size_t) j * dst->nb[2],
                       (const char *) w->data + (size_t) expert_of_slot[j] * stride,
                       stride);
            }
        };

        if (n_threads <= 1) {
            copy_range(0, k);
        } else {
            std::vector<std::thread> pool;
            pool.reserve((size_t) n_threads);
            for (int t = 0; t < n_threads; ++t) {
                const int lo = (int) ((long long) t       * k / n_threads);
                const int hi = (int) ((long long) (t + 1) * k / n_threads);
                if (lo < hi) {
                    pool.emplace_back(copy_range, lo, hi);
                }
            }
            for (auto & th : pool) {
                th.join();
            }
        }
    }

    // zero any unused tail slots [k, k_max): the remapped ids never point there,
    // but a device mul_mat_id kernel may still touch those expert rows, so they
    // must be valid (all-zero quant blocks decode to 0) rather than uninitialized.
    const int64_t k_max = dst->ne[2];
    if ((int64_t) k < k_max) {
        memset((char *) dst->data + (size_t) k * dst->nb[2], 0,
               (size_t) (k_max - k) * dst->nb[2]);
    }
}

void llama_moe_remap_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    (void) userdata;
    if (ith != 0) {
        return;
    }

    const int64_t n_used   = a->ne[0];
    const int64_t n_tokens = a->ne[1];
    const int64_t n        = n_used * n_tokens;

    std::vector<int32_t> flat;
    llama_moe_flatten_ids(a, flat);

    // bound the scratch table by the largest id present; slot ordering is
    // first-appearance based, so it is identical to the gather callback's.
    int n_expert = 1;
    for (int64_t p = 0; p < n; ++p) {
        if (flat[(size_t) p] + 1 > n_expert) {
            n_expert = flat[(size_t) p] + 1;
        }
    }

    std::vector<int32_t> slot_of_pos((size_t) n);
    std::vector<int32_t> expert_of_slot((size_t) n);
    llama_moe_compaction(flat.data(), n_used, n_tokens, n_expert,
                         slot_of_pos.data(), expert_of_slot.data());

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_used; ++e) {
            char * d = (char *) dst->data + t * dst->nb[1] + e * dst->nb[0];
            *(int32_t *) d = slot_of_pos[(size_t) (t * n_used + e)];
        }
    }
}

//
// persistent VRAM expert cache
//

struct llama_moe_cache_pool {
    ggml_tensor *         src    = nullptr;  // full [ne0,ne1,n_expert] CPU expert weights
    ggml_context *        ctx    = nullptr;  // owns the device cache tensor
    ggml_backend_buffer_t buffer = nullptr;  // device buffer
    ggml_tensor *         dev    = nullptr;  // [ne0,ne1,capacity] device cache tensor
    ggml_tensor *         slot_table = nullptr; // [1,n_expert] i32 device tensor: expert -> slot
    ggml_tensor *         sel_buf    = nullptr; // [n_used,n_tokens] i32: latest selection (graph writes)

    int      capacity  = 0;
    int      n_expert  = 0;
    int      ring_next = 0;                    // async ring-buffer eviction cursor
    uint64_t stride    = 0;                    // bytes per expert (src->nb[2])
    bool     async     = false;

    std::vector<int32_t>  slot_expert;        // capacity: expert id per slot, or -1
    std::vector<uint64_t> slot_age;           // capacity: LRU stamp
    std::vector<int32_t>  expert_slot;        // n_expert: slot per expert, or -1
    uint64_t              tick = 0;
};

static std::mutex g_moe_cache_mutex;
static std::unordered_map<const ggml_tensor *, llama_moe_cache_pool *> g_moe_cache_pools;

static ggml_backend_t llama_moe_pick_device_backend(ggml_backend_sched_t sched) {
    const int n = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n; ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t d = ggml_backend_get_device(b);
        if (d && ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return b;
        }
    }
    return nullptr;
}

static llama_moe_cache_pool * llama_moe_cache_get_or_create(ggml_backend_sched_t sched,
                                                            ggml_tensor *        exps,
                                                            int                  capacity,
                                                            bool                 async,
                                                            bool                 prefill) {
    std::lock_guard<std::mutex> lock(g_moe_cache_mutex);

    auto it = g_moe_cache_pools.find(exps);
    if (it != g_moe_cache_pools.end()) {
        return it->second;
    }

    ggml_backend_t backend = llama_moe_pick_device_backend(sched);
    if (!backend) {
        return nullptr;
    }

    const int n_expert = (int) exps->ne[2];
    if (capacity <= 0 || capacity > n_expert) {
        capacity = n_expert;
    }

    llama_moe_cache_pool * p = new llama_moe_cache_pool();
    p->src      = exps;
    p->n_expert = n_expert;
    p->stride   = (uint64_t) exps->nb[2];
    p->capacity = capacity;
    p->async    = async;
    p->slot_expert.assign((size_t) capacity, -1);
    p->slot_age.assign((size_t) capacity, 0);
    p->expert_slot.assign((size_t) n_expert, -1);

    struct ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, /*.no_alloc =*/ true };
    p->ctx = ggml_init(ip);
    p->dev = ggml_new_tensor_3d(p->ctx, exps->type, exps->ne[0], exps->ne[1], capacity);
    p->slot_table = ggml_new_tensor_2d(p->ctx, GGML_TYPE_I32, 1, n_expert);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    p->buffer = ggml_backend_alloc_ctx_tensors_from_buft(p->ctx, buft);
    if (!p->buffer) {
        ggml_free(p->ctx);
        delete p;
        return nullptr;
    }

    if (prefill && capacity == n_expert) {
        // load every expert into its own slot and set an identity slot table
        for (int e = 0; e < n_expert; ++e) {
            ggml_backend_tensor_set(p->dev,
                                    (const char *) exps->data + (size_t) e * p->stride,
                                    (size_t) e * p->stride, (size_t) p->stride);
            p->slot_expert[(size_t) e] = e;
            p->expert_slot[(size_t) e] = e;
        }
        std::vector<int32_t> ident((size_t) n_expert);
        for (int e = 0; e < n_expert; ++e) {
            ident[(size_t) e] = e;
        }
        ggml_backend_tensor_set(p->slot_table, ident.data(), 0, (size_t) n_expert * sizeof(int32_t));
    }

    g_moe_cache_pools[exps] = p;
    return p;
}

// map_custom1 callback: resolve each selected expert to a cache slot, loading misses
// into an LRU slot (synchronous H2D for now), and write the slot indices to dst.
static void llama_moe_cache_remap_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    llama_moe_cache_pool * p = (llama_moe_cache_pool *) userdata;

    const int64_t n_used   = a->ne[0];
    const int64_t n_tokens = a->ne[1];
    const int64_t n        = n_used * n_tokens;

    std::vector<int32_t> flat;
    llama_moe_flatten_ids(a, flat);

    std::lock_guard<std::mutex> lock(g_moe_cache_mutex);

    std::vector<char>    used((size_t) p->capacity, 0);
    std::vector<int32_t> pos_slot((size_t) n, -1);

    // pass 1: hits (experts already resident)
    for (int64_t pos = 0; pos < n; ++pos) {
        const int32_t e = flat[(size_t) pos];
        const int32_t slot = (e >= 0 && e < p->n_expert) ? p->expert_slot[(size_t) e] : -1;
        if (slot >= 0) {
            pos_slot[(size_t) pos] = slot;
            used[(size_t) slot] = 1;
            p->slot_age[(size_t) slot] = ++p->tick;
        }
    }
    // pass 2: misses -> load into an LRU slot not used by this token
    for (int64_t pos = 0; pos < n; ++pos) {
        if (pos_slot[(size_t) pos] >= 0) {
            continue;
        }
        const int32_t e = flat[(size_t) pos];
        if (e < 0 || e >= p->n_expert) {
            pos_slot[(size_t) pos] = 0;
            continue;
        }
        if (p->expert_slot[(size_t) e] >= 0) { // loaded by an earlier miss this call
            const int32_t slot = p->expert_slot[(size_t) e];
            pos_slot[(size_t) pos] = slot;
            used[(size_t) slot] = 1;
            p->slot_age[(size_t) slot] = ++p->tick;
            continue;
        }
        int      victim = -1;
        uint64_t best   = UINT64_MAX;
        for (int s = 0; s < p->capacity; ++s) {
            if (used[(size_t) s]) {
                continue;
            }
            if (p->slot_expert[(size_t) s] < 0) { victim = s; break; } // empty slot first
            if (p->slot_age[(size_t) s] < best) { best = p->slot_age[(size_t) s]; victim = s; }
        }
        if (victim < 0) {
            // capacity smaller than this token's unique experts: substitute a resident one
            pos_slot[(size_t) pos] = pos_slot[0] >= 0 ? pos_slot[0] : 0;
            continue;
        }
        if (p->slot_expert[(size_t) victim] >= 0) {
            p->expert_slot[(size_t) p->slot_expert[(size_t) victim]] = -1;
        }
        // synchronous load of expert e into the victim slot
        ggml_backend_tensor_set(p->dev,
                                (const char *) p->src->data + (size_t) e * p->stride,
                                (size_t) victim * p->stride,
                                (size_t) p->stride);
        p->slot_expert[(size_t) victim] = e;
        p->expert_slot[(size_t) e]      = victim;
        p->slot_age[(size_t) victim]    = ++p->tick;
        used[(size_t) victim]           = 1;
        pos_slot[(size_t) pos]          = victim;
    }

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t u = 0; u < n_used; ++u) {
            char * d = (char *) dst->data + t * dst->nb[1] + u * dst->nb[0];
            *(int32_t *) d = pos_slot[(size_t) (t * n_used + u)];
        }
    }
}

ggml_tensor * llama_moe_cache_build(ggml_context *       ctx0,
                                    ggml_backend_sched_t sched,
                                    ggml_tensor *        exps,
                                    ggml_tensor *        selected_experts,
                                    int                  capacity,
                                    bool                 async,
                                    ggml_tensor **       out_ids) {
    *out_ids = nullptr;
    capacity = llama_moe_clamp_capacity(capacity, (int) exps->ne[2], (int) selected_experts->ne[0]);
    llama_moe_cache_pool * p = llama_moe_cache_get_or_create(sched, exps, capacity, async, /*prefill=*/ false);
    if (!p) {
        return nullptr;
    }
    ggml_tensor * ids = ggml_map_custom1(ctx0, selected_experts, llama_moe_cache_remap_cb, 1, p);
    *out_ids = ids;
    return p->dev;
}

ggml_tensor * llama_moe_cache_build_getrows(ggml_context *       ctx0,
                                            ggml_backend_sched_t sched,
                                            ggml_tensor *        exps,
                                            ggml_tensor *        selected_experts,
                                            ggml_tensor **       out_ids) {
    *out_ids = nullptr;
    llama_moe_cache_pool * p = llama_moe_cache_get_or_create(sched, exps, /*capacity=*/ 0, /*async=*/ false, /*prefill=*/ true);
    if (!p || !p->slot_table) {
        return nullptr;
    }
    // remapped ids = slot_table[selected_experts], entirely on the compute device
    ggml_tensor * gr = ggml_get_rows(ctx0, p->slot_table, selected_experts); // [1, n_used, n_tokens]
    *out_ids = ggml_reshape_2d(ctx0, gr, selected_experts->ne[0], selected_experts->ne[1]);
    return p->dev;
}

//
// Per-layer async expert cache with drop-on-miss (the practical async path).
//
// One cache per MoE layer covers all its projection tensors, so an expert's residency
// is a single per-layer decision. A background loader keeps the hot experts resident;
// the device remap is a get_rows over a shared slot table. On a miss the expert is not
// swapped for a wrong one - the caller masks its routing weight to zero via
// resident_flag, gently dropping it. This stays coherent at small cache sizes.
//

struct llama_moe_proj_store {
    ggml_tensor * src    = nullptr; // host expert weights [ne0,ne1,n_expert]
    ggml_tensor * dev    = nullptr; // device cache [ne0,ne1,capacity]
    uint64_t      stride = 0;       // bytes per expert
};

struct llama_moe_layer_cache {
    int n_expert = 0;
    int capacity = 0;

    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_tensor * slot_table = nullptr; // [1,n_expert] i32: expert -> slot (sentinel `capacity` if missing)
    ggml_tensor * sel_buf    = nullptr; // [n_used,n_tokens] i32: latest selection

    std::vector<llama_moe_proj_store> proj;

    std::vector<int32_t>  slot_expert; // capacity: expert in slot, or -1
    std::vector<int32_t>  expert_slot; // n_expert: slot of expert, or -1
    std::vector<uint64_t> slot_age;    // capacity: LRU stamp
    int                   ring_next = 0; // grace-period eviction cursor
    uint64_t              tick = 0;
};

static std::mutex g_moe_layer_mutex;
static std::unordered_map<const ggml_tensor *, llama_moe_layer_cache *> g_moe_layer_caches;

static std::thread       g_moe_loader_thread;
static std::atomic<bool> g_moe_loader_run{false};

// background loader: keep each layer cache's hot experts resident. On a miss, evict the
// least-recently-used slot and load the expert into it across ALL projections, then
// publish slot_table and finally resident_flag. resident_flag is written LAST so the
// compute never treats an expert as resident before its slot and data are in place.
static void llama_moe_loader_main() {
    const bool no_loader = getenv("LLAMA_MOE_NOLOADER") != nullptr; // diagnostic: freeze cache at prewarm
    std::vector<int32_t> sel;
    while (g_moe_loader_run.load(std::memory_order_relaxed)) {
        if (no_loader) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
            for (auto & kv : g_moe_layer_caches) {
                llama_moe_layer_cache * c = kv.second;
                if (!c->sel_buf) {
                    continue;
                }
                const int64_t n = c->sel_buf->ne[0] * c->sel_buf->ne[1];
                sel.resize((size_t) n);
                ggml_backend_tensor_get(c->sel_buf, sel.data(), 0, (size_t) n * sizeof(int32_t));

                // pass 1: refresh LRU ages for experts already resident (so this token's
                // hits are never chosen as eviction victims for this token's misses)
                for (int64_t i = 0; i < n; ++i) {
                    const int32_t e = sel[(size_t) i];
                    if (e >= 0 && e < c->n_expert && c->expert_slot[(size_t) e] >= 0) {
                        c->slot_age[(size_t) c->expert_slot[(size_t) e]] = ++c->tick;
                    }
                }
                // pass 2: load misses into ring slots. A round-robin cursor gives every
                // freshly loaded slot a grace period of `capacity` loads before it can be
                // reused, so the compute never reads a slot that is being overwritten.
                for (int64_t i = 0; i < n; ++i) {
                    const int32_t e = sel[(size_t) i];
                    if (e < 0 || e >= c->n_expert || c->expert_slot[(size_t) e] >= 0) {
                        continue;
                    }
                    const int slot = c->ring_next;
                    c->ring_next   = (c->ring_next + 1) % c->capacity;
                    const int old_e = c->slot_expert[(size_t) slot];
                    // drop the outgoing expert first: point it at the zero sentinel slot so the
                    // compute stops using this slot before its data is overwritten
                    if (old_e >= 0 && old_e < c->n_expert) {
                        ggml_backend_tensor_set(c->slot_table, &c->capacity, (size_t) old_e * sizeof(int32_t), sizeof(int32_t));
                        c->expert_slot[(size_t) old_e] = -1;
                    }
                    // load the expert into this slot for every projection of the layer
                    for (auto & pr : c->proj) {
                        ggml_backend_tensor_set(pr.dev,
                                                (const char *) pr.src->data + (size_t) e * pr.stride,
                                                (size_t) slot * pr.stride, (size_t) pr.stride);
                    }
                    c->slot_expert[(size_t) slot] = e;
                    c->expert_slot[(size_t) e]    = slot;
                    c->slot_age[(size_t) slot]    = ++c->tick;
                    // activate last: publish the slot only after its data is in place, so the
                    // compute never sees this expert pointing at a half-written slot
                    ggml_backend_tensor_set(c->slot_table, &slot, (size_t) e * sizeof(int32_t), sizeof(int32_t));
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

static llama_moe_cache_pool * llama_moe_cache_get_or_create_async(ggml_backend_sched_t sched,
                                                                  ggml_tensor *        exps,
                                                                  ggml_tensor *        selected_experts,
                                                                  int                  capacity) {
    std::lock_guard<std::mutex> lock(g_moe_cache_mutex);

    auto it = g_moe_cache_pools.find(exps);
    if (it != g_moe_cache_pools.end()) {
        return it->second;
    }

    ggml_backend_t backend = llama_moe_pick_device_backend(sched);
    if (!backend) {
        return nullptr;
    }

    const int n_expert = (int) exps->ne[2];
    if (capacity <= 0 || capacity > n_expert) {
        capacity = n_expert;
    }

    llama_moe_cache_pool * p = new llama_moe_cache_pool();
    p->src      = exps;
    p->n_expert = n_expert;
    p->stride   = (uint64_t) exps->nb[2];
    p->capacity = capacity;
    p->async    = true;
    p->slot_expert.assign((size_t) capacity, -1);
    p->slot_age.assign((size_t) capacity, 0);
    p->expert_slot.assign((size_t) n_expert, -1);

    struct ggml_init_params ip = { 3 * ggml_tensor_overhead(), nullptr, /*.no_alloc =*/ true };
    p->ctx        = ggml_init(ip);
    p->dev        = ggml_new_tensor_3d(p->ctx, exps->type, exps->ne[0], exps->ne[1], capacity);
    p->slot_table = ggml_new_tensor_2d(p->ctx, GGML_TYPE_I32, 1, n_expert);
    p->sel_buf    = ggml_new_tensor_2d(p->ctx, GGML_TYPE_I32, selected_experts->ne[0], selected_experts->ne[1]);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    p->buffer = ggml_backend_alloc_ctx_tensors_from_buft(p->ctx, buft);
    if (!p->buffer) {
        ggml_free(p->ctx);
        delete p;
        return nullptr;
    }

    // prefill the cache with the first `capacity` experts and a fallback slot table so
    // that every expert resolves to some valid (if stale) slot from the very first token.
    for (int s = 0; s < capacity; ++s) {
        ggml_backend_tensor_set(p->dev,
                                (const char *) exps->data + (size_t) s * p->stride,
                                (size_t) s * p->stride, (size_t) p->stride);
        p->slot_expert[(size_t) s] = s;
        p->expert_slot[(size_t) s] = s;
    }
    std::vector<int32_t> table((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        table[(size_t) e] = (e < capacity) ? e : (e % capacity);
    }
    ggml_backend_tensor_set(p->slot_table, table.data(), 0, (size_t) n_expert * sizeof(int32_t));

    g_moe_cache_pools[exps] = p;

    if (!g_moe_loader_run.exchange(true)) {
        if (g_moe_loader_thread.joinable()) {
            g_moe_loader_thread.join();
        }
        g_moe_loader_thread = std::thread(llama_moe_loader_main);
    }
    return p;
}

ggml_tensor * llama_moe_cache_build_async(ggml_context *       ctx0,
                                          ggml_cgraph *        gf,
                                          ggml_backend_sched_t sched,
                                          ggml_tensor *        exps,
                                          ggml_tensor *        selected_experts,
                                          int                  capacity,
                                          ggml_tensor **       out_ids) {
    *out_ids = nullptr;
    llama_moe_cache_pool * p = llama_moe_cache_get_or_create_async(sched, exps, selected_experts, capacity);
    if (!p || !p->slot_table || !p->sel_buf) {
        return nullptr;
    }
    // publish the latest selection for the background loader (device-side copy)
    ggml_tensor * cpy = ggml_cpy(ctx0, selected_experts, p->sel_buf);
    ggml_build_forward_expand(gf, cpy);
    // remap on the device via the slot table (no CPU op in the critical path)
    ggml_tensor * gr = ggml_get_rows(ctx0, p->slot_table, selected_experts);
    *out_ids = ggml_reshape_2d(ctx0, gr, selected_experts->ne[0], selected_experts->ne[1]);
    return p->dev;
}

llama_moe_layer_cache * llama_moe_layer_cache_get(ggml_backend_sched_t sched,
                                                  ggml_tensor * const * exps_list,
                                                  int                   n_proj,
                                                  ggml_tensor *         selected_experts,
                                                  int                   capacity) {
    if (n_proj <= 0 || !exps_list || !exps_list[0]) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);

    const ggml_tensor * key = exps_list[0];
    auto it = g_moe_layer_caches.find(key);
    if (it != g_moe_layer_caches.end()) {
        return it->second;
    }

    ggml_backend_t backend = llama_moe_pick_device_backend(sched);
    if (!backend) {
        return nullptr;
    }

    const int n_expert = (int) exps_list[0]->ne[2];
    capacity = llama_moe_clamp_capacity(capacity, n_expert, (int) selected_experts->ne[0]);

    llama_moe_layer_cache * c = new llama_moe_layer_cache();
    c->n_expert = n_expert;
    c->capacity = capacity;
    c->slot_expert.assign((size_t) capacity, -1);
    c->slot_age.assign((size_t) capacity, 0);
    c->expert_slot.assign((size_t) n_expert, -1);

    const size_t n_tensors = 2 + (size_t) n_proj;
    struct ggml_init_params ip = { n_tensors * ggml_tensor_overhead(), nullptr, /*.no_alloc =*/ true };
    c->ctx           = ggml_init(ip);
    c->slot_table    = ggml_new_tensor_2d(c->ctx, GGML_TYPE_I32, 1, n_expert);
    // sel_buf holds one decode step's selection [n_used,1]; prefill warming loads directly
    c->sel_buf       = ggml_new_tensor_2d(c->ctx, GGML_TYPE_I32, selected_experts->ne[0], 1);
    c->proj.resize((size_t) n_proj);
    for (int i = 0; i < n_proj; ++i) {
        c->proj[(size_t) i].src    = exps_list[i];
        c->proj[(size_t) i].stride = (uint64_t) exps_list[i]->nb[2];
        // capacity resident slots + 1 zero sentinel slot (index `capacity`) for misses
        c->proj[(size_t) i].dev    = ggml_new_tensor_3d(c->ctx, exps_list[i]->type,
                                                        exps_list[i]->ne[0], exps_list[i]->ne[1], capacity + 1);
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    c->buffer = ggml_backend_alloc_ctx_tensors_from_buft(c->ctx, buft);
    if (!c->buffer) {
        ggml_free(c->ctx);
        delete c;
        return nullptr;
    }

    // zero the sentinel slot so a missing expert routed here contributes ~nothing
    // (all-zero k-quant blocks dequantize to 0), then prewarm slots 0..capacity-1 with
    // experts 0..capacity-1; the loader adapts to the real working set from there
    {
        std::vector<char> zeros((size_t) c->proj[0].stride, 0);
        for (auto & pr : c->proj) {
            zeros.resize((size_t) pr.stride, 0);
            ggml_backend_tensor_set(pr.dev, zeros.data(), (size_t) capacity * pr.stride, (size_t) pr.stride);
        }
    }
    for (int s = 0; s < capacity; ++s) {
        for (auto & pr : c->proj) {
            ggml_backend_tensor_set(pr.dev, (const char *) pr.src->data + (size_t) s * pr.stride,
                                    (size_t) s * pr.stride, (size_t) pr.stride);
        }
        c->slot_expert[(size_t) s] = s;
        c->expert_slot[(size_t) s] = s;
    }
    std::vector<int32_t> table((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        table[(size_t) e] = (e < capacity) ? e : capacity; // missing -> zero sentinel slot
    }
    ggml_backend_tensor_set(c->slot_table, table.data(), 0, (size_t) n_expert * sizeof(int32_t));

    // sel_buf starts invalid (-1) so the loader ignores it until decode publishes a selection
    {
        std::vector<int32_t> neg((size_t) c->sel_buf->ne[0] * (size_t) c->sel_buf->ne[1], -1);
        ggml_backend_tensor_set(c->sel_buf, neg.data(), 0, neg.size() * sizeof(int32_t));
    }

    g_moe_layer_caches[key] = c;

    if (!g_moe_loader_run.exchange(true)) {
        if (g_moe_loader_thread.joinable()) {
            g_moe_loader_thread.join();
        }
        g_moe_loader_thread = std::thread(llama_moe_loader_main);
    }
    return c;
}

ggml_tensor * llama_moe_layer_cache_slot_table(llama_moe_layer_cache * c) {
    return c ? c->slot_table : nullptr;
}

ggml_tensor * llama_moe_layer_cache_dev(llama_moe_layer_cache * c, const ggml_tensor * exps) {
    if (!c) {
        return nullptr;
    }
    for (auto & pr : c->proj) {
        if (pr.src == exps) {
            return pr.dev;
        }
    }
    return nullptr;
}

void llama_moe_layer_cache_publish(llama_moe_layer_cache * c,
                                   ggml_context *          ctx0,
                                   ggml_cgraph *           gf,
                                   ggml_tensor *           selected_experts) {
    if (!c || !c->sel_buf) {
        return;
    }
    // sel_buf is sized to the first (prefill) selection; a later/decode selection may be
    // narrower, so copy into a matching view of its leading columns. Skip if it cannot fit.
    if (selected_experts->ne[0] != c->sel_buf->ne[0] || selected_experts->ne[1] > c->sel_buf->ne[1]) {
        return;
    }
    ggml_tensor * dst = c->sel_buf;
    if (selected_experts->ne[1] != c->sel_buf->ne[1]) {
        dst = ggml_view_2d(ctx0, c->sel_buf, c->sel_buf->ne[0], selected_experts->ne[1], c->sel_buf->nb[1], 0);
    }
    ggml_tensor * cpy = ggml_cpy(ctx0, selected_experts, dst);
    ggml_build_forward_expand(gf, cpy);
}

// map_custom1 callback (prefill only): synchronously load the prompt's most-frequently
// selected experts into the cache, in the compute stream, so decode starts warm. This is
// the one-time "sync refresh" - it runs while no decode is in flight, so its cross-thread
// device writes are fully ordered before the decode graph reads the cache.
static void llama_moe_layer_warm_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    (void) dst;
    if (ith != 0) {
        return;
    }
    llama_moe_layer_cache * c = (llama_moe_layer_cache *) userdata;

    std::vector<int32_t> flat;
    llama_moe_flatten_ids(a, flat);

    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);

    // count how often each expert is selected across the prompt
    std::vector<int> freq((size_t) c->n_expert, 0);
    for (int32_t e : flat) {
        if (e >= 0 && e < c->n_expert) {
            freq[(size_t) e]++;
        }
    }
    // reset, then load the `capacity` most-frequent prompt experts into slots 0..k-1
    c->slot_expert.assign((size_t) c->capacity, -1);
    c->expert_slot.assign((size_t) c->n_expert, -1);
    std::vector<int32_t> table((size_t) c->n_expert, c->capacity); // all -> zero sentinel slot

    // pick the slot assignment first (cheap, serial), then do the expensive expert reads in
    // parallel below
    std::vector<int> load_e;
    std::vector<int> load_slot;
    load_e.reserve((size_t) c->capacity);
    load_slot.reserve((size_t) c->capacity);
    for (int slot = 0; slot < c->capacity; ++slot) {
        int best_e = -1;
        int best_f = 0;
        for (int e = 0; e < c->n_expert; ++e) {
            if (freq[(size_t) e] > best_f) {
                best_f = freq[(size_t) e];
                best_e = e;
            }
        }
        if (best_e < 0) {
            break; // no more selected experts to load
        }
        freq[(size_t) best_e] = 0; // consumed
        load_e.push_back(best_e);
        load_slot.push_back(slot);
        table[(size_t) best_e]          = slot;
        c->slot_expert[(size_t) slot]   = best_e;
        c->expert_slot[(size_t) best_e] = slot;
        c->slot_age[(size_t) slot]      = ++c->tick;
    }

    // Load the selected experts. Reads are issued in ascending file-offset (expert index)
    // order and split into contiguous per-thread ranges, so each thread streams the file
    // forward instead of seeking randomly - much faster on an SSD. Concurrent
    // ggml_backend_tensor_set is safe here (per-thread CUDA streams, distinct dst slots).
    {
        const size_t nl = load_e.size();
        std::vector<size_t> order(nl);
        for (size_t i = 0; i < nl; ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return load_e[a] < load_e[b]; });

        // hint the OS to stream the slabs in, in ascending-offset order
        for (size_t idx = 0; idx < nl; ++idx) {
            const int e = load_e[order[idx]];
            for (auto & pr : c->proj) {
                llama_moe_prefetch((const char *) pr.src->data + (size_t) e * pr.stride, pr.stride);
            }
        }

        const char * env = getenv("LLAMA_MOE_IO_THREADS");
        int n_threads = env ? atoi(env) : 16;
        if (n_threads < 1) { n_threads = 1; }
        if (n_threads > (int) nl) { n_threads = (int) nl; }

        auto load_range = [&](size_t lo, size_t hi) {
            for (size_t idx = lo; idx < hi; ++idx) {
                const int e    = load_e[order[idx]];
                const int slot = load_slot[order[idx]];
                for (auto & pr : c->proj) {
                    const char * src = (const char *) pr.src->data + (size_t) e * pr.stride;
                    ggml_backend_tensor_set(pr.dev, src, (size_t) slot * pr.stride, (size_t) pr.stride);
                }
            }
        };

        if (n_threads <= 1) {
            load_range(0, nl);
        } else {
            std::vector<std::thread> pool;
            pool.reserve((size_t) n_threads);
            for (int t = 0; t < n_threads; ++t) {
                const size_t lo = (size_t) ((long long) t       * (long long) nl / n_threads);
                const size_t hi = (size_t) ((long long) (t + 1) * (long long) nl / n_threads);
                if (lo < hi) {
                    pool.emplace_back(load_range, lo, hi);
                }
            }
            for (auto & th : pool) {
                th.join();
            }
        }
    }
    ggml_backend_tensor_set(c->slot_table, table.data(), 0, (size_t) c->n_expert * sizeof(int32_t));
}

void llama_moe_layer_cache_warm(llama_moe_layer_cache * c,
                                ggml_context *          ctx0,
                                ggml_cgraph *           gf,
                                ggml_tensor *           selected_experts) {
    if (!c) {
        return;
    }
    ggml_tensor * op = ggml_map_custom1(ctx0, selected_experts, llama_moe_layer_warm_cb, 1, c);
    ggml_build_forward_expand(gf, op);
}

void llama_moe_cache_shutdown(void) {
    // stop the background loader before the model/CUDA context is torn down, so it can
    // no longer touch freed expert weights or device buffers.
    if (g_moe_loader_run.exchange(false)) {
        if (g_moe_loader_thread.joinable()) {
            g_moe_loader_thread.join();
        }
    }
}
