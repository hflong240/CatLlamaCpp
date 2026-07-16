#include "llama-moe-stream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
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

// Positioned read of `bytes` at absolute `off` from a FILE* into `dst`. Uses each thread's OWN handle
// (no shared file position), so several io threads can read the same model file concurrently. A plain
// fseek+fread on a private handle is thread-safe because the position lives in that FILE*, not shared.
static inline bool llama_moe_pread(FILE * fp, uint64_t off, void * dst, size_t bytes) {
    if (!fp) { return false; }
    if (llama_moe_fseek64(fp, (int64_t) off, SEEK_SET) != 0) { return false; }
    return fread(dst, 1, bytes, fp) == bytes;
}

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

// Allocate a host buffer and lock it into physical RAM (never paged out), so reads from it never
// fault to disk. Returns nullptr on failure (caller falls back to mmap source). Used by the RAM
// residency tier to hold high-weight experts.
static void * llama_moe_ram_alloc(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
#ifdef _WIN32
    void * p = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) {
        return nullptr;
    }
    // best-effort lock; requires SeLockMemoryPrivilege for large sizes. If it fails the memory is
    // still committed/usable (just pageable) - reads still avoid the mmap file, only risk is a soft
    // page fault to the pagefile under pressure, far cheaper than the model-file random read.
    (void) VirtualLock(p, bytes);
    return p;
#else
    void * p = nullptr;
    if (posix_memalign(&p, 4096, bytes) != 0 || !p) {
        return nullptr;
    }
    (void) mlock(p, bytes);
    return p;
#endif
}

static void llama_moe_ram_free(void * p, size_t bytes) {
    if (!p) {
        return;
    }
#ifdef _WIN32
    (void) VirtualUnlock(p, bytes);
    VirtualFree(p, 0, MEM_RELEASE);
#else
    (void) munlock(p, bytes);
    free(p);
#endif
}

// Evict a range of the mmap'd model file from the OS page cache, so caching a high-weight expert
// in our locked RAM pool does not leave a duplicate copy squatting in page cache (which would
// double the memory footprint and, since the model >> RAM, thrash the cache). Best-effort.
static void llama_moe_evict_pagecache(const void * addr, size_t bytes) {
    if (!addr || bytes == 0) {
        return;
    }
#ifdef _WIN32
    // For a file-mapping view, VirtualUnlock on a non-locked range asks the OS to trim these pages
    // from the working set (documented behaviour); combined with the manager reclaiming clean
    // file-backed pages under pressure, this keeps the mapped expert bytes from pinning RAM.
    (void) VirtualUnlock(const_cast<void *>(addr), bytes);
#else
    // MADV_DONTNEED on a private/file mapping drops the resident pages; re-access re-faults from disk.
    (void) madvise(const_cast<void *>(addr), bytes, MADV_DONTNEED);
#endif
}

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

bool llama_moe_plan_remap(const int * freq,
                          int         n_expert,
                          const int32_t * expert_slot,
                          float       threshold,
                          int *       out_n_unique,
                          int *       out_n_miss,
                          int32_t *   out_ranked,
                          int *       out_n_ranked) {
    int n_unique = 0;
    int n_miss   = 0;
    for (int e = 0; e < n_expert; ++e) {
        if (freq[e] <= 0) {
            continue;
        }
        out_ranked[n_unique++] = e;
        if (expert_slot[e] < 0) {
            n_miss++;
        }
    }
    // rank all selected experts by descending selection frequency (ties: lower expert index,
    // for ascending-offset file reads). The caller keeps the top `capacity` of these resident
    // and drops the rest; ranking by frequency keeps the experts that matter most this step.
    std::sort(out_ranked, out_ranked + n_unique,
              [&](int32_t x, int32_t y) { return freq[x] != freq[y] ? freq[x] > freq[y] : x < y; });

    if (out_n_unique) { *out_n_unique = n_unique; }
    if (out_n_miss)   { *out_n_miss   = n_miss; }
    if (out_n_ranked) { *out_n_ranked = n_unique; }

    // sync when a large fraction of this step's unique experts is not resident (e.g. the cold
    // first step of a prompt); otherwise misses are dropped to the zero sentinel slot
    return n_unique > 0 && (float) n_miss > threshold * (float) n_unique;
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
    ggml_tensor * src    = nullptr; // host expert weights [ne0,ne1,n_expert] (mmap-backed, may page from disk)
    ggml_tensor * dev    = nullptr; // device cache [ne0,ne1,capacity]
    uint64_t      stride = 0;       // bytes per expert
    void *        ram    = nullptr; // optional RAM residency pool [ram_capacity * stride] (locked, never paged)
    FILE *        fp     = nullptr; // no-mmap: compute-thread file handle (read under g_moe_layer_mutex)
    FILE *        fp_ld  = nullptr; // no-mmap: SEPARATE loader-thread file handle (fill_ram reads unlocked;
                                    // a private handle so its file position never races the compute fp)
    uint64_t      foff0  = 0;       // absolute file offset of expert 0 (no-mmap)
    std::string   fpath;            // no-mmap: model file path (so parallel_load can open per-thread handles)
};

struct llama_moe_layer_cache {
    int n_expert  = 0;
    int capacity  = 0;
    int n_used    = 0;   // per-token expert activation
    int n_sentinel = 0;  // zero "sentinel" slots for dropped experts (<= n_used); see below

    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    // Pinned (page-locked) host staging arena for the CPU-remap sync path (LLAMA_MOE_PINNED_STAGE). One
    // stride-sized slot per (io-thread, projection) so a non-mmap expert read lands in pinned memory and
    // the following H2D runs at full PCIe rate instead of the pageable-memory rate. Allocated via the
    // backend-agnostic host-buffer type (real pinned memory when the device is CUDA; harmless plain host
    // memory otherwise). Sized on first use in parallel_load; freed at shutdown.
    ggml_backend_dev_t    stage_dev  = nullptr; // device whose host-buffer type backs the pinned arena
    ggml_backend_t        dev_backend = nullptr; // device backend handle (for async H2D + one synchronize)
    ggml_backend_buffer_t stage_buf  = nullptr; // owns the pinned bytes
    char *                stage_base = nullptr; // stage_buf base ptr
    size_t                stage_slot = 0;        // bytes per staging slot (== max proj stride)
    int                   stage_nthread = 0;     // io threads the arena is sized for

    ggml_tensor * slot_table = nullptr; // [1,n_expert] i32: expert -> slot (sentinel `capacity` if missing)
    ggml_tensor * sel_buf    = nullptr; // [n_used,n_tokens] i32: latest selection

    // Second device table for the pure-GPU decode remap. Unlike slot_table (which uses the `capacity`
    // sentinel = "drop to a zero slot" for non-resident experts), EVERY entry of stale_table points at
    // a real, settled slot in [0,capacity): a resident expert -> its own slot; a non-resident expert ->
    // a borrowed real (stale) slot. Decode's get_rows reads THIS table, so a miss reuses a real old
    // expert instead of zeroing the FFN branch - the same "stale reuse" that makes budget=2 coherent,
    // but expressed with no per-layer CPU op. slot_table keeps its sentinel semantics for the loader's
    // drop-first bookkeeping and the prefill CPU path. See llama_moe_boundary_sync / moe_stale_on_*.
    ggml_tensor * stale_table = nullptr; // [1,n_expert] i32: expert -> a real settled slot (never sentinel)
    std::vector<int32_t>  stale_of_expert; // n_expert: host mirror of stale_table (maintained under mutex)
    int                   last_settled_slot = 0; // most recently loaded slot; evict redirects stale here

    std::vector<llama_moe_proj_store> proj;

    std::vector<int32_t>  slot_expert; // capacity: expert in slot, or -1
    std::vector<int32_t>  expert_slot; // n_expert: slot of expert, or -1
    std::vector<uint64_t> slot_age;    // capacity: LRU stamp
    std::vector<uint64_t> slot_pin;    // capacity: pin generation (== pin_gen => in use by the current step)
    int                   ring_next = 0; // grace-period eviction cursor
    uint64_t              tick = 0;
    uint64_t              pin_gen = 0;   // current step's pin generation; bumped by each remap callback
    float                 sync_threshold = 0.5f; // miss fraction above which the remap sync-loads
    int                   sync_budget = 0; // >0: sync-load at most this many top-weight missing experts/step
    bool                  weighted_evict = false; // prefer evicting low-score slots (protect high-weight core)
    std::vector<float>    expert_score;  // n_expert: exp-decayed high-weight (top-2) usage score

    // Stale reuse (decode): each routing position u keeps the VRAM slot it last used. When position u's
    // expert misses the cache, instead of dropping to a zero sentinel we REUSE pos_slot[u] - a real
    // (stale) expert from a previous step - which perturbs the FFN far less than zeroing the branch.
    // Distinct positions keep distinct slots, so the n_used ids in a token stay unique (MMQ-safe).
    std::vector<int32_t>  pos_slot;      // n_used: the VRAM slot routing position u currently maps to
    bool                  pos_init = false; // pos_slot seeded yet (first real step seeds it)

    // RAM residency tier (LLAMA_MOE_RAM_CAP experts/layer): high-weight experts are copied into a
    // locked host pool so a VRAM miss reads from RAM (~25 GB/s) instead of faulting mmap from disk
    // (~1 GB/s). The model (170GB) exceeds RAM (64GB), and the OS page cache is plain LRU that cold
    // low-weight experts thrash, so we manage residency explicitly by expert_score. 0 => tier off.
    int                   ram_capacity = 0;
    std::vector<int32_t>  ram_slot;    // n_expert: RAM pool slot of expert, or -1
    std::vector<int32_t>  ram_expert;  // ram_capacity: expert in RAM slot, or -1
    std::vector<float>    ram_score;   // ram_capacity: score snapshot of the resident expert (for eviction)
    bool                  evict_after_ram = false; // drop an expert's mmap page-cache copy once it is in RAM
    std::vector<int32_t>  loader_prev_sel; // last selection the loader scored (to detect a new step)

    // decode miss-rate diagnostic (accumulated across steps, printed periodically)
    uint64_t              diag_steps = 0;
    uint64_t              diag_unique = 0;
    uint64_t              diag_miss = 0;
    std::vector<uint64_t> diag_top2; // n_expert: how often each expert appeared in a token's top-2
};

static std::mutex g_moe_layer_mutex;
static std::unordered_map<const ggml_tensor *, llama_moe_layer_cache *> g_moe_layer_caches;

// Registered on-disk location of each expert tensor (no-mmap takeover). exps -> (path, offset0).
struct llama_moe_expert_file { std::string path; uint64_t offset0 = 0; };
static std::unordered_map<const ggml_tensor *, llama_moe_expert_file> g_moe_expert_files;

void llama_moe_register_expert_file(const ggml_tensor * exps, const char * path, uint64_t file_offset) {
    if (!exps || !path) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    g_moe_expert_files[exps] = llama_moe_expert_file{ std::string(path), file_offset };
}

// Best source pointer for expert `e` of projection `pi`: the locked RAM pool if resident there
// (fast, never faults to disk), else the mmap-backed weights (may page from disk on first touch).
// Returns nullptr if the expert is not directly addressable (no-mmap mode with the expert only on
// disk) - callers that can stage through a buffer should use llama_moe_read_expert instead.
// Caller holds g_moe_layer_mutex.
static inline const char * llama_moe_layer_src(const llama_moe_layer_cache * c, size_t pi, int e) {
    const llama_moe_proj_store & pr = c->proj[pi];
    if (pr.ram && e >= 0 && e < c->n_expert) {
        const int32_t rs = c->ram_slot[(size_t) e];
        if (rs >= 0) {
            return (const char *) pr.ram + (size_t) rs * pr.stride;
        }
    }
    if (pr.fp) {
        return nullptr; // no-mmap: not directly addressable unless RAM-resident; use read_expert
    }
    return (const char *) pr.src->data + (size_t) e * pr.stride;
}

// Copy expert `e` of projection `pi` into `dst` (>= stride bytes). Source priority: locked RAM pool
// (fast), else own file handle via fread (no-mmap), else mmap-backed weights. Returns true on
// success. Caller holds g_moe_layer_mutex. This is the read path that works in every mode; use it
// wherever the source may only be on disk (no-mmap) rather than a stable host pointer.
static bool llama_moe_read_expert(const llama_moe_layer_cache * c, size_t pi, int e, void * dst) {
    const llama_moe_proj_store & pr = c->proj[pi];
    if (e < 0 || e >= c->n_expert) {
        return false;
    }
    if (pr.ram) {
        const int32_t rs = c->ram_slot[(size_t) e];
        if (rs >= 0) {
            memcpy(dst, (const char *) pr.ram + (size_t) rs * pr.stride, (size_t) pr.stride);
            return true;
        }
    }
    if (pr.fp) {
        if (llama_moe_fseek64(pr.fp, (int64_t) (pr.foff0 + (uint64_t) e * pr.stride), SEEK_SET) != 0) {
            return false;
        }
        return fread(dst, 1, (size_t) pr.stride, pr.fp) == (size_t) pr.stride;
    }
    memcpy(dst, (const char *) pr.src->data + (size_t) e * pr.stride, (size_t) pr.stride);
    return true;
}

// stale_table maintenance (caller holds g_moe_layer_mutex). The invariant is that every expert maps to
// a REAL slot in [0,capacity), so the pure-GPU decode get_rows never lands on a zero sentinel.
//   moe_stale_on_load : expert e just became resident in `slot` -> point its stale entry at its own slot.
//   moe_stale_on_evict: expert old_e is leaving `dying_slot`. We deliberately leave stale_of_expert[old_e]
//       pointing at dying_slot: it was set there by old_e's on_load, and after the caller overwrites the
//       slot with the incoming expert, old_e's stale entry naturally borrows that (real, hot) successor.
//       This spreads misses across all `capacity` slots' current occupants (each slot's evictees borrow
//       that slot's new expert) instead of collapsing every miss onto one expert - which is what destroys
//       the FFN signal and degenerates output. So on_evict is intentionally a NO-OP; kept as a call site
//       for symmetry/documentation and in case a future scheme needs an explicit redirect.
static inline void moe_stale_on_load(llama_moe_layer_cache * c, int e, int slot) {
    if (!c->stale_table || e < 0 || e >= c->n_expert) { return; }
    if ((int) c->stale_of_expert.size() != c->n_expert)  { return; }
    c->stale_of_expert[(size_t) e] = slot;
    ggml_backend_tensor_set(c->stale_table, &slot, (size_t) e * sizeof(int32_t), sizeof(int32_t));
}

static inline void moe_stale_on_evict(llama_moe_layer_cache * c, int old_e, int dying_slot) {
    // intentionally no-op: see the note above. old_e keeps pointing at dying_slot, which the caller is
    // about to fill with the incoming expert, so old_e borrows that real successor (diverse stale).
    (void) c; (void) old_e; (void) dying_slot;
}

// Load expert `e` into cache `slot` for every projection of the layer. Drops the outgoing occupant
// first (slot_table sentinel + stale redirect, so the compute stops using this slot before its bytes
// change), copies the expert bytes (RAM pool / no-mmap file / mmap, via the shared source helpers),
// then publishes the slot (slot_table + stale_table + bookkeeping). Caller holds g_moe_layer_mutex and
// has chosen an unpinned, settled `slot`. Shared by the background loader and the boundary sync so both
// keep slot_table AND stale_table consistent by construction.
//
// `staged`, when non-null, is an array of n_proj host pointers pre-read OUTSIDE the lock (the loader
// freads the slow disk bytes into private buffers before taking the mutex, so the lock-hold no longer
// includes the disk read - only the H2D + bookkeeping). A null `staged[pi]` (or a null `staged`) falls
// back to reading projection pi inline (RAM pool / mmap direct pointer / fread), the original behavior.
static void moe_layer_load_into_slot(llama_moe_layer_cache * c, int e, int slot, std::vector<char> & ldbuf,
                                     const char * const * staged = nullptr) {
    const int old_e = c->slot_expert[(size_t) slot];
    // drop the outgoing expert first: point it at the zero sentinel (slot_table) and at a safe settled
    // slot (stale_table) so the compute stops reading THIS slot before we overwrite its data
    if (old_e >= 0 && old_e < c->n_expert) {
        ggml_backend_tensor_set(c->slot_table, &c->capacity, (size_t) old_e * sizeof(int32_t), sizeof(int32_t));
        c->expert_slot[(size_t) old_e] = -1;
        moe_stale_on_evict(c, old_e, slot);
    }
    // load the expert into this slot for every projection. Prefer pre-staged bytes (disk read already
    // done outside the lock), then the RAM pool / mmap direct pointer, else fread inline.
    for (size_t pi = 0; pi < c->proj.size(); ++pi) {
        if (staged && staged[pi]) {
            ggml_backend_tensor_set(c->proj[pi].dev, staged[pi],
                                    (size_t) slot * c->proj[pi].stride, (size_t) c->proj[pi].stride);
            continue;
        }
        const char * sp = llama_moe_layer_src(c, pi, e);
        if (sp) {
            ggml_backend_tensor_set(c->proj[pi].dev, sp,
                                    (size_t) slot * c->proj[pi].stride, (size_t) c->proj[pi].stride);
        } else {
            ldbuf.resize((size_t) c->proj[pi].stride);
            if (llama_moe_read_expert(c, pi, e, ldbuf.data())) {
                ggml_backend_tensor_set(c->proj[pi].dev, ldbuf.data(),
                                        (size_t) slot * c->proj[pi].stride, (size_t) c->proj[pi].stride);
            }
        }
    }
    c->slot_expert[(size_t) slot] = e;
    c->expert_slot[(size_t) e]    = slot;
    c->slot_age[(size_t) slot]    = ++c->tick;
    // activate last: publish the slot only after its data is in place, so the compute never sees this
    // expert pointing at a half-written slot. stale_table now points e at its own real slot.
    ggml_backend_tensor_set(c->slot_table, &slot, (size_t) e * sizeof(int32_t), sizeof(int32_t));
    moe_stale_on_load(c, e, slot);
    c->last_settled_slot = slot;
}

static uint64_t g_moe_layer_vram_bytes = 0; // cumulative device bytes of all per-layer caches (under mutex)

// decode-path timing breakdown (under g_moe_layer_mutex); printed by the diag block, then reset.
static uint64_t g_diag_cb_ns    = 0; // total time inside the remap callback
static uint64_t g_diag_pl_ns    = 0; // of which, in parallel_load (disk fread + H2D of synced experts)
static uint64_t g_diag_sync_cnt = 0; // experts sync-loaded (per layer, summed)
static uint64_t g_diag_ram_hit  = 0; // of those, how many were served from the RAM pool (no disk)
static uint64_t g_diag_lockwait_ns = 0; // time the remap callback waited to ACQUIRE g_moe_layer_mutex

static std::thread       g_moe_loader_thread;
static std::atomic<bool> g_moe_loader_run{false};

// Fill each layer cache's locked RAM residency pool toward its highest-score experts, so VRAM
// misses read from RAM (~25 GB/s) instead of faulting the model file from disk. Runs as a separate
// loader phase so the slow part - reading an expert's bytes from disk - happens WITHOUT holding
// g_moe_layer_mutex (never stalls decode's publish or the CPU remap). Per call, promotes at most a
// bounded number of experts across all caches to keep each unlocked read window short.
//
// Safety of the lock-free RAM write: we take the mutex to (1) choose a target expert `e` and victim
// slot `v` whose current occupant we UNMAP from ram_slot first (so no ram_slot entry points at `v`
// while we overwrite it), then release the mutex and fread into `pr.ram + v*stride`; finally retake
// the mutex to publish ram_slot[e]=v. A concurrent reader only ever dereferences a RAM slot via
// ram_slot[expert] (see llama_moe_read_expert / llama_moe_layer_src), and no ram_slot maps to `v`
// during the unlocked write, so it never reads half-written bytes.
static void llama_moe_loader_fill_ram(std::vector<char> & scratch) {
    (void) scratch;
    const int fill_budget_env = getenv("LLAMA_MOE_RAM_FILL") ? atoi(getenv("LLAMA_MOE_RAM_FILL")) : 64;
    int fill_budget = fill_budget_env > 0 ? fill_budget_env : 64; // experts promoted per loader tick (all caches)

    // snapshot the cache list under the mutex (map itself is populated-once at graph build, but be safe)
    std::vector<llama_moe_layer_cache *> caches;
    {
        std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
        caches.reserve(g_moe_layer_caches.size());
        for (auto & kv : g_moe_layer_caches) { caches.push_back(kv.second); }
    }

    for (llama_moe_layer_cache * c : caches) {
        if (fill_budget <= 0) { break; }
        if (c->ram_capacity <= 0 || c->proj.empty() || !c->proj[0].ram) { continue; }

        // keep promoting this layer's best not-yet-RAM-resident expert until the pool holds the
        // top `ram_capacity` scorers (or we run out of per-tick budget)
        for (int per_layer = 0; per_layer < c->ram_capacity && fill_budget > 0; ++per_layer) {
            int   target_e = -1;
            int   victim   = -1;
            float target_sc = 0.0f;
            {
                std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
                if ((int) c->expert_score.size() != c->n_expert) { break; }
                // best-scoring expert not currently in the RAM pool. Prefer score, but if all scores
                // are ~0 (early, or weighted_evict off) still promote by index so the pool fills to
                // capacity - an empty RAM slot is always better than a disk fault.
                float best = -1.0f;
                for (int e = 0; e < c->n_expert; ++e) {
                    if (c->ram_slot[(size_t) e] >= 0) { continue; }
                    if (c->expert_score[(size_t) e] > best) { best = c->expert_score[(size_t) e]; target_e = e; }
                }
                if (target_e < 0) { break; } // every expert already RAM-resident
                target_sc = best;
                // pick a victim RAM slot: an empty one first (fill toward capacity regardless of
                // score), else the lowest LIVE-score occupant, and only if the incoming expert beats
                // it by a hysteresis margin (avoids two boundary experts evicting each other every
                // tick). Occupants are judged by their CURRENT expert_score, matching the VRAM tier.
                bool have_empty = false;
                float lowest = 3.4e38f;
                for (int r = 0; r < c->ram_capacity; ++r) {
                    if (c->ram_expert[(size_t) r] < 0) { victim = r; have_empty = true; break; }
                    const int32_t re = c->ram_expert[(size_t) r];
                    const float   rs = (re >= 0 && re < c->n_expert) ? c->expert_score[(size_t) re] : 0.0f;
                    if (rs < lowest) { lowest = rs; victim = r; }
                }
                if (!have_empty) {
                    // non-empty pool: require the incoming expert to clearly out-score the weakest
                    // occupant (10% margin) before paying a swap; otherwise the pool already holds the
                    // hot set and we stop for this layer.
                    if (victim < 0 || target_sc <= lowest * 1.1f) { break; }
                }
                // unmap the outgoing occupant BEFORE we release the lock and overwrite its bytes, so
                // no ram_slot entry points at `victim` during the unlocked write
                const int32_t old_e = c->ram_expert[(size_t) victim];
                if (old_e >= 0 && old_e < c->n_expert) { c->ram_slot[(size_t) old_e] = -1; }
                c->ram_expert[(size_t) victim] = -1; // slot is "being filled", owned by us, unreadable
            }

            // slow part, NO lock held: read the expert's bytes into the RAM pool slot for every proj
            bool ok = true;
            for (size_t pi = 0; pi < c->proj.size(); ++pi) {
                llama_moe_proj_store & pr = c->proj[pi];
                if (!pr.ram) { continue; }
                char * ramdst = (char *) pr.ram + (size_t) victim * pr.stride;
                if (pr.fp_ld) {
                    // use the loader's PRIVATE file handle - never the compute-thread pr.fp - so this
                    // unlocked seek+read cannot race the compute callback's read on a shared position
                    if (llama_moe_fseek64(pr.fp_ld, (int64_t) (pr.foff0 + (uint64_t) target_e * pr.stride), SEEK_SET) != 0 ||
                        fread(ramdst, 1, (size_t) pr.stride, pr.fp_ld) != (size_t) pr.stride) {
                        ok = false; break;
                    }
                } else {
                    memcpy(ramdst, (const char *) pr.src->data + (size_t) target_e * pr.stride, (size_t) pr.stride);
                    if (c->evict_after_ram) {
                        llama_moe_evict_pagecache((const char *) pr.src->data + (size_t) target_e * pr.stride, (size_t) pr.stride);
                    }
                }
            }

            // publish under the lock: the slot's bytes are in place, make it readable as `target_e`
            {
                std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
                if (ok) {
                    c->ram_expert[(size_t) victim] = target_e;
                    c->ram_score[(size_t) victim]  = target_sc;
                    c->ram_slot[(size_t) target_e] = victim; // publish last
                } else {
                    c->ram_expert[(size_t) victim] = -1; // leave slot empty on read failure
                }
            }
            fill_budget--;
        }
    }
}

// background loader: keep each layer cache's hot experts resident. On a miss, evict the
// least-recently-used slot and load the expert into it across ALL projections, then
// publish slot_table and finally resident_flag. resident_flag is written LAST so the
// compute never treats an expert as resident before its slot and data are in place.
static void llama_moe_loader_main() {
    const bool no_loader = getenv("LLAMA_MOE_NOLOADER") != nullptr; // diagnostic: freeze cache at prewarm
    std::vector<int32_t> sel;
    std::vector<char>    ldbuf; // scratch for reading an expert from file (no-mmap) before H2D
    std::vector<std::vector<char>> ldstage; // per-(miss,proj) staging buffers, pre-read outside the lock
    while (g_moe_loader_run.load(std::memory_order_relaxed)) {
        if (no_loader) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        // Snapshot the cache list under a brief registry lock, then process each cache under its OWN
        // acquisition of g_moe_layer_mutex, RELEASING the lock between caches. This bounds how long the
        // decode remap callback (which needs the same mutex per layer) waits on the loader to at most
        // ONE cache's work, instead of the whole 80-cache pass. The loader used to hold the global lock
        // across ALL caches' disk reads + H2D, so the per-layer callback stalled ~430ms/token behind it
        // (measured). Each per-cache critical section is unchanged - I/O still runs under the lock, so no
        // new torn-read / concurrent-H2D hazard is introduced; the lock is simply released 80x per pass
        // so the callback can interleave. Caches are add-only during a run (freed only at shutdown after
        // the loader is joined), so snapshotting the pointers is safe.
        std::vector<llama_moe_layer_cache *> ldcaches;
        {
            std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
            ldcaches.reserve(g_moe_layer_caches.size());
            for (auto & kv : g_moe_layer_caches) { ldcaches.push_back(kv.second); }
        }
        for (llama_moe_layer_cache * c : ldcaches) {
            // Phase A (under lock): score + age bookkeeping, and COLLECT this cache's miss experts.
            // No disk I/O here - just decide what to load. Release the lock at the end of this scope.
            std::vector<int> miss_e;      // loader-local; experts to load this pass (deduped)
            std::vector<char> miss_disk;  // parallel flag: 1 = fread from disk needed, 0 = RAM/inline
            {
                std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
                if (!c->sel_buf) {
                    continue;
                }
                const int64_t n = c->sel_buf->ne[0] * c->sel_buf->ne[1];
                sel.resize((size_t) n);
                ggml_backend_tensor_get(c->sel_buf, sel.data(), 0, (size_t) n * sizeof(int32_t));

                // pass 0: update weight-aware scores from the published selection. sel is argsort
                // output (position 0 = top-1 / highest gate weight), so weight positions decay by
                // 1/(rank+1). A slow global decay lets the stable high-weight core accumulate score;
                // eviction below then protects high-score slots so the core stays resident.
                if (c->weighted_evict && n > 0) {
                    if ((int) c->expert_score.size() != c->n_expert) { c->expert_score.assign((size_t) c->n_expert, 0.0f); }
                    bool changed = (sel != c->loader_prev_sel);
                    if (changed) {
                        for (float & s : c->expert_score) { s *= 0.98f; }
                        const int64_t n_used_sel = c->sel_buf->ne[0];
                        for (int64_t i = 0; i < n; ++i) {
                            const int32_t e = sel[(size_t) i];
                            if (e >= 0 && e < c->n_expert) {
                                const int64_t rank = i % (n_used_sel > 0 ? n_used_sel : 1);
                                c->expert_score[(size_t) e] += 1.0f / (float) (rank + 1);
                            }
                        }
                        c->loader_prev_sel = sel;
                    }
                }

                // pass 1: refresh LRU ages for experts already resident (so this token's
                // hits are never chosen as eviction victims for this token's misses)
                for (int64_t i = 0; i < n; ++i) {
                    const int32_t e = sel[(size_t) i];
                    if (e >= 0 && e < c->n_expert && c->expert_slot[(size_t) e] >= 0) {
                        c->slot_age[(size_t) c->expert_slot[(size_t) e]] = ++c->tick;
                    }
                }

                // pass 2a: collect this pass's miss experts (deduped). Flag whether each needs a disk
                // read (no-mmap file handle present AND not already in the RAM pool); those are pre-read
                // OUTSIDE the lock below so the slow fread no longer stalls the decode remap callback.
                for (int64_t i = 0; i < n; ++i) {
                    const int32_t e = sel[(size_t) i];
                    if (e < 0 || e >= c->n_expert || c->expert_slot[(size_t) e] >= 0) {
                        continue;
                    }
                    bool seen = false;
                    for (int me : miss_e) { if (me == e) { seen = true; break; } }
                    if (seen) { continue; }
                    const bool has_fp   = !c->proj.empty() && c->proj[0].fp_ld != nullptr;
                    const bool ram_res  = !c->ram_slot.empty() && c->ram_slot[(size_t) e] >= 0;
                    miss_e.push_back(e);
                    miss_disk.push_back((has_fp && !ram_res) ? 1 : 0);
                }
            } // lock released: the disk reads below run WITHOUT g_moe_layer_mutex held

            if (miss_e.empty()) { continue; }

            // Phase B (NO lock): pre-read the disk-bound misses' immutable weight bytes into private
            // staging buffers via the loader's OWN file handles (fp_ld), so the lock-hold in Phase C no
            // longer includes the slow fread. Expert bytes are immutable, staging buffers are loader-
            // local, and fp_ld is only touched by this thread - so this needs no lock. RAM/mmap misses
            // are left un-staged (staged ptr null -> Phase C reads them inline, which is fast).
            const size_t n_proj = c->proj.size();
            ldstage.resize(miss_e.size() * n_proj);
            for (size_t k = 0; k < miss_e.size(); ++k) {
                if (!miss_disk[k]) { continue; }
                const int e = miss_e[k];
                for (size_t pi = 0; pi < n_proj; ++pi) {
                    llama_moe_proj_store & pr = c->proj[pi];
                    std::vector<char> & buf = ldstage[k * n_proj + pi];
                    buf.resize((size_t) pr.stride);
                    if (pr.fp_ld && llama_moe_fseek64(pr.fp_ld, (int64_t) (pr.foff0 + (uint64_t) e * pr.stride), SEEK_SET) == 0) {
                        if (fread(buf.data(), 1, (size_t) pr.stride, pr.fp_ld) != (size_t) pr.stride) {
                            miss_disk[k] = 0; // read failed: fall back to inline read under the lock
                        }
                    } else {
                        miss_disk[k] = 0;
                    }
                }
            }

            // Phase C (under lock): publish each still-missing expert into a ring slot, feeding the
            // pre-staged bytes when available (H2D + bookkeeping only - no disk read under the lock).
            {
                std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
                std::vector<const char *> staged(n_proj, nullptr);
                for (size_t k = 0; k < miss_e.size(); ++k) {
                    const int32_t e = miss_e[k];
                    if (e < 0 || e >= c->n_expert || c->expert_slot[(size_t) e] >= 0) {
                        continue; // became resident (e.g. the callback loaded it) since Phase A
                    }
                    const float in_score = (c->weighted_evict && (int) c->expert_score.size() == c->n_expert)
                                            ? c->expert_score[(size_t) e] : 0.0f;
                    int slot = -1;
                    for (int tries = 0; tries < c->capacity; ++tries) {
                        const int cand = c->ring_next;
                        c->ring_next   = (c->ring_next + 1) % c->capacity;
                        if (c->slot_pin[(size_t) cand] == c->pin_gen) {
                            continue; // pinned by the current step, never overwrite
                        }
                        if (c->weighted_evict) {
                            const int32_t se = c->slot_expert[(size_t) cand];
                            const float   sc = (se >= 0 && se < c->n_expert && (int) c->expert_score.size() == c->n_expert)
                                                ? c->expert_score[(size_t) se] : 0.0f;
                            if (se >= 0 && sc > in_score && tries < c->capacity - 1) {
                                continue;
                            }
                        }
                        slot = cand;
                        break;
                    }
                    if (slot < 0) {
                        break; // every slot pinned by the current step: leave misses dropped for now
                    }
                    const char * const * staged_ptr = nullptr;
                    if (miss_disk[k]) {
                        for (size_t pi = 0; pi < n_proj; ++pi) { staged[pi] = ldstage[k * n_proj + pi].data(); }
                        staged_ptr = staged.data();
                    }
                    moe_layer_load_into_slot(c, e, slot, ldbuf, staged_ptr);
                }
            }
        }
        // RAM residency tier fill (separate phase, outside the per-cache lock above). Keep the
        // highest-score experts resident in each layer's locked host pool so VRAM misses read from
        // RAM (~25 GB/s) instead of faulting the model file from disk. The slow part - the fread of
        // an expert's bytes - is done WITHOUT holding g_moe_layer_mutex (so it never stalls decode's
        // publish or the CPU remap); the mutex is taken only briefly to pick a target and to publish.
        llama_moe_loader_fill_ram(ldbuf);
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
    // The graph is built once during the sched_reserve/graph_reserve measurement pass, before
    // the model's expert weights are populated (with mmap=false the host buffers are filled by
    // load_tensors only after reserve). Creating the cache now would prewarm-copy from an
    // unbacked source pointer (CUDA "invalid argument" in ggml_backend_tensor_set). Defer until
    // the source data is resident; the caller falls back to the compaction gather for the pass.
    for (int i = 0; i < n_proj; ++i) {
        if (!exps_list[i] || !exps_list[i]->data) {
            return nullptr;
        }
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
    const int n_used   = (int) selected_experts->ne[0];
    capacity = llama_moe_clamp_capacity(capacity, n_expert, n_used);

    // Number of physical zero sentinel slots for dropped experts. Each costs one full expert slab of
    // VRAM per layer, so keep it small: 1 by default. A token with more drops than this reuses spare
    // (unused-this-token) resident slots for the overflow, staying MMQ-safe. Bounded to [1, n_used];
    // raise toward n_used only if the wrong-but-in-bounds overflow contributions hurt quality.
    int n_sentinel = 1;
    if (const char * s_env = getenv("LLAMA_MOE_SENTINELS")) {
        n_sentinel = atoi(s_env);
    }
    if (n_sentinel < 1)      { n_sentinel = 1; }
    if (n_sentinel > n_used) { n_sentinel = n_used; }

    llama_moe_layer_cache * c = new llama_moe_layer_cache();
    c->n_expert   = n_expert;
    c->capacity   = capacity;
    c->n_used     = n_used;
    c->n_sentinel = n_sentinel;
    // Weight-aware residency: the background loader protects high-score (recurring top-1/top-2)
    // experts from eviction so the stable high-weight core stays resident on the pure-GPU decode
    // path (rare misses drop to the sentinel). On by default; disable with LLAMA_MOE_WEIGHTED_EVICT=0.
    c->weighted_evict = true;
    if (const char * w_env = getenv("LLAMA_MOE_WEIGHTED_EVICT")) { c->weighted_evict = atoi(w_env) != 0; }
    c->slot_expert.assign((size_t) capacity, -1);
    c->slot_age.assign((size_t) capacity, 0);
    c->slot_pin.assign((size_t) capacity, 0);
    c->expert_slot.assign((size_t) n_expert, -1);
    c->expert_score.assign((size_t) n_expert, 0.0f);
    // seed each routing position u with a distinct real prewarm slot (slot u holds expert u after
    // prewarm below), so stale reuse starts from real experts and the n_used positions never collide.
    c->pos_slot.assign((size_t) n_used, 0);
    for (int u = 0; u < n_used; ++u) { c->pos_slot[(size_t) u] = (u < capacity) ? u : 0; }
    c->pos_init = false;

    // capacity resident slots + n_sentinel zero sentinel slots [capacity .. capacity+n_sentinel)
    const int n_slots = capacity + n_sentinel;

    const size_t n_tensors = 3 + (size_t) n_proj;
    struct ggml_init_params ip = { n_tensors * ggml_tensor_overhead(), nullptr, /*.no_alloc =*/ true };
    c->ctx           = ggml_init(ip);
    c->slot_table    = ggml_new_tensor_2d(c->ctx, GGML_TYPE_I32, 1, n_expert);
    c->stale_table   = ggml_new_tensor_2d(c->ctx, GGML_TYPE_I32, 1, n_expert);
    // sel_buf holds one decode step's selection [n_used,1]; prefill warming loads directly
    c->sel_buf       = ggml_new_tensor_2d(c->ctx, GGML_TYPE_I32, selected_experts->ne[0], 1);
    c->proj.resize((size_t) n_proj);
    for (int i = 0; i < n_proj; ++i) {
        c->proj[(size_t) i].src    = exps_list[i];
        c->proj[(size_t) i].stride = (uint64_t) exps_list[i]->nb[2];
        c->proj[(size_t) i].dev    = ggml_new_tensor_3d(c->ctx, exps_list[i]->type,
                                                        exps_list[i]->ne[0], exps_list[i]->ne[1], n_slots);
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    c->buffer = ggml_backend_alloc_ctx_tensors_from_buft(c->ctx, buft);
    if (!c->buffer) {
        ggml_free(c->ctx);
        delete c;
        return nullptr;
    }
    // remember the device so parallel_load can allocate a pinned host staging arena from its
    // host-buffer type (backend-agnostic; real pinned memory on CUDA, plain host memory elsewhere)
    c->stage_dev = ggml_backend_get_device(backend);
    c->dev_backend = backend; // for async H2D on the device stream + a single synchronize per load batch

    // no-mmap takeover: if enabled and every projection has a registered file location, open a
    // private FILE* per projection and read experts by offset (fread) instead of from the mmap
    // `src->data`. This keeps the OS page cache out of the expert path entirely - the only expert
    // bytes in RAM are the ones we deliberately hold in the locked pool. Falls back to mmap if any
    // projection lacks registration.
    if (getenv("LLAMA_MOE_NOMMAP")) {
        bool all_registered = true;
        for (int i = 0; i < n_proj; ++i) {
            auto fit = g_moe_expert_files.find(exps_list[i]);
            if (fit == g_moe_expert_files.end()) { all_registered = false; break; }
        }
        if (all_registered) {
            bool ok = true;
            for (int i = 0; i < n_proj && ok; ++i) {
                const llama_moe_expert_file & ef = g_moe_expert_files[exps_list[i]];
                FILE * fp = fopen(ef.path.c_str(), "rb");
                if (!fp) { ok = false; break; }
                // separate handle for the background loader's fill_ram (it reads outside the mutex,
                // so it must not share a file position with the compute-thread fp). Optional: if the
                // second open fails, fill_ram simply won't run for this layer (no correctness issue).
                FILE * fp_ld = fopen(ef.path.c_str(), "rb");
                c->proj[(size_t) i].fp     = fp;
                c->proj[(size_t) i].fp_ld  = fp_ld;
                c->proj[(size_t) i].foff0  = ef.offset0;
                c->proj[(size_t) i].fpath  = ef.path; // for per-thread handles in parallel_load
            }
            if (ok) {
                static bool logged = false;
                if (!logged) { logged = true;
                    LLAMA_LOG_INFO("MoE stream: no-mmap takeover active - experts read from the model file "
                                   "by offset, bypassing the OS page cache.\n");
                }
            } else {
                for (int i = 0; i < n_proj; ++i) {
                    if (c->proj[(size_t) i].fp)    { fclose(c->proj[(size_t) i].fp);    c->proj[(size_t) i].fp    = nullptr; }
                    if (c->proj[(size_t) i].fp_ld) { fclose(c->proj[(size_t) i].fp_ld); c->proj[(size_t) i].fp_ld = nullptr; }
                }
                LLAMA_LOG_WARN("MoE stream: no-mmap takeover requested but file open failed; using mmap.\n");
            }
        } else {
            LLAMA_LOG_WARN("MoE stream: LLAMA_MOE_NOMMAP set but expert file offsets not registered; "
                           "using mmap. (Is the model loaded with expert-file registration?)\n");
        }
    }

    // high-weight experts that miss the VRAM cache are read from RAM instead of faulting mmap from
    // disk. LLAMA_MOE_RAM_CAP = experts/layer to hold in RAM (0/unset => tier off). Clamp to n_expert.
    if (const char * rc_env = getenv("LLAMA_MOE_RAM_CAP")) {
        int rc = atoi(rc_env);
        if (rc > n_expert) { rc = n_expert; }
        if (rc > 0) {
            bool ok = true;
            for (int i = 0; i < n_proj && ok; ++i) {
                c->proj[(size_t) i].ram = llama_moe_ram_alloc((size_t) rc * (size_t) c->proj[(size_t) i].stride);
                if (!c->proj[(size_t) i].ram) { ok = false; }
            }
            if (ok) {
                c->ram_capacity = rc;
                c->ram_slot.assign((size_t) n_expert, -1);
                c->ram_expert.assign((size_t) rc, -1);
                c->ram_score.assign((size_t) rc, 0.0f);
                // default on: once an expert is in the locked RAM pool, evict its mmap page-cache
                // copy so both do not occupy RAM. Disable with LLAMA_MOE_RAM_EVICT=0.
                c->evict_after_ram = true;
                if (const char * ev = getenv("LLAMA_MOE_RAM_EVICT")) { c->evict_after_ram = atoi(ev) != 0; }
            } else {
                // partial alloc failed: free whatever we got, run without the RAM tier
                for (int i = 0; i < n_proj; ++i) {
                    if (c->proj[(size_t) i].ram) {
                        llama_moe_ram_free(c->proj[(size_t) i].ram, (size_t) rc * (size_t) c->proj[(size_t) i].stride);
                        c->proj[(size_t) i].ram = nullptr;
                    }
                }
                LLAMA_LOG_WARN("MoE stream: RAM residency pool alloc failed (wanted %d experts/layer); "
                               "running without it (misses fault mmap from disk)\n", rc);
            }
        }
    }

    // Track cumulative cache VRAM and warn if it likely won't fit: an overflowing cache does not OOM
    // here - the driver silently spills it to system RAM (CUDA "system memory fallback"), which makes
    // decode PCIe-bound and ~10x slower. That looks like a mysterious slowdown, so surface it loudly.
    {
        g_moe_layer_vram_bytes += (uint64_t) ggml_backend_buffer_get_size(c->buffer);
        size_t dev_free = 0, dev_total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &dev_free, &dev_total);
        const double gib = 1024.0 * 1024.0 * 1024.0;
        const double per_layer_gib = (double) ggml_backend_buffer_get_size(c->buffer) / gib;
        static bool logged = false;
        if (!logged) {
            logged = true;
            LLAMA_LOG_INFO("MoE stream: per-layer expert cache = %.2f GiB (capacity=%d + %d sentinel slots); "
                           "device free %.1f / %.1f GiB. Cache grows per MoE layer - watch the total below.\n",
                           per_layer_gib, capacity, n_sentinel, dev_free / gib, dev_total / gib);
        }
        static bool warned = false;
        if (!warned && dev_total > 0 && g_moe_layer_vram_bytes > (uint64_t) (0.90 * (double) dev_total)) {
            warned = true;
            LLAMA_LOG_WARN("MoE stream: expert cache reached %.1f GiB, approaching device capacity (%.1f GiB); "
                           "if it overflows, the driver spills to system RAM and decode becomes PCIe-bound "
                           "(very slow, looks like a ~10x slowdown). Lower LLAMA_MOE_CACHE_CAP - each cap unit "
                           "is ~%.2f GiB across all layers built so far.\n",
                           g_moe_layer_vram_bytes / gib, dev_total / gib,
                           per_layer_gib / (double) (capacity + n_sentinel));
        }
    }

    // zero the n_sentinel sentinel slots [capacity, capacity+n_sentinel) so a dropped expert routed to
    // any of them contributes ~nothing (all-zero k-quant blocks dequantize to 0), then prewarm slots
    // 0..capacity-1 with experts 0..capacity-1; the loader adapts to the real working set from there
    {
        std::vector<char> zeros;
        for (auto & pr : c->proj) {
            zeros.assign((size_t) pr.stride, 0);
            for (int s = capacity; s < n_slots; ++s) {
                ggml_backend_tensor_set(pr.dev, zeros.data(), (size_t) s * pr.stride, (size_t) pr.stride);
            }
        }
    }
    {
        std::vector<char> prewarm;
        for (int s = 0; s < capacity; ++s) {
            for (size_t pi = 0; pi < c->proj.size(); ++pi) {
                const char * sp = llama_moe_layer_src(c, pi, s);
                if (sp) {
                    ggml_backend_tensor_set(c->proj[pi].dev, sp, (size_t) s * c->proj[pi].stride, (size_t) c->proj[pi].stride);
                } else {
                    prewarm.resize((size_t) c->proj[pi].stride);
                    if (llama_moe_read_expert(c, pi, s, prewarm.data())) {
                        ggml_backend_tensor_set(c->proj[pi].dev, prewarm.data(), (size_t) s * c->proj[pi].stride, (size_t) c->proj[pi].stride);
                    }
                }
            }
            c->slot_expert[(size_t) s] = s;
            c->expert_slot[(size_t) s] = s;
        }
    }
    std::vector<int32_t> table((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        table[(size_t) e] = (e < capacity) ? e : capacity; // missing -> zero sentinel slot
    }
    ggml_backend_tensor_set(c->slot_table, table.data(), 0, (size_t) n_expert * sizeof(int32_t));

    // stale_table: every expert maps to a REAL slot (never the sentinel). Resident experts (e<capacity,
    // prewarmed into slot e) map to their own slot; the rest borrow slot (e%capacity), which after
    // prewarm holds a real expert. Decode's get_rows reads this, so a miss reuses a real (stale) expert
    // instead of zeroing the branch. The loader/boundary sync keep it current via moe_stale_on_*.
    c->stale_of_expert.assign((size_t) n_expert, 0);
    for (int e = 0; e < n_expert; ++e) {
        c->stale_of_expert[(size_t) e] = (e < capacity) ? e : (capacity > 0 ? e % capacity : 0);
    }
    ggml_backend_tensor_set(c->stale_table, c->stale_of_expert.data(), 0, (size_t) n_expert * sizeof(int32_t));
    c->last_settled_slot = 0;

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

ggml_tensor * llama_moe_layer_cache_stale_table(llama_moe_layer_cache * c) {
    return c ? c->stale_table : nullptr;
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

// Ensure the cache's pinned host staging arena is allocated for `n_threads` io threads (one stride-sized
// slot per (thread, projection)). Backend-agnostic: uses the device's host-buffer type, which is real
// page-locked memory on CUDA. Returns the base ptr and per-slot stride via the cache fields, or leaves
// stage_base=nullptr on failure (caller falls back to a pageable heap buffer). Caller holds the mutex.
static void llama_moe_ensure_stage(llama_moe_layer_cache * c, int n_threads) {
    if (!c->stage_dev || n_threads < 1) { return; }
    // per-slot bytes = max projection stride (so any proj's expert fits)
    size_t slot = 0;
    for (auto & pr : c->proj) { if (pr.stride > slot) { slot = (size_t) pr.stride; } }
    if (slot == 0) { return; }
    const size_t n_slots = (size_t) n_threads * c->proj.size();
    if (c->stage_base && c->stage_slot >= slot && c->stage_nthread >= n_threads) {
        return; // already big enough
    }
    if (c->stage_buf) { ggml_backend_buffer_free(c->stage_buf); c->stage_buf = nullptr; c->stage_base = nullptr; }
    ggml_backend_buffer_type_t hbuft = ggml_backend_dev_host_buffer_type(c->stage_dev);
    if (!hbuft) { return; } // device has no pinned host buffer type; caller uses pageable fallback
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(hbuft, n_slots * slot);
    if (!buf) { return; }
    c->stage_buf     = buf;
    c->stage_base    = (char *) ggml_backend_buffer_get_base(buf);
    c->stage_slot    = slot;
    c->stage_nthread = n_threads;
}


// into contiguous per-thread ranges, so each thread streams the file forward instead of seeking
// randomly - much faster on an SSD. Concurrent ggml_backend_tensor_set is safe: per-thread CUDA
// streams, distinct dst slots. Synchronous H2D (see the ordering contract on the remap callback
// below), so all copies have landed on return.
//
// Two opt-in accelerations of this 97%-of-callback-time hot path (LLAMA_MOE_PINNED_STAGE, default on):
//  - PINNED STAGING: no-mmap expert bytes are read into a page-locked host arena, so the following H2D
//    runs at full PCIe rate instead of the pageable-memory rate.
//  - PARALLEL no-mmap READS: each io thread opens its OWN FILE* on the model file and uses positioned
//    reads, so no-mmap mode is no longer forced single-threaded (the old shared-FILE* seek race is gone).
// Output is byte-identical either way (same experts, same bytes, faster transport).
static void llama_moe_layer_parallel_load(llama_moe_layer_cache *   c,
                                          const std::vector<int> &  load_e,
                                          const std::vector<int> &  load_slot) {
    const size_t nl = load_e.size();
    if (nl == 0) {
        return;
    }
    std::vector<size_t> order(nl);
    for (size_t i = 0; i < nl; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return load_e[a] < load_e[b]; });

    // hint the OS to stream the slabs in, in ascending-offset order (skip experts already in the RAM
    // pool, and skip entirely in no-mmap mode where we never touch the mmap file)
    for (size_t idx = 0; idx < nl; ++idx) {
        const int e = load_e[order[idx]];
        for (size_t pi = 0; pi < c->proj.size(); ++pi) {
            if (!c->proj[pi].fp && !(c->proj[pi].ram && c->ram_slot[(size_t) e] >= 0)) {
                llama_moe_prefetch((const char *) c->proj[pi].src->data + (size_t) e * c->proj[pi].stride, c->proj[pi].stride);
            }
        }
    }

    const char * env = getenv("LLAMA_MOE_IO_THREADS");
    int n_threads = env ? atoi(env) : 16;
    if (n_threads < 1) { n_threads = 1; }
    if (n_threads > (int) nl) { n_threads = (int) nl; }

    // Fast-path accelerations (LLAMA_MOE_PINNED_STAGE, default on; set =0 to A/B against the old path).
    //  - parallel no-mmap reads: each thread opens its own FILE* and uses positioned reads, so no-mmap
    //    is no longer forced single-threaded (the shared-FILE* seek race that required n_threads=1 is
    //    avoided by private handles).
    //  - pinned staging: the per-thread read buffer is page-locked host memory so the H2D runs at full
    //    PCIe rate. Falls back to a pageable heap buffer if the pinned arena can't be allocated.
    bool fast = true;
    if (const char * fe = getenv("LLAMA_MOE_PINNED_STAGE")) { fast = atoi(fe) != 0; }

    bool any_fp = false;
    for (auto & pr : c->proj) { if (pr.fp) { any_fp = true; break; } }

    if (any_fp && !fast) {
        // legacy no-mmap: shared FILE* per projection, seek+read not thread-safe -> single thread
        n_threads = 1;
    }

    // pinned staging arena (one stride slot per thread*proj); if it fails, threads use a pageable buffer
    if (fast) { llama_moe_ensure_stage(c, n_threads); }
    const bool use_pinned = fast && c->stage_base != nullptr;

    // Async-H2D batching (default on): the old path issued a synchronous ggml_backend_tensor_set per
    // expert-projection, and the CUDA backend does cudaMemcpy + cudaStreamSynchronize on EVERY call
    // (ggml-cuda.cu set_tensor). At ~48 experts x 3 proj = ~144 H2D/token that per-call synchronize was
    // ~150-290ms/token of pure fixed overhead (measured), dwarfing the ~40ms of actual bytes. Instead we
    // record each landed read as a pending H2D and, after all reads complete, issue them as
    // ggml_backend_tensor_set_async on the device stream followed by ONE ggml_backend_synchronize. The
    // caller (map_custom callback) still returns only after that single sync, so the ordering contract
    // (experts resident before the downstream matmul) is preserved. Set LLAMA_MOE_ASYNC_H2D=0 to disable.
    static const bool async_h2d = []() {
        const char * e = getenv("LLAMA_MOE_ASYNC_H2D");
        return !e || atoi(e) != 0; // default on
    }() && c->dev_backend != nullptr;

    struct pending_h2d { ggml_tensor * dev; size_t off; const char * src; size_t bytes; };
    std::vector<pending_h2d> pend;
    std::mutex pend_mu;

    auto load_range = [&](size_t lo, size_t hi, int tid) {
        std::vector<char> scratch;                 // pageable fallback when no pinned arena
        std::vector<pending_h2d> local;            // this thread's landed H2Ds (async mode)
        std::vector<FILE *> tfp;                   // this thread's private no-mmap handles (per proj)
        const bool par_read = fast && any_fp;      // this thread does its own file reads
        if (par_read) {
            tfp.resize(c->proj.size(), nullptr);
            for (size_t pi = 0; pi < c->proj.size(); ++pi) {
                if (!c->proj[pi].fpath.empty()) { tfp[pi] = fopen(c->proj[pi].fpath.c_str(), "rb"); }
            }
        }
        for (size_t idx = lo; idx < hi; ++idx) {
            const int e    = load_e[order[idx]];
            const int slot = load_slot[order[idx]];
            for (size_t pi = 0; pi < c->proj.size(); ++pi) {
                const llama_moe_proj_store & pr = c->proj[pi];
                const size_t stride = (size_t) pr.stride;
                // Prefer a directly-addressable source (RAM pool or mmap): H2D straight from it.
                const char * sp = llama_moe_layer_src(c, pi, e);
                if (sp) {
                    if (async_h2d) { local.push_back({pr.dev, (size_t) slot * stride, sp, stride}); }
                    else { ggml_backend_tensor_set(pr.dev, sp, (size_t) slot * stride, stride); }
                    continue;
                }
                // no-mmap: read expert bytes into a (pinned, if available) staging slot, then H2D.
                char * stg = nullptr;
                if (use_pinned) {
                    const size_t si = ((size_t) tid * c->proj.size() + pi); // unique per (thread,proj)
                    stg = c->stage_base + si * c->stage_slot;
                } else {
                    scratch.resize(stride);
                    stg = scratch.data();
                }
                bool ok;
                if (par_read && tfp[pi]) {
                    ok = llama_moe_pread(tfp[pi], pr.foff0 + (uint64_t) e * stride, stg, stride);
                } else {
                    // single-thread / no private handle: use the shared read path (holds compute fp)
                    ok = llama_moe_read_expert(c, pi, e, stg);
                }
                if (ok) {
                    // NOTE: in async mode a pinned staging slot is reused per (thread,proj) across
                    // experts, so we must issue its H2D before the next expert overwrites it. With the
                    // pinned arena the async copy reads the slot immediately on enqueue on the same
                    // stream, but to be safe against reuse we issue staged H2D synchronously here and
                    // only async-batch the direct-source (RAM/mmap, stable pointer) copies above.
                    ggml_backend_tensor_set(pr.dev, stg, (size_t) slot * stride, stride);
                }
            }
        }
        for (FILE * f : tfp) { if (f) { fclose(f); } }
        if (async_h2d && !local.empty()) {
            std::lock_guard<std::mutex> lk(pend_mu);
            pend.insert(pend.end(), local.begin(), local.end());
        }
    };

    if (n_threads <= 1) {
        load_range(0, nl, 0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve((size_t) n_threads);
        for (int t = 0; t < n_threads; ++t) {
            const size_t lo = (size_t) ((long long) t       * (long long) nl / n_threads);
            const size_t hi = (size_t) ((long long) (t + 1) * (long long) nl / n_threads);
            if (lo < hi) {
                pool.emplace_back(load_range, lo, hi, t);
            }
        }
        for (auto & th : pool) {
            th.join();
        }
    }

    // async H2D batch: issue all direct-source copies on the device stream from THIS thread, then a
    // single synchronize (instead of one cudaStreamSynchronize per copy). Sources are stable host
    // pointers (RAM pool / mmap), safe to read after the reads completed above.
    if (async_h2d && !pend.empty()) {
        for (const auto & p : pend) {
            ggml_backend_tensor_set_async(c->dev_backend, p.dev, p.src, p.off, p.bytes);
        }
        ggml_backend_synchronize(c->dev_backend);
    }
}

// map_custom1 callback: remap this step's routing ids [n_used,n_tokens] to cache-slot ids,
// running for BOTH prefill and decode. Correctness contract (CUDA single-GPU): a map_custom op
// is always CPU-assigned, so the scheduler runs it as its own split that fully completes before
// the downstream mul_mat_id GPU split is enqueued; the synchronous ggml_backend_tensor_set here
// blocks the host until the H2D copy lands, so every slot this callback resolves is resident in
// VRAM before the matmul that reads these ids runs. See llama_moe_cache_remap_cb for the same
// pattern on the per-tensor cache.
//
// Concurrency vs the background loader (shares g_moe_layer_mutex): this callback does NOT advance
// ring_next (the loader owns it); it evicts by age-LRU among slots not pinned this step, and pins
// every slot it resolves (pin_gen). The loader skips pinned slots, so it cannot overwrite a slot
// whose data the not-yet-run matmul still needs. Bumping pin_gen at entry releases the previous
// step's pins - safe because graph evaluations are serialized, so that step's matmul has finished.
static void llama_moe_layer_remap_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    llama_moe_layer_cache * c = (llama_moe_layer_cache *) userdata;

    const int64_t n_used   = a->ne[0];
    const int64_t n_tokens = a->ne[1];
    const int64_t n        = n_used * n_tokens;

    std::vector<int32_t> flat;
    llama_moe_flatten_ids(a, flat);

    const int64_t diag_lock0 = ggml_time_us();
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    const int64_t diag_t0 = ggml_time_us();
    // lock-acquire wait (contention with the background loader) - accumulated separately so we can see
    // if the callback is stalling on the mutex vs doing its own work. Printed in the timing block below.
    if (getenv("LLAMA_MOE_DIAG")) { g_diag_lockwait_ns += (uint64_t) (diag_t0 - diag_lock0); }

    // new pin generation for this step; slots pinned by the previous step are now releasable
    const uint64_t gen = ++c->pin_gen;

    // per-expert selection frequency this step
    std::vector<int> freq((size_t) c->n_expert, 0);
    for (int64_t p = 0; p < n; ++p) {
        const int32_t e = flat[(size_t) p];
        if (e >= 0 && e < c->n_expert) {
            freq[(size_t) e]++;
        }
    }

    // Weight-aware scoring: track which experts recurrently appear as a token's top-1/top-2 (highest
    // gate weight). hy3's routing churns the low-weight tail every token but its high-weight core is
    // stable (~a few dozen experts cover ~90% of top-2 picks), so a score that decays slowly and is
    // bumped by top-1/top-2 membership identifies that core. Eviction then protects high-score slots,
    // keeping the core resident so most steps hit it and need no in-line sync-load.
    if (c->weighted_evict) {
        if ((int) c->expert_score.size() != c->n_expert) { c->expert_score.assign((size_t) c->n_expert, 0.0f); }
        for (float & s : c->expert_score) { s *= 0.97f; } // slow global decay
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t u = 0; u < n_used; ++u) {
                const int32_t e = flat[(size_t) (t * n_used + u)];
                if (e >= 0 && e < c->n_expert) {
                    // weight by routing rank: top-1 gets most, decaying with position
                    c->expert_score[(size_t) e] += 1.0f / (float) (u + 1);
                }
            }
        }
    }

    std::vector<int32_t> ranked((size_t) c->n_expert);
    int n_unique = 0, n_miss = 0, n_ranked = 0;
    bool do_sync = llama_moe_plan_remap(freq.data(), c->n_expert, c->expert_slot.data(),
                                        c->sync_threshold, &n_unique, &n_miss,
                                        ranked.data(), &n_ranked);

    // decode miss-rate diagnostic: accumulate per-cache and print periodically so the real per-step
    // miss fraction (does hy3 nearly fully swap experts every token?) is observable. WARN-level so it
    // shows without -v. Only one cache (whichever hits the interval) prints, to avoid 80x spam.
    c->diag_steps  += (uint64_t) n_tokens;
    c->diag_unique += (uint64_t) n_unique;
    c->diag_miss   += (uint64_t) n_miss;
    if (getenv("LLAMA_MOE_DIAG")) {
        if ((int) c->diag_top2.size() != c->n_expert) { c->diag_top2.assign((size_t) c->n_expert, 0); }
        // count each token's top-1 and top-2 (routing positions 0 and 1 = highest gate weight)
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t u = 0; u < 2 && u < n_used; ++u) {
                const int32_t e = flat[(size_t) (t * n_used + u)];
                if (e >= 0 && e < c->n_expert) { c->diag_top2[(size_t) e]++; }
            }
        }
    }
    if (getenv("LLAMA_MOE_DIAG") && n_tokens == 1 && (c->diag_steps % 32) == 0) {
        // how concentrated are the top-2 picks? report what fraction of all top-2 selections fall on
        // the `capacity` most-frequent top-2 experts (high => a stable high-weight core the cache can hold)
        std::vector<uint64_t> sorted = c->diag_top2;
        std::sort(sorted.begin(), sorted.end(), [](uint64_t a, uint64_t b){ return a > b; });
        uint64_t total = 0, top_cap = 0;
        for (int e = 0; e < c->n_expert; ++e) { total += sorted[(size_t) e]; }
        int distinct = 0;
        for (int e = 0; e < c->n_expert; ++e) { if (sorted[(size_t) e] > 0) { distinct++; } }
        for (int e = 0; e < c->capacity && e < c->n_expert; ++e) { top_cap += sorted[(size_t) e]; }
        LLAMA_LOG_WARN("MoE diag [cache %p]: %llu steps, miss frac %.0f%%, capacity=%d | top-2 core: "
                       "%d distinct experts ever in top-2, hottest %d cover %.0f%% of all top-2 picks\n",
                       (void *) c, (unsigned long long) c->diag_steps,
                       100.0 * (double) c->diag_miss / (double) (c->diag_unique ? c->diag_unique : 1),
                       c->capacity, distinct, c->capacity,
                       100.0 * (double) top_cap / (double) (total ? total : 1));
    }

    // Budget mode (LLAMA_MOE_SYNC_BUDGET=N): instead of the all-or-nothing threshold sync, load at
    // most N of this step's MISSING experts, choosing the highest-weight ones. selected_experts is
    // argsort output (descending gate weight), so the missing experts encountered earliest in
    // routing-position order are exactly the highest-weight misses. The rest stay dropped and the
    // background loader warms them for later tokens. This bounds per-step disk reads to N (not the
    // full miss count), trading a little quality for predictable, much higher throughput.
    // Build the position-ordered unique expert list (first-appearance == descending weight).
    std::vector<int32_t> pos_ranked;
    if (c->sync_budget > 0) {
        do_sync = true; // budget mode always loads its top-N misses (threshold no longer gates)
        pos_ranked.reserve((size_t) n_unique);
        std::vector<char> seen((size_t) c->n_expert, 0);
        for (int64_t p = 0; p < n; ++p) {
            const int32_t e = flat[(size_t) p];
            if (e >= 0 && e < c->n_expert && !seen[(size_t) e]) {
                seen[(size_t) e] = 1;
                pos_ranked.push_back(e);
            }
        }
    }
    // Budget mode semantics: only the top-K (K = sync_budget) highest-weight experts of this step are
    // "guaranteed present". We look at exactly those K positions: any already in VRAM cost nothing;
    // any that miss are sync-loaded. Experts ranked below K are never sync-loaded here - they stay
    // dropped/async. So a step whose top-K are all resident does ZERO sync loads (no disk), which is
    // the fast common case once the high-weight core is cached. (Non-budget mode keeps the old
    // threshold behaviour: consider all unique experts, sync every miss.)
    const int32_t * order   = (c->sync_budget > 0) ? pos_ranked.data() : ranked.data();
    int             order_n = (c->sync_budget > 0) ? (int) pos_ranked.size() : n_ranked;
    // First decode step for this cache: sync-load ALL of this token's missing experts (ignore the
    // budget cap) so the very first token runs on the full real expert set, establishing a correct
    // baseline before stale reuse takes over on later steps. (User intent: "first token loads all 8".)
    const bool first_decode = (n_tokens == 1 && !c->pos_init);
    if (c->sync_budget > 0 && order_n > c->sync_budget && !first_decode) {
        order_n = c->sync_budget; // steady state: only inspect the top-K positions
    }
    if (n_tokens == 1) { c->pos_init = true; }

    // Resolve residency. Hits pin their slot; misses among the inspected positions are loaded into an
    // unpinned victim slot (bytes loaded in parallel below), then pinned too. Any expert not inspected
    // or not loaded maps to a zero sentinel slot when writing dst.
    std::vector<int> load_e;
    std::vector<int> load_slot;
    for (int r = 0; r < order_n; ++r) {
        const int32_t e = order[(size_t) r];
        int slot = c->expert_slot[(size_t) e];
        if (slot >= 0) {
            c->slot_pin[(size_t) slot] = gen;
            c->slot_age[(size_t) slot] = ++c->tick;
            continue;
        }
        // budget mode: every miss among the inspected top-K positions is synced. non-budget mode:
        // gated by the threshold decision (do_sync).
        if (c->sync_budget <= 0 && !do_sync) {
            continue; // below threshold: leave dropped, loader will warm it
        }
        // pick an eviction victim among slots not pinned this step: empty first; then, in weighted
        // mode, the LOWEST-score slot (protect the high-weight core), else the oldest (age-LRU).
        int      victim = -1;
        uint64_t oldest = UINT64_MAX;
        float    lowest = 3.4e38f;
        for (int s = 0; s < c->capacity; ++s) {
            if (c->slot_pin[(size_t) s] == gen) {
                continue; // pinned by this step's earlier hits/loads
            }
            if (c->slot_expert[(size_t) s] < 0) { victim = s; break; } // empty slot first
            if (c->weighted_evict) {
                const int32_t se = c->slot_expert[(size_t) s];
                const float   sc = (se >= 0 && se < c->n_expert) ? c->expert_score[(size_t) se] : 0.0f;
                if (sc < lowest) { lowest = sc; victim = s; }
            } else {
                if (c->slot_age[(size_t) s] < oldest) { oldest = c->slot_age[(size_t) s]; victim = s; }
            }
        }
        if (victim < 0) {
            break; // capacity exhausted this step: remaining experts drop to the sentinel
        }
        // drop the outgoing expert from the device slot_table first (before its slab is
        // overwritten), then claim the slot; the big byte copy happens in parallel afterwards
        const int old_e = c->slot_expert[(size_t) victim];
        if (old_e >= 0 && old_e < c->n_expert) {
            ggml_backend_tensor_set(c->slot_table, &c->capacity, (size_t) old_e * sizeof(int32_t), sizeof(int32_t));
            c->expert_slot[(size_t) old_e] = -1;
            moe_stale_on_evict(c, old_e, victim); // keep stale_table real for the pure-GPU decode path
        }
        c->slot_expert[(size_t) victim] = e;
        c->expert_slot[(size_t) e]      = victim;
        c->slot_age[(size_t) victim]    = ++c->tick;
        c->slot_pin[(size_t) victim]    = gen;
        load_e.push_back(e);
        load_slot.push_back(victim);
    }

    // load the missed experts' bytes into their claimed slots (parallel disk reads)
    const int64_t diag_pl0 = ggml_time_us();
    // Measurement (LLAMA_MOE_DIAG): of the experts we are about to sync-load, how many are already in
    // the locked RAM pool (fast ~0.5ms H2D) vs must hit disk (~4ms fread)? Counted here under the held
    // mutex on the same ram_slot predicate parallel_load uses, so it is race-free and exact. This is the
    // ceiling on what a lookahead RAM-prefetch could recover: only the DISK fraction is convertible.
    if (getenv("LLAMA_MOE_DIAG")) {
        for (int le : load_e) {
            if (le >= 0 && le < c->n_expert && !c->ram_slot.empty() && c->ram_slot[(size_t) le] >= 0) {
                g_diag_ram_hit++;
            }
        }
    }
    llama_moe_layer_parallel_load(c, load_e, load_slot);
    g_diag_pl_ns    += (uint64_t) (ggml_time_us() - diag_pl0);
    g_diag_sync_cnt += (uint64_t) load_e.size();
    // publish the freshly loaded slots into the device slot_table only after their data is in
    // place (keeps slot_table consistent for the background loader's bookkeeping)
    for (size_t i = 0; i < load_e.size(); ++i) {
        ggml_backend_tensor_set(c->slot_table, &load_slot[i], (size_t) load_e[i] * sizeof(int32_t), sizeof(int32_t));
        moe_stale_on_load(c, load_e[i], load_slot[i]); // mirror into stale_table for the pure-GPU decode path
        c->last_settled_slot = load_slot[i];
    }

    // Write the resolved slot per routing position. A resident expert -> its real slot; a dropped
    // Write the resolved slot per routing position. Resident expert -> its real slot. A MISSING
    // expert is handled differently for decode vs prefill:
    //  - DECODE (n_tokens==1): STALE REUSE - reuse the slot this routing position used last step
    //    (pos_slot[u]), which holds a real (if stale) expert. Perturbs the FFN far less than zeroing
    //    the branch. pos_slot[u] is refreshed to the real slot whenever position u hits.
    //  - PREFILL (n_tokens>1): drop to a zero sentinel slot (no stable per-position history across a
    //    batch; prefill loads its working set anyway and correctness, not staleness, is the concern).
    // Both keep the n_used slot ids within a token DISTINCT (MMQ requirement): resident experts have
    // distinct slots; for the rest we pick a slot not yet used this token (preferring the stale/sentinel
    // choice, else any free slot - always exists since n_slots = capacity + n_sentinel > n_used).
    const int n_slots = c->capacity + c->n_sentinel;
    const bool stale_reuse = (n_tokens == 1); // decode path
    std::vector<char> used_slot((size_t) n_slots);
    for (int64_t t = 0; t < n_tokens; ++t) {
        std::fill(used_slot.begin(), used_slot.end(), 0);
        // pass 1: mark slots taken by resident (hit) experts
        for (int64_t u = 0; u < n_used; ++u) {
            const int32_t e    = flat[(size_t) (t * n_used + u)];
            const int32_t slot = (e >= 0 && e < c->n_expert) ? c->expert_slot[(size_t) e] : -1;
            if (slot >= 0) { used_slot[(size_t) slot] = 1; }
        }
        // pass 2: assign a slot to each position
        int next_sentinel = c->capacity;
        int next_spare    = 0;
        for (int64_t u = 0; u < n_used; ++u) {
            const int32_t e    = flat[(size_t) (t * n_used + u)];
            int32_t slot = (e >= 0 && e < c->n_expert) ? c->expert_slot[(size_t) e] : -1;
            if (slot >= 0) {
                // hit: record this position's current real slot for future stale reuse
                if (stale_reuse && u < (int64_t) c->pos_slot.size()) { c->pos_slot[(size_t) u] = slot; }
            } else {
                // miss: prefer the stale slot this position used last step (decode), else a zero
                // sentinel (prefill); if that choice is already used this token, take any free slot.
                int32_t cand = -1;
                if (stale_reuse && u < (int64_t) c->pos_slot.size()) {
                    cand = c->pos_slot[(size_t) u];
                    if (cand < 0 || cand >= n_slots || used_slot[(size_t) cand]) { cand = -1; }
                }
                if (cand < 0) {
                    while (next_sentinel < n_slots && used_slot[(size_t) next_sentinel]) { next_sentinel++; }
                    if (next_sentinel < n_slots) {
                        cand = next_sentinel;
                    } else {
                        while (next_spare < n_slots && used_slot[(size_t) next_spare]) { next_spare++; }
                        cand = next_spare; // exists: at most n_used-1 slots used so far
                    }
                }
                slot = cand;
                if (stale_reuse && u < (int64_t) c->pos_slot.size()) { c->pos_slot[(size_t) u] = slot; }
            }
            used_slot[(size_t) slot] = 1;
            // Pin any resident slot ([0,capacity)) this token emits - including stale-reused slots -
            // so the background loader's victim scan skips it and cannot overwrite the slab between
            // now and when the downstream matmul reads it (a torn k-quant read otherwise). Hits were
            // already pinned in the residency loop; this covers the stale-reuse and spare-fallback
            // choices which are otherwise unprotected. Sentinel slots (>=capacity) are never loader
            // victims, so they need no pin. Released next step when pin_gen is bumped (decode steps
            // are separated by a full sched synchronize, so the prior matmul has finished by then).
            if (slot < c->capacity) { c->slot_pin[(size_t) slot] = gen; }
            *(int32_t *) ((char *) dst->data + t * dst->nb[1] + u * dst->nb[0]) = slot;
        }
    }

    // timing breakdown: accumulate total callback time; print + reset every 640 layer-calls (~8
    // decode steps at 80 layers) when LLAMA_MOE_DIAG is set, so we can see where a token's time goes.
    g_diag_cb_ns += (uint64_t) (ggml_time_us() - diag_t0);
    if (getenv("LLAMA_MOE_DIAG") && n_tokens == 1) {
        static uint64_t calls = 0;
        if (++calls % 640 == 0) {
            const double cb_ms = g_diag_cb_ns / 1000.0;
            const double pl_ms = g_diag_pl_ns / 1000.0;
            const double lw_ms = g_diag_lockwait_ns / 1000.0;
            // per-step (640 calls / 80 layers = 8 steps): callback time and how much is disk/H2D load
            const uint64_t sc = g_diag_sync_cnt ? g_diag_sync_cnt : 1;
            LLAMA_LOG_WARN("MoE timing/640-calls: callback %.0fms, parallel_load(disk+H2D) %.0fms (%.0f%%), "
                           "other(CPU remap) %.0fms; LOCK-WAIT %.0fms; sync-loaded %llu (RAM-hit %.0f%%, DISK %llu)\n",
                           cb_ms, pl_ms, cb_ms > 0 ? 100.0 * pl_ms / cb_ms : 0.0, cb_ms - pl_ms, lw_ms,
                           (unsigned long long) g_diag_sync_cnt,
                           100.0 * (double) g_diag_ram_hit / (double) sc,
                           (unsigned long long) (g_diag_sync_cnt - g_diag_ram_hit));
            g_diag_cb_ns = g_diag_pl_ns = g_diag_sync_cnt = g_diag_ram_hit = g_diag_lockwait_ns = 0;
        }
    }
}

ggml_tensor * llama_moe_layer_cache_remap(llama_moe_layer_cache * c,
                                          ggml_context *          ctx0,
                                          ggml_tensor *           selected_experts,
                                          float                   threshold) {
    if (!c) {
        return nullptr;
    }
    // guard the shapes ggml_mul_mat_id/the callback assume: a 2D i32 selection with at least one
    // routing position. llama-server can emit n_outputs==0 prefill ubatches (selection
    // [n_used, 0]); return nullptr so the caller falls back instead of building a degenerate op.
    const int64_t n_sel = selected_experts->ne[0] * selected_experts->ne[1];
    if (n_sel <= 0 || selected_experts->ne[2] != 1 || selected_experts->ne[3] != 1) {
        return nullptr;
    }
    c->sync_threshold = threshold;
    // weighted_evict is set at cache creation (default on) so the background loader uses it too;
    // do not clobber it here.
    // Per-step sync budget applies to DECODE only (n_tokens==1). Prefill must load the whole prompt
    // working set (bounded by capacity via the threshold), so budget must not throttle it.
    c->sync_budget = 0;
    if (selected_experts->ne[1] <= 1) {
        if (const char * b_env = getenv("LLAMA_MOE_SYNC_BUDGET")) {
            c->sync_budget = atoi(b_env);
            if (c->sync_budget < 0) { c->sync_budget = 0; }
        } else if (const char * hb = getenv("LLAMA_MOE_HYBRID_BUDGET")) {
            // static per-layer hybrid: the CPU-path layers use this budget without the global
            // SYNC_BUDGET (which would force ALL layers onto the CPU path). Decode only.
            c->sync_budget = atoi(hb);
            if (c->sync_budget < 0) { c->sync_budget = 0; }
        }
    }
    return ggml_map_custom1(ctx0, selected_experts, llama_moe_layer_remap_cb, 1, c);
}

// Token-boundary backpressure sync (LLAMA_MOE_SYNC_BOUNDARY). Called once per decode token AFTER the
// graph has computed and the GPU is synced, BEFORE the next token's graph is built. For each layer it
// ensures the top `budget` (highest-weight) experts this token selected are resident in VRAM: any the
// background loader has not kept resident are synchronously loaded NOW, which blocks the next token
// until they land. This bounds how far generation can drift on stale experts to at most one token,
// WITHOUT a per-layer CPU op - decode stays on the pure-GPU stale_table path; this is a once-per-token
// backstop that only stalls when the loader actually fell behind on a top-weight expert. When the hot
// core is warm every layer's top-k is already resident, so this finds nothing to do and does not stall
// (full speed). Returns the number of experts synchronously loaded this token (0 => no stall).
//
// Safety: runs at the token boundary, so this token's matmul has finished (GPU synced by the caller)
// and the next token's graph has not been built - there is NO in-flight matmul reading the slots. The
// only other writer is the background loader, which shares g_moe_layer_mutex (held here throughout), so
// the loads are race-free. sel is argsort output (position 0 = top-1 / highest gate weight), so the
// first `budget` positions are exactly the highest-weight experts; experts ranked below budget are left
// to the loader / stale reuse (this is the "top-2 present => do not descend to top-3/4" semantics).
int llama_moe_boundary_sync(int budget) {
    if (budget <= 0) {
        return 0;
    }
    // Probe-only mode: measure this token's pre-matmul expert residency but load NOTHING, so we observe
    // the LOADER's steady-state residency in isolation (the decisive fast+correct diagnostic). Decides
    // whether hy3's instantaneous working set fits in CAP (loader bug, fixable) or genuinely exceeds it
    // (capacity wall). Set LLAMA_MOE_BOUNDARY_PROBE_ONLY=1 with a nonzero LLAMA_MOE_SYNC_BOUNDARY.
    const bool probe_only = getenv("LLAMA_MOE_BOUNDARY_PROBE_ONLY") != nullptr;
    int n_loaded = 0;
    std::vector<int32_t> sel;
    std::vector<char>    ldbuf;

    // residency probe accumulators for THIS token (summed across all layers), counted BEFORE any load:
    // how many of the token's routed experts were already VRAM-resident, split into the top-2 subset
    // and the full n_used set. Separates "loader keeps the hot set warm" from "working set > CAP".
    int probe_top2_hit = 0, probe_top2_tot = 0;
    int probe_full_hit = 0, probe_full_tot = 0;

    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    for (auto & kv : g_moe_layer_caches) {
        llama_moe_layer_cache * c = kv.second;
        if (!c->sel_buf || !c->stale_table || c->capacity <= 0) {
            continue;
        }
        const int64_t n_used = c->sel_buf->ne[0];
        const int64_t n_cols = c->sel_buf->ne[1];
        if (n_used <= 0 || n_cols <= 0) {
            continue;
        }
        // read this layer's latest published selection (decode publishes [n_used,1] into column 0)
        const int64_t n = n_used * n_cols;
        sel.resize((size_t) n);
        ggml_backend_tensor_get(c->sel_buf, sel.data(), 0, (size_t) n * sizeof(int32_t));

        // residency probe (BEFORE any loads): for this layer count how many of this token's routed
        // experts were VRAM-resident, over the top-2 subset and the full n_used set. sel is argsort
        // output (position 0 = top-1), so u<2 is the top-2. This is the number that settles whether a
        // fast (pure-GPU, no per-token load) path can also be correct.
        for (int u = 0; u < (int) n_used; ++u) {
            const int32_t e = sel[(size_t) u];
            if (e < 0 || e >= c->n_expert) { continue; }
            const bool res = c->expert_slot[(size_t) e] >= 0;
            probe_full_tot++;
            if (res) { probe_full_hit++; }
            if (u < 2) { probe_top2_tot++; if (res) { probe_top2_hit++; } }
        }
        if (probe_only) { continue; } // observe loader-only steady state; issue no boundary loads

        int k = budget;
        if (k > (int) n_used) { k = (int) n_used; }

        // mark slots held by this token's resident top-k so a sibling miss does not evict them
        std::vector<char> claimed((size_t) c->capacity, 0);
        for (int u = 0; u < k; ++u) {
            const int32_t e = sel[(size_t) u];
            if (e < 0 || e >= c->n_expert) { continue; }
            const int32_t slot = c->expert_slot[(size_t) e];
            if (slot >= 0 && slot < c->capacity) { claimed[(size_t) slot] = 1; }
        }

        for (int u = 0; u < k; ++u) {
            const int32_t e = sel[(size_t) u];
            if (e < 0 || e >= c->n_expert) { continue; }
            if (c->expert_slot[(size_t) e] >= 0) {
                continue; // top-k expert already resident: nothing to do (do not descend below top-k)
            }
            // pick a victim slot not claimed by this token's top-k: empty first, then the lowest-score
            // (weighted) or oldest (age-LRU) occupant, mirroring the loader so the high-weight core is
            // preserved. No pin check needed: the prior matmul is done and the loader is blocked on the
            // mutex, so any slot not claimed above is safe to overwrite.
            int      victim = -1;
            uint64_t oldest = UINT64_MAX;
            float    lowest = 3.4e38f;
            for (int s = 0; s < c->capacity; ++s) {
                if (claimed[(size_t) s]) { continue; }
                if (c->slot_expert[(size_t) s] < 0) { victim = s; break; }
                if (c->weighted_evict) {
                    const int32_t se = c->slot_expert[(size_t) s];
                    const float   sc = (se >= 0 && se < c->n_expert && (int) c->expert_score.size() == c->n_expert)
                                        ? c->expert_score[(size_t) se] : 0.0f;
                    if (sc < lowest) { lowest = sc; victim = s; }
                } else {
                    if (c->slot_age[(size_t) s] < oldest) { oldest = c->slot_age[(size_t) s]; victim = s; }
                }
            }
            if (victim < 0) {
                break; // every slot claimed by this token's top-k (only if k==capacity): nothing to evict
            }
            moe_layer_load_into_slot(c, e, victim, ldbuf);
            claimed[(size_t) victim] = 1;
            n_loaded++;
        }
    }

    // optional diagnostic: how often does the boundary sync actually stall, and how many experts does
    // it load per token? A low average means the loader keeps the hot core warm (near full speed). The
    // pre-load residency fractions are the decisive fast+correct signal: if full-n_used residency is
    // high yet quality degrades, the problem is the stale tail / thrash (fixable), not capacity churn.
    if (getenv("LLAMA_MOE_DIAG")) {
        static uint64_t tokens = 0, loads = 0, stalls = 0;
        static uint64_t t2h = 0, t2t = 0, fh = 0, ft = 0;
        tokens++;
        loads += (uint64_t) n_loaded;
        if (n_loaded > 0) { stalls++; }
        t2h += (uint64_t) probe_top2_hit; t2t += (uint64_t) probe_top2_tot;
        fh  += (uint64_t) probe_full_hit; ft  += (uint64_t) probe_full_tot;
        if ((tokens % 64) == 0) {
            LLAMA_LOG_WARN("MoE boundary/64-tok: %llu loaded, %llu/64 stalled (%.0f%%), avg %.2f/token | "
                           "pre-load residency: top-2 %.1f%%, full %.1f%%%s\n",
                           (unsigned long long) loads, (unsigned long long) stalls,
                           100.0 * (double) stalls / 64.0, (double) loads / 64.0,
                           100.0 * (double) t2h / (double) (t2t ? t2t : 1),
                           100.0 * (double) fh / (double) (ft ? ft : 1),
                           probe_only ? " [PROBE-ONLY: loader steady state, no boundary loads]" : "");
            loads = stalls = 0; t2h = t2t = fh = ft = 0;
        }
    }
    return n_loaded;
}

void llama_moe_cache_shutdown(void) {
    // stop the background loader before the model/CUDA context is torn down, so it can
    // no longer touch freed expert weights or device buffers.
    if (g_moe_loader_run.exchange(false)) {
        if (g_moe_loader_thread.joinable()) {
            g_moe_loader_thread.join();
        }
    }
    // free the locked RAM residency pools and no-mmap file handles (loader stopped, no concurrent access)
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    for (auto & kv : g_moe_layer_caches) {
        llama_moe_layer_cache * c = kv.second;
        if (!c) {
            continue;
        }
        for (auto & pr : c->proj) {
            if (pr.ram) {
                llama_moe_ram_free(pr.ram, (size_t) c->ram_capacity * (size_t) pr.stride);
                pr.ram = nullptr;
            }
            if (pr.fp) {
                fclose(pr.fp);
                pr.fp = nullptr;
            }
            if (pr.fp_ld) {
                fclose(pr.fp_ld);
                pr.fp_ld = nullptr;
            }
        }
        if (c->stage_buf) {
            ggml_backend_buffer_free(c->stage_buf);
            c->stage_buf  = nullptr;
            c->stage_base = nullptr;
        }
    }
}
