#include "llama-moe-stream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
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
#include "ggml-cpu.h"
#include "llama-impl.h"
#include "llama.h" // fork: llama_moe_verify_gate_* are exported to tools (server)

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
#  include <unistd.h>
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

// Bytes the RAM tier asked to pin, split three ways: pinned (the lock succeeded), pageable (the lock
// failed and we kept the pool anyway) and refused (the lock failed and we handed the bytes straight
// back rather than keep a pool the OS can page out).
//
// Why "refused" is the default outcome: the pool duplicates bytes the model file on disk already holds.
// Pinned, it is a real memory tier. Unpinned, it is ANONYMOUS committed memory, and anonymous memory has
// exactly one place to go when the OS wants the frames - the pagefile. So an unpinned pool converts an
// eviction that would have been FREE (drop a file-backed page, re-read it from the model file later, on
// the drive the model already lives on) into a pagefile WRITE plus a pagefile read, usually on a
// different drive. Measured on 64 GB / a 73 GB expert file: a 36.4 GiB unpinned pool, pagefile peak
// 30.9 GB, system commit charge 41 GiB above physical RAM.
//
// Note on WHY the lock fails, because the comment that used to sit here named the wrong privilege:
// VirtualLock does not need SeLockMemoryPrivilege (that one is for large pages / AWE). Its ceiling is the
// process MINIMUM WORKING SET - MSDN: the most pages a process can lock is its minimum working set minus
// a small overhead. The default minimum is a couple of hundred KiB, so a multi-GiB lock fails
// deterministically, not intermittently. llama_moe_ram_quota_reserve raises that ceiling first.
static std::atomic<uint64_t> g_moe_ram_lock_ok{0};
static std::atomic<uint64_t> g_moe_ram_lock_fail{0};
static std::atomic<uint64_t> g_moe_ram_lock_refused{0};

// Raise the process minimum working set so the pool can be pinned? OFF BY DEFAULT, and it must stay
// that way until the bugcheck below is understood.
//
// Observed on Windows 10 19045 / 64 GiB: asking for a ~38 GiB HARD minimum working set (36 GiB pool +
// slack) and then locking it took the machine down with 0x0000003B SYSTEM_SERVICE_EXCEPTION, first
// argument 0xC00000A1 = STATUS_QUOTA_EXCEEDED - i.e. the kernel raised a quota exception while servicing
// the request. Not deterministic: the same request had been granted, and the pool fully pinned, on
// several earlier runs of the same build. A knob that can bugcheck the host is not a default no matter
// what it does for pagefile traffic, and a non-deterministic one cannot be validated by testing it once.
//
// Turning it on again (LLAMA_MOE_RAM_QUOTA=1) should be paired with a hard minimum that stays a small
// fraction of physical RAM; the sizes that provoked this were most of the box.
static bool llama_moe_ram_quota_enabled(void) {
    static const bool on = moe_env_on("LLAMA_MOE_RAM_QUOTA", false);
    return on;
}

// Keep a pool whose pin failed? Defaults to the quota state above, because the two only make sense
// together: VirtualLock's ceiling IS the minimum working set, so requiring a pin without raising it
// refuses every multi-GiB pool and silently deletes the whole RAM tier. So with the quota raise off
// (the default) this is off too, which is exactly the pre-fix behavior - an unpinned, pageable pool.
// Set LLAMA_MOE_RAM_REQUIRE_PIN explicitly to override either way.
static bool llama_moe_ram_require_pin(void) {
    static const bool on = moe_env_on("LLAMA_MOE_RAM_REQUIRE_PIN", llama_moe_ram_quota_enabled());
    return on;
}

// Record one pin attempt we are KEEPING and WARN the first time one failed, so a pageable pool shows up
// in the log instead of only as throughput.
static void llama_moe_ram_lock_record(bool ok, size_t bytes) {
    if (ok) {
        g_moe_ram_lock_ok.fetch_add(bytes, std::memory_order_relaxed);
        return;
    }
    g_moe_ram_lock_fail.fetch_add(bytes, std::memory_order_relaxed);
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        LLAMA_LOG_WARN("MoE stream: keeping an UNPINNED host expert pool (LLAMA_MOE_RAM_REQUIRE_PIN=0). "
                       "It is pageable anonymous memory, so the OS may write it to the pagefile and read "
                       "it back - which costs more than reading those experts from the model file, and "
                       "writes to the pagefile's drive for bytes the model file already holds.\n");
    }
}

// Drop `bytes` from whichever side of the gauge still has room for it. The allocation that owned them
// did not record which side it landed on, and the pin outcome is uniform in practice (it depends on the
// process working-set quota, not on the individual range), so charging the pinned side first is exact
// whenever every attempt agreed and off by at most one pool otherwise.
static void llama_moe_ram_lock_forget(size_t bytes) {
    uint64_t ok = g_moe_ram_lock_ok.load(std::memory_order_relaxed);
    const uint64_t take = ok < (uint64_t) bytes ? ok : (uint64_t) bytes;
    if (take) { g_moe_ram_lock_ok.fetch_sub(take, std::memory_order_relaxed); }
    const uint64_t rest = (uint64_t) bytes - take;
    if (rest) {
        uint64_t bad = g_moe_ram_lock_fail.load(std::memory_order_relaxed);
        g_moe_ram_lock_fail.fetch_sub(bad < rest ? bad : rest, std::memory_order_relaxed);
    }
}

// Allocate a host buffer and lock it into physical RAM, so reads from it never fault to disk AND the OS
// can never write it to the pagefile. Returns nullptr if the allocation fails, or if the LOCK fails and
// LLAMA_MOE_RAM_REQUIRE_PIN is on (the default) - the caller then runs without a RAM tier for that
// projection and reads those experts from the model file instead. *pinned reports the lock outcome, for
// the caller that decides whether dropping the page-cache copy is still a win.
static void * llama_moe_ram_alloc(size_t bytes, bool * pinned) {
    if (pinned) { *pinned = false; }
    if (bytes == 0) {
        return nullptr;
    }
#ifdef _WIN32
    void * p = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) {
        return nullptr;
    }
    const bool ok = VirtualLock(p, bytes) != 0;
#else
    void * p = nullptr;
    if (posix_memalign(&p, 4096, bytes) != 0 || !p) {
        return nullptr;
    }
    const bool ok = mlock(p, bytes) == 0;
#endif
    if (!ok && llama_moe_ram_require_pin()) {
        // Hand the bytes straight back. Keeping them would trade a free file-backed eviction for a
        // pagefile round trip (see the g_moe_ram_lock_* comment above), and it would also leave tens of
        // GiB of commit charge behind for a tier that cannot deliver what it promises.
        g_moe_ram_lock_refused.fetch_add(bytes, std::memory_order_relaxed);
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            LLAMA_LOG_WARN("MoE stream: refusing the host expert pool - a %.0f MiB allocation could not be "
                           "pinned into physical RAM, so it would be pageable anonymous memory that the OS "
                           "can only evict to the pagefile, for bytes the model file already holds. Running "
                           "without the RAM tier: those experts are read from the model file instead, which "
                           "is a read on the model's own drive and a write nowhere. To fit a pinnable pool "
                           "lower LLAMA_MOE_RAM_CAP or raise LLAMA_MOE_RAM_RESERVE_MB; to keep an unpinned "
                           "pool anyway set LLAMA_MOE_RAM_REQUIRE_PIN=0.\n", bytes/(1024.0*1024.0));
        }
#ifdef _WIN32
        VirtualFree(p, 0, MEM_RELEASE);
#else
        free(p);
#endif
        return nullptr;
    }
    llama_moe_ram_lock_record(ok, bytes);
    if (pinned) { *pinned = ok; }
    return p;
}

static void llama_moe_ram_free(void * p, size_t bytes) {
    if (!p) {
        return;
    }
    // keep the pinned/pageable figures a LIVE gauge, not a running total: the startup auto-capacity
    // recap frees every pool and rebuilds it, so counting allocations only would report 2x the pool.
    llama_moe_ram_lock_forget(bytes);
#ifdef _WIN32
    (void) VirtualUnlock(p, bytes);
    VirtualFree(p, 0, MEM_RELEASE);
#else
    (void) munlock(p, bytes);
    free(p);
#endif
}

// Available physical RAM in bytes (free, not total), or 0 if it cannot be measured.
static uint64_t llama_moe_avail_ram(void) {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        return (uint64_t) ms.ullAvailPhys;
    }
    return 0;
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long psize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psize > 0) {
        return (uint64_t) pages * (uint64_t) psize;
    }
    return 0;
#else
    return 0;
#endif
}

static void llama_moe_commit_charge(uint64_t * charge, uint64_t * phys_total); // defined just below

// Raise this process's MINIMUM WORKING SET so the RAM tier can actually LOCK `want_bytes` into physical
// memory, and report how much the OS agreed to.
//
// This is the piece that was missing all along. VirtualLock's ceiling is the minimum working set (MSDN:
// the most pages a process can lock is its minimum working set minus a small overhead), which defaults to
// a couple of hundred KiB - so every multi-GiB pin attempt failed deterministically and the "locked host
// RAM" tier had always been ordinary pageable memory. Raising the minimum needs SE_INC_WORKING_SET_NAME,
// which normal users have held by default since Vista; it just has to be ENABLED in the token, which
// nothing here used to do. A granted HARD minimum is a commitment that those frames stay resident, and
// resident frames are never written to the pagefile - which is the whole point.
//
// The OS is also the only honest answer to "how much fits". Computing it means guessing at terms we
// cannot see (other processes, the page-cache working set of a 73 GB model file, the driver's own host
// allocations), so instead: ask for the whole amount and binary-search down until the OS grants it. No
// estimation coefficients, and the answer comes from the component that gets to decide.
//
// Idempotent - the first caller's request wins and later callers get that answer, because the pools are
// created one layer at a time and the lock budget is process-wide (see the call site). Returns bytes
// granted. When there is no answer to be had it returns `want_bytes` unchanged, meaning "no information,
// do not clamp on my account": on non-Windows mlock is bounded by RLIMIT_MEMLOCK, which a process cannot
// raise for itself, so there the pin result is the only signal and the caller learns it that way.
static uint64_t llama_moe_ram_quota_reserve(uint64_t want_bytes) {
    static uint64_t granted = 0;
    static bool     done    = false;
    if (done) { return granted; }
    done    = true;
    granted = want_bytes;
    if (want_bytes == 0) { return 0; }
    // Baseline for the "host pool" line printed once the pools exist: the commit charge BEFORE any pool is
    // allocated. Everything in it is anonymous memory that is not ours to shrink, so it is the honest
    // "everything else" term - and if it already sits near physical RAM then no pool size is safe and the
    // pagefile traffic is not coming from us. Subtracting the pool from the later figure cannot tell those
    // two apart, which is why this is sampled here instead of inferred.
    {
        uint64_t charge = 0, phys = 0;
        llama_moe_commit_charge(&charge, &phys);
        const double gib = 1024.0*1024.0*1024.0;
        LLAMA_LOG_WARN("MoE stream: before host pools - system commit charge %.1f GiB of %.1f GiB physical, "
                       "avail phys %.1f GiB; the pool wants %.1f GiB.\n",
                       charge/gib, phys/gib, llama_moe_avail_ram()/gib, want_bytes/gib);
    }
    // Default path: leave the minimum working set alone. See llama_moe_ram_quota_enabled for why this is
    // opt-in (it bugchecked the host once with STATUS_QUOTA_EXCEEDED). Without it a multi-GiB pool cannot
    // be pinned, so the pool stays pageable - the pre-fix behavior.
    if (!llama_moe_ram_quota_enabled()) {
        LLAMA_LOG_WARN("MoE stream: not raising the process minimum working set (LLAMA_MOE_RAM_QUOTA is "
                       "off by default), so the host expert pool stays pageable anonymous memory that the "
                       "OS can write to the pagefile. LLAMA_MOE_RAM_QUOTA=1 pins it instead - EXPERIMENTAL, "
                       "it has been seen to bugcheck Windows with STATUS_QUOTA_EXCEEDED.\n");
        return granted; // want_bytes, i.e. no clamp: the pin result is the only signal, as before
    }
#ifdef _WIN32
    // Enable SE_INC_WORKING_SET_NAME in this process's token. advapi32 is resolved on demand so this adds
    // no link dependency - same reason as the DXGI probe below: fork-local diagnostics and knobs must not
    // change what upstream links against.
    {
        typedef BOOL (WINAPI * open_tok_fn)(HANDLE, DWORD, PHANDLE);
        typedef BOOL (WINAPI * lookup_fn)(LPCSTR, LPCSTR, PLUID);
        typedef BOOL (WINAPI * adjust_fn)(HANDLE, BOOL, PTOKEN_PRIVILEGES, DWORD, PTOKEN_PRIVILEGES, PDWORD);
        if (HMODULE adv = LoadLibraryA("advapi32.dll")) {
            open_tok_fn open_tok = (open_tok_fn) (void *) GetProcAddress(adv, "OpenProcessToken");
            lookup_fn   lookup   = (lookup_fn)   (void *) GetProcAddress(adv, "LookupPrivilegeValueA");
            adjust_fn   adjust   = (adjust_fn)   (void *) GetProcAddress(adv, "AdjustTokenPrivileges");
            HANDLE tok = nullptr;
            if (open_tok && lookup && adjust &&
                open_tok(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &tok) && tok) {
                LUID luid;
                if (lookup(nullptr, "SeIncreaseWorkingSetPrivilege", &luid)) {
                    TOKEN_PRIVILEGES tp;
                    tp.PrivilegeCount           = 1;
                    tp.Privileges[0].Luid       = luid;
                    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                    (void) adjust(tok, FALSE, &tp, 0, nullptr, nullptr); // best effort
                }
                CloseHandle(tok);
            }
        }
    }
    SIZE_T ws_min = 0, ws_max = 0;
    DWORD  ws_flags = 0;
    if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &ws_min, &ws_max, &ws_flags)) {
        return granted; // unmeasurable: let the pin result speak instead
    }
    // Headroom above the pool for everything else this process must keep resident (non-expert weights,
    // KV cache, the pinned H2D staging arena, the runtime). Without it the pool's own lock can consume
    // the whole quota and then the code that has to run against it gets trimmed instead.
    const uint64_t slack = 2048ull * 1024 * 1024;
    auto try_set = [&](uint64_t extra) {
        const SIZE_T req_min = (SIZE_T) ((uint64_t) ws_min + extra + slack);
        const SIZE_T req_max = (SIZE_T) (req_min + slack); // soft max, so growth above it is still allowed
        return SetProcessWorkingSetSizeEx(GetCurrentProcess(), req_min, req_max,
                                          QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE) != 0;
    };
    // Probe the whole request first. The common case is that the OS just says yes, and a bisection that
    // starts at want/2 can never come back up to want in a bounded number of steps - 12 halvings of 35 GiB
    // still leaves it ~9 MiB short, which is harmless for the pool but makes the log claim the OS "granted
    // only 35.7 GiB of the 35.7 GiB" it asked for.
    uint64_t best = 0;
    if (try_set(want_bytes)) {
        best = want_bytes;
    } else {
        uint64_t lo = 0, hi = want_bytes - 1; // lo = known grantable, hi = upper bound still in play
        for (int it = 0; it < 12 && lo < hi; ++it) {
            const uint64_t mid = lo + (hi - lo + 1) / 2;
            if (try_set(mid)) { best = mid; lo = mid; } else { hi = mid - 1; }
        }
        // The search may have ended on a REFUSED attempt, which leaves the quota wherever the last accepted
        // call put it - not necessarily `best`. Re-apply so the granted figure and the live quota agree.
        if (best > 0) { (void) try_set(best); }
    }
    granted = best;
    const double gib = 1024.0*1024.0*1024.0;
    if (granted >= want_bytes) {
        LLAMA_LOG_INFO("MoE stream: raised the process minimum working set to cover a %.1f GiB pinned host "
                       "expert pool (hard minimum, so the OS cannot trim those frames to the pagefile).\n",
                       want_bytes/gib);
    } else {
        LLAMA_LOG_WARN("MoE stream: the OS granted only %.1f GiB of the %.1f GiB minimum working set the host "
                       "expert pool asked for, so the pool is capped there. The experts that no longer fit "
                       "are read from the model file instead - slower per miss, but no pagefile traffic.\n",
                       granted/gib, want_bytes/gib);
    }
#endif
    return granted;
}

// System-wide commit charge and physical total, in bytes; both 0 if unmeasurable. A commit charge
// above physical total means the OS is backing part of the working set with the pagefile - invisible
// to the VRAM/RAM audits and to available-RAM readings, but it shows up as throughput.
static void llama_moe_commit_charge(uint64_t * charge, uint64_t * phys_total) {
    *charge     = 0;
    *phys_total = 0;
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        // ullTotalPageFile is the commit LIMIT (RAM + pagefile), ullAvailPageFile what is left of it.
        *charge     = ms.ullTotalPageFile - ms.ullAvailPageFile;
        *phys_total = ms.ullTotalPhys;
    }
#endif
}

// WDDM video-memory spill probe (Windows only; all-zero elsewhere).
//
// Why this exists: the VRAM AUDIT's "free" figure cannot answer whether the expert cache overflowed -
// on this platform it reads near zero at any cache size (see the comment on llama_moe_vram_audit). When
// the driver runs out of real VRAM it does not fail an allocation, it silently starts backing part of
// this process's video memory with host RAM over PCIe. DXGI reports that split directly, per process:
//   LOCAL     = real device memory this process is using, against the budget the driver grants it
//   NON_LOCAL = host memory this process is using AS video memory - non-zero means we spilled
// So NON_LOCAL usage is the one number that turns "did we overflow VRAM" from a guess into a reading.
//
// Confound to keep in mind when reading it: our own pinned H2D staging arena is host memory visible to
// the GPU, so a small steady NON_LOCAL figure (arena-sized) is expected and is not a cache overflow.
// What matters is NON_LOCAL growing with the expert cache or growing during decode.
//
// dxgi.dll is loaded on demand so nothing links against it: this is a fork-local diagnostic and must
// not add a build dependency. Adapter selection is by closest DedicatedVideoMemory to `dev_total` (the
// figure ggml already reports for the compute device), which is unambiguous on a single-GPU box and
// picks the right one on mixed-size multi-GPU. Returns false if unmeasurable.
struct llama_moe_vram_seg {
    uint64_t local_used     = 0;
    uint64_t local_budget   = 0;
    uint64_t nonlocal_used  = 0;
    uint64_t nonlocal_budget = 0;
};

#if defined(_WIN32) && defined(__has_include)
#  if __has_include(<dxgi1_4.h>)
#    define LLAMA_MOE_HAVE_DXGI 1
#  endif
#endif

#ifdef LLAMA_MOE_HAVE_DXGI
#include <dxgi1_4.h>

static bool llama_moe_vram_segments(uint64_t dev_total, llama_moe_vram_seg * out) {
    // Kill switch. This is diagnostics-only: it brings DXGI/COM into a process that otherwise only talks
    // to CUDA, so it has to be possible to take back out without a rebuild - both to bisect a crash
    // against it and because a probe is never worth breaking a run for.
    if (!moe_env_on("LLAMA_MOE_VRAM_PROBE", true)) {
        return false;
    }
    typedef HRESULT (WINAPI * create_factory_fn)(REFIID, void **);
    // one-time resolve; both statics stay put for the process lifetime (the DLL is never unloaded)
    static create_factory_fn create = nullptr;
    static bool resolved = false;
    if (!resolved) {
        resolved = true;
        if (HMODULE h = LoadLibraryA("dxgi.dll")) {
            create = (create_factory_fn) (void *) GetProcAddress(h, "CreateDXGIFactory1");
        }
    }
    if (!create) {
        return false;
    }
    IDXGIFactory1 * factory = nullptr;
    if (FAILED(create(__uuidof(IDXGIFactory1), (void **) &factory)) || !factory) {
        return false;
    }
    IDXGIAdapter3 * best = nullptr;
    uint64_t best_delta = UINT64_MAX;
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1 * a1 = nullptr;
        if (FAILED(factory->EnumAdapters1(i, &a1)) || !a1) {
            break;
        }
        DXGI_ADAPTER_DESC1 desc;
        IDXGIAdapter3 * a3 = nullptr;
        if (SUCCEEDED(a1->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void **) &a3)) && a3) {
            const uint64_t vram  = (uint64_t) desc.DedicatedVideoMemory;
            const uint64_t delta = vram > dev_total ? vram - dev_total : dev_total - vram;
            if (delta < best_delta) {
                if (best) { best->Release(); }
                best_delta = delta;
                best       = a3;
            } else {
                a3->Release();
            }
        }
        a1->Release();
    }
    factory->Release();
    if (!best) {
        return false;
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO loc, non;
    const bool ok = SUCCEEDED(best->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &loc)) &&
                    SUCCEEDED(best->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non));
    best->Release();
    if (!ok) {
        return false;
    }
    out->local_used       = loc.CurrentUsage;
    out->local_budget     = loc.Budget;
    out->nonlocal_used    = non.CurrentUsage;
    out->nonlocal_budget  = non.Budget;
    return true;
}
#else
static bool llama_moe_vram_segments(uint64_t dev_total, llama_moe_vram_seg * out) {
    (void) dev_total; (void) out;
    return false;
}
#endif

// Highest NON_LOCAL usage seen so far, so a spill that only happens mid-decode is still reported after
// the fact rather than only while it is happening.
static std::atomic<uint64_t> g_moe_vram_spill_peak{0};

// Sample the probe and fold it into the peak. Returns false if unmeasurable. Cheap enough for the
// per-token path only at a low duty cycle - callers rate-limit it (a DXGI query is a COM call, not a
// register read).
static bool llama_moe_vram_spill_sample(uint64_t dev_total, llama_moe_vram_seg * out) {
    if (!llama_moe_vram_segments(dev_total, out)) {
        return false;
    }
    uint64_t prev = g_moe_vram_spill_peak.load(std::memory_order_relaxed);
    while (out->nonlocal_used > prev &&
           !g_moe_vram_spill_peak.compare_exchange_weak(prev, out->nonlocal_used,
                                                        std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
    return true;
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

// Auto-pick the per-layer resident expert count (CACHE_CAP) from VRAM, so users need not compute it by
// hand (the old default of "full n_expert" silently OOMs / spills to CUDA system-memory-fallback on
// models whose experts far exceed VRAM - a ~10x slowdown that looks mysterious). Budgeting is subtle:
// this runs at the FIRST MoE layer's cache creation, when the other (n_moe_layers-1) layer caches are
// NOT yet allocated, so dev_free is transiently large. We must therefore budget for ALL layers up front
// AND reserve headroom for the KV cache + compute graph buffers that also grow on-device. We size the
// TOTAL expert-cache footprint to (dev_free - reserve) * frac, then divide by layers. Returns a capacity
// in [n_used, n_expert]; 0 if it cannot measure. Env overrides:
//   LLAMA_MOE_CACHE_CAP set (>0) -> honored verbatim, this is not called
//   LLAMA_MOE_VRAM_FRAC=F        -> fraction of the (free - reserve) budget for the cache (default 0.97)
//   LLAMA_MOE_VRAM_RESERVE_MB=N  -> MiB kept free as a fragmentation margin (default 512; KV/compute are
//                                   already allocated + excluded from the measured free, so this is small)
// Zero "sentinel" slots per layer. Single source of truth: the auto-capacity budget and the cache
// allocation MUST agree, or the cache overshoots free VRAM by the difference (see llama_moe_auto_capacity).
// fork: DISABLED - a tried-and-measured dead end, kept for the record. The idea was to make the auto
// capacity truly automatic instead of a flat guess: defer cache creation for llama_context's first pp
// reserve, read the real compute-buffer size, then size the cache against it. It **OOMs**: with no cache,
// build_moe_ffn falls back to the compaction gather, whose transient is the FULL n_expert-wide expert
// tensor (1.8 GiB/layer on dsv4), so the measurement graph needs far MORE compute buffer than the real one
// and the reserve fails outright ("failed to allocate compute pp buffers"). Doing this properly needs a
// provisional minimum-capacity cache (right graph shape, small transient) plus a cache teardown/realloc
// between reserve passes. Until then the flat reserve stands and llama_moe_vram_audit() makes the resulting
// over-commit visible.
// Original intent:
// llama_moe_auto_capacity has to run at the FIRST MoE layer's cache creation, i.e. during graph BUILD,
// before the scheduler's compute buffer exists - so on its own it can only hold back a fixed margin, and
// that margin was measured to under-count dsv4's graph by ~2.5 GiB, leaving 0.00 GiB free and costing ~47%
// decode. Fix the ordering instead of the constant: llama_context defers cache creation for its first pp
// reserve (the graph is still valid - build_moe_ffn falls back to the compaction gather when moe_lc is
// null), reads the real compute-buffer size, publishes it here, and the later reserve passes size the cache
// against that. Note the deferred graph uses the compaction path, whose per-layer transient is larger than
// the cache path's, so the measurement errs on the side of leaving MORE headroom - never less.
// Consecutive loader passes that promoted NOTHING. The prefill-boundary drain waits on this so the
// loader's backlog is worked off BEFORE decode starts, instead of trickling through decode while holding
// g_moe_layer_mutex (measured: that contention was the whole post-sweep decode regression).
static std::atomic<int> g_moe_loader_idle{0};

// fork: opportunistic prefetch accounting (LLAMA_MOE_HASH_PREFETCH). requested = ids queued that were not
// already resident; landed = ids the loader actually published. The ratio is not the metric that matters -
// whether they landed BEFORE the FFN read them is, and that shows up as the per-layer HIT% in LAYERDBG.
static std::atomic<uint64_t> g_moe_pf_req{0};
static std::atomic<uint64_t> g_moe_pf_landed{0};
// Number of queued-but-unserviced prefetch ids. Lets the loader poll for pending work with one relaxed
// atomic load instead of taking g_moe_layer_mutex, so it can afford to check BETWEEN caches rather than
// only once per pass - the whole value of a prefetch is latency, and a full 43-cache pass is far longer
// than the window between the token entering decode and the FFN reading those experts.
static std::atomic<int> g_moe_pf_pending{0};
// Ids that were already resident when queued. resident/(resident+req) is entry-time residency, which is the
// number to compare against the callback's HIT% - a large gap means the slot was recycled mid-step.
static std::atomic<uint64_t> g_moe_pf_resident{0};
static std::atomic<bool>   g_moe_defer_caches{false};
static std::atomic<size_t> g_moe_compute_reserve{0};
// Set for the duration of the measurement reserve pass (llama_moe_cache_defer_prewarm). A cache built
// while this is set fills one slot instead of `capacity`, because the recap is about to drop it.
static std::atomic<bool>   g_moe_defer_prewarm{false};
// Cap chosen by llama_moe_auto_capacity, cached across layers. File-scope rather than a function-local
// static so llama_moe_recap_from_compute_reserve can invalidate it after the real compute-buffer size
// is measured; the first (max dev_free) result is the one every layer must share, see the note there.
static int g_moe_cached_cap = 0;
static bool g_moe_autocap_logged = false;
// dev_free as llama_moe_auto_capacity saw it, plus the cache size it then budgeted. The audit subtracts both
// from the free VRAM measured after everything is allocated: whatever is left is VRAM that got consumed AFTER
// the capacity decision was made, which is precisely what auto capacity cannot see and must reserve for.
static std::atomic<size_t> g_moe_autocap_free{0};
static std::atomic<size_t> g_moe_autocap_planned{0};

int llama_moe_sentinel_count(int n_used) {
    // The prefill group sweep needs one sentinel per routing position (a token can have all n_used
    // positions outside the current expert group); otherwise one is enough.
    int n = llama_moe_prefill_sweep_enabled() ? n_used : 1;
    if (const char * s_env = getenv("LLAMA_MOE_SENTINELS")) {
        n = atoi(s_env);
    }
    if (n < 1)      { n = 1; }
    if (n > n_used) { n = n_used; }
    return n;
}

int llama_moe_auto_capacity(ggml_backend_sched_t sched,
                            ggml_tensor * const * exps_list, int n_proj,
                            int n_expert, int n_used, int n_moe_layers) {
    if (n_moe_layers <= 0 || n_proj <= 0) { return 0; }
    // Compute ONCE (at the first MoE layer) and reuse for every layer. build_moe_ffn calls this per
    // layer, but dev_free shrinks as each layer's cache allocates - so recomputing would give later
    // layers a smaller cap, leaving VRAM unused (measured: per-layer recompute filled only ~17 GB where
    // a uniform cap fills ~21.5 GB on a 24 GB card). A single model runs one geometry, so caching the
    // first (max-free) result is correct and reproduces the hand-tuned uniform-cap behaviour.
    // Reset by llama_moe_recap_from_compute_reserve once the real compute-buffer size is known.
    if (g_moe_cached_cap > 0) { return g_moe_cached_cap; }

    ggml_backend_t backend = llama_moe_pick_device_backend(sched);
    if (!backend) { return 0; }
    size_t dev_free = 0, dev_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(backend), &dev_free, &dev_total);
    if (dev_free == 0) { return 0; }

    // bytes one resident expert costs across all projections of a layer
    uint64_t per_expert = 0;
    for (int i = 0; i < n_proj; ++i) {
        if (!exps_list[i]) { return 0; }
        per_expert += (uint64_t) exps_list[i]->nb[2];
    }
    if (per_expert == 0) { return 0; }

    // Byte budget (LLAMA_MOE_CACHE_BYTES, set by --moe-stream-cache N[MB|GB]): a total-VRAM budget for the
    // expert cache across all MoE layers, converted here because this is the first point that knows both
    // n_moe_layers and the per-expert byte cost. Precedence is LLAMA_MOE_CACHE_CAP (explicit experts/layer)
    // > this > auto-from-free-VRAM; the CAP check lives at the call site, so reaching here means CAP was
    // unset. The budget covers the cache slabs only - sentinel slots and every other device allocation are
    // on top of it.
    if (const char * be = getenv("LLAMA_MOE_CACHE_BYTES")) {
        const unsigned long long want = strtoull(be, nullptr, 10);
        if (want > 0) {
            const uint64_t denom = (uint64_t) n_moe_layers * per_expert;
            long long cap = denom ? (long long) ((uint64_t) want / denom) : 0;
            if (cap < 1)        { cap = 1; }
            if (cap > n_expert) { cap = n_expert; }
            static bool logged = false;
            if (!logged) {
                logged = true;
                const double gib = 1024.0*1024.0*1024.0;
                LLAMA_LOG_WARN("MoE stream: CACHE_BYTES budget %.2f GiB -> CACHE_CAP=%lld (~%.2f GiB across "
                               "%d MoE layers, %.1f MiB/expert); sentinel slots are extra.\n",
                               (double) want / gib, cap, (double) ((uint64_t) cap * denom) / gib,
                               n_moe_layers, per_expert / (1024.0*1024.0));
            }
            return (int) cap;
        }
    }

    // Fraction of (dev_free - reserve) the expert cache may claim. Kept near 1.0 deliberately: the real
    // "something else will want VRAM" allowance is the ABSOLUTE reserve below, not this. See the reserve
    // comment for why the distinction matters when moving between cards.
    double frac = 0.97;
    if (const char * fe = getenv("LLAMA_MOE_VRAM_FRAC")) {
        const double f = atof(fe);
        if (f > 0.05 && f < 0.995) { frac = f; }
    }
    // ABSOLUTE VRAM headroom kept free out of dev_free. This is the knob that decides capacity, and it is
    // absolute rather than a fraction on purpose.
    //
    // dev_free is measured at the first MoE layer's cache creation, during the first real decode graph, so
    // the non-expert weights, KV cache and compute buffers are already allocated and excluded from it. Even
    // so, filling the rest with expert slabs is measurably wrong: past a threshold the driver starts paging
    // part of the working set over PCIe, and that paging is invisible in every VRAM counter (llama_moe_vram_
    // audit's free figure reads ~0.00 GiB at ANY cache size - raising this reserve by 0.9 GiB left the free
    // figure unchanged, the absorption simply grew to match). It shows up only as throughput.
    //
    // Measured on dsv4 IQ2 / 24 GiB (dev_free 16.14 GiB, 43 layers, 7.19 MiB/expert, sweep on so
    // slots = cap + 6), prefill / decode tok/s against TOTAL SLOTS:
    //     38 slots  63.8 / 14.44      44 slots  75.9 / 12.21      47 slots  47.6 /  9.35
    //     42 slots  76.1 / 14.61      46 slots  47.3 / 10.86      50 slots  50.3 / 10.41
    // The knee is sharp between 44 and 46 slots and costs ~1.5x on BOTH prefill and decode past it, while
    // more cache below the knee buys nothing (38 slots is no faster than 42). 42 slots = 12.68 GiB, i.e. it
    // wanted ~3.4 GiB of dev_free left alone, hence the default.
    //
    // Why absolute and not a frac: the requirement is a total-committed-VRAM ceiling, so it does not scale
    // with card size. Expressing it as a fraction happens to reproduce the same capacity here but transfers
    // badly - on a larger card it leaves too much idle, and on a SMALLER one it leaves less than the needed
    // headroom and lands past the knee, which is the expensive direction.
    //
    // What does not transfer even in this form: the ~3.4 GiB is itself a function of ubatch (compute buffer),
    // context length, KV quantisation and architecture, so a bigger ubatch or ctx wants more. Re-measure per
    // configuration with a capacity sweep (tools/dev/bench_moe.sh with LLAMA_MOE_CACHE_CAP=N, ~40s a run);
    // the knee is sharp and shows on both prefill and decode. The audit cannot find it for you.
    // (The pinned H2D staging arena is a single global buffer that shows up as Windows "Shared GPU memory" -
    // host RAM, NOT a dedicated-VRAM spill, so it is excluded here. A plain mmap'd file's page-cache working
    // set is not cudaHostRegister'd and does not count.)
    uint64_t reserve = (uint64_t) 3072 * 1024 * 1024;
    // Second pass, after llama_moe_recap_from_compute_reserve published a measured size: the compute
    // buffer is ALREADY allocated by then, so the dev_free read above has it subtracted. Subtracting the
    // measured size again double-counts it - measured on dsv4/24GiB, ctx 131072/ub 2048: cap 33 -> 23,
    // i.e. WORSE than the flat guess it was meant to replace. What is left to hold back here is only what
    // dev_free still does NOT account for, which is the whole point of splitting the old flat reserve:
    //   flat 3072 MiB = compute buffer (varies with ctx x ubatch) + everything else (does not)
    // and only the second part is a portable constant. "Everything else" is not just fragmentation: the
    // driver starts paging part of the working set over PCIe before dev_free reaches zero, and that is
    // invisible in every VRAM counter (llama_moe_vram_audit reads ~0 free at any cache size) - it shows
    // up only as throughput. Measured on dsv4 IQ2 / 24 GiB / ctx 16384 / ub 2048, N=3 alternating runs
    // with a 6894-token prompt and a standby purge before each: margin 1536 -> cap 37 (43 slots) decoded
    // 7.69 / 7.67 / 7.15 tok/s against the flat-guess cap 36 (42 slots) at 10.01 / 8.28 / 8.17 -
    // non-overlapping, i.e. ONE capacity step past the knee, matching the older 42-vs-44-slot sweep.
    // 1856 is the margin that leaves cap unchanged wherever the flat guess was already right (ctx 16384
    // ub 2048 -> 36, ctx 131072 ub 512/256 -> 33) and only tightens where it was not (ctx 131072 ub 2048:
    // compute buffer 2885 MiB left the flat 3072 with a 187 MiB margin, so cap 33 was overcommitted ->
    // 28). It is therefore a no-regression default reverse-engineered from known-good configurations,
    // NOT a swept optimum - re-sweep LLAMA_MOE_VRAM_MARGIN_MB per card against tok/s, the audit cannot
    // find the knee for you.
    if (g_moe_compute_reserve.load(std::memory_order_acquire) != 0) {
        reserve = (uint64_t) 1856 * 1024 * 1024;
        if (const char * me = getenv("LLAMA_MOE_VRAM_MARGIN_MB")) {
            const long long m = atoll(me);
            if (m >= 0) { reserve = (uint64_t) m * 1024 * 1024; }
        }
    }
    if (const char * re = getenv("LLAMA_MOE_VRAM_RESERVE_MB")) {
        const long long r = atoll(re);
        if (r >= 0) { reserve = (uint64_t) r * 1024 * 1024; }
    }
    // TOTAL device budget for the whole expert cache (all layers). Use dev_free (measured at the first
    // layer's cache creation, before the other layer caches allocate) minus the reserve for KV/compute.
    // dev_free ~= dev_total here since streaming keeps expert weights off-device, so this is close to
    // the real ceiling; the reserve absorbs the KV + compute buffers that grow afterward.
    const uint64_t usable = dev_free > reserve ? (uint64_t) (frac * (double) (dev_free - reserve)) : 0;
    // Out of VRAM must NOT read as "no opinion". Returning 0 leaves cap=0 at the call site, and
    // llama_moe_clamp_capacity maps 0 to FULL residency - so "nothing fits" allocated every expert, which
    // is exactly backwards. Measured on qwen3.8-flash-next + MTP: the draft context reached here with
    // dev_free already below the reserve and got a full 512-expert cache (1.74 GiB), putting the process
    // 4138 MiB over the driver's VRAM budget. n_used is the smallest capacity the remap can work with (one
    // slot per routed expert of a token), so it is the honest answer for "there is no room".
    if (usable == 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            const double gib = 1024.0*1024.0*1024.0;
            LLAMA_LOG_WARN("MoE stream: only %.2f GiB device memory free, at or below the %.2f GiB reserve "
                           "- falling back to the MINIMUM expert cache (CACHE_CAP=%d, one slot per routed "
                           "expert). Expect heavy sync-loading. Lower LLAMA_MOE_VRAM_RESERVE_MB, free VRAM, "
                           "or set LLAMA_MOE_CACHE_CAP by hand.\n",
                           dev_free/gib, reserve/gib, n_used);
        }
        g_moe_cached_cap = n_used; // keep every layer of this model on the same capacity
        return n_used;
    }
    // each layer holds (cap + n_sentinel) slabs; total = n_moe_layers * (cap+n_sentinel) * per_expert <=
    // usable. Subtracting a hardcoded 1 here (as this did) silently overshoots the VRAM budget by
    // (n_sentinel-1) * n_moe_layers * per_expert whenever more than one sentinel is in use - about
    // 1.5 GiB at 43 layers / 6 sentinels / 7.2 MiB per expert, which is enough to spill the cache into
    // Windows shared GPU memory and cost several times the throughput.
    const uint64_t denom = (uint64_t) n_moe_layers * per_expert;
    const int      n_sen = llama_moe_sentinel_count(n_used);
    long long cap = (long long) (usable / denom) - n_sen;

    if (cap < n_used)   { cap = n_used; }     // below the per-token activation is unusable; clamp up
    if (cap > n_expert) { cap = n_expert; }   // no point exceeding the whole layer
    if (!g_moe_autocap_logged) {
        g_moe_autocap_logged = true;
        const double gib = 1024.0*1024.0*1024.0;
        const double cache_gib = (double) ((uint64_t)(cap + n_sen) * denom) / gib;
        const bool   measured  = g_moe_compute_reserve.load(std::memory_order_acquire) != 0;
        // WARN level so it is visible before llama-completion pauses the log for generation.
        LLAMA_LOG_WARN("MoE stream: auto CACHE_CAP=%lld -> ~%.1f GiB expert cache across %d MoE layers "
                       "(free %.1f GiB, reserve %.1f GiB for KV/compute [%s], frac %.2f, %.1f MiB/expert). "
                       "Override with LLAMA_MOE_CACHE_CAP; tune LLAMA_MOE_VRAM_FRAC / LLAMA_MOE_VRAM_RESERVE_MB.\n",
                       cap, cache_gib, n_moe_layers, dev_free/gib, reserve/gib,
                       measured ? "margin only; compute buffer already in free" : "flat guess, pre-measurement",
                       frac, per_expert/(1024.0*1024.0));
    }
    g_moe_autocap_free.store(dev_free, std::memory_order_release);
    g_moe_autocap_planned.store((size_t) ((uint64_t) (cap + n_sen) * denom), std::memory_order_release);
    g_moe_cached_cap = (int) cap; // reuse for all remaining layers (see note at function top)
    return (int) cap;
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

struct llama_moe_layer_cache;

// fork: userdata for one pass of the prefill expert-group sweep. One entry per static expert-index
// group; addresses must stay valid from graph build until compute, so the owning vector is sized
// exactly once and never resized afterwards.
struct llama_moe_group_ud {
    llama_moe_layer_cache * c    = nullptr;
    int                     e_lo = 0;
    int                     e_hi = 0;
};

struct llama_moe_layer_cache {
    int n_expert  = 0;
    int capacity  = 0;
    int n_used    = 0;   // per-token expert activation
    int n_sentinel = 0;  // zero "sentinel" slots for dropped experts (<= n_used); see below
    int il        = -1;  // layer index parsed from the key tensor name; diagnostics only (LLAMA_MOE_LAYERDBG)

    // fork: opportunistic prefetch queue (LLAMA_MOE_HASH_PREFETCH). Experts this layer is KNOWN to need on
    // the token now entering decode, queued before the graph is built so the background loader can land them
    // ahead of the FFN. Written by llama_moe_layer_prefetch, drained at the top of the loader's pass.
    // Deliberately a HINT with no downstream dependency: an entry that does not land in time changes nothing,
    // because moe_layer_claim_slot leaves expert_slot[e] == -1 until publish, so the remap callback still sees
    // the expert as absent and applies the ordinary sync-budget / coverage / stale rules to it.
    std::vector<int32_t> prefetch_want;

    // Every expert id of the token now entering decode, whether resident or not. The drain must not evict
    // one of these to make room for another: the token needs all of them, so recycling a resident one to
    // load a missing one is a pure loss. prefetch_want alone is not enough for that - it holds only the
    // MISSING ids, and the resident ones are exactly the ones worth protecting.
    std::vector<int32_t> prefetch_keep;

    // prefill expert-group sweep: one descriptor per group, built once on first use
    std::vector<llama_moe_group_ud> group_ud;

    // Top-`capacity` experts of the most recent prefill ubatch, ranked by SELECTION FREQUENCY with the
    // exact comparator llama_moe_plan_remap uses (descending freq, ties by lower index). This is the
    // resident set an ORDINARY (capacity-capped) prefill would have left behind, so it is the target the
    // post-sweep refill must hit if decode is to behave the same. Note this is a different ranking from
    // expert_score (rank-weighted 1/(u+1), decayed): a score-ranked refill only reached ~49% overlap with
    // the ordinary prefill's resident set, which is why it changed nothing. Written under g_moe_layer_mutex.
    std::vector<int32_t> freq_ranked;

    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    // Owning context's scheduler, used as an identity tag only (never dereferenced). g_moe_layer_caches is
    // process-global, so a SECOND llama_context in the same process (a speculative/MTP draft model, an
    // embedding context, ...) sees the first model's caches too. The startup capacity recap must only drop
    // the caches it built itself: freeing another context's caches leaves that context's already-built
    // graphs holding dangling llama_moe_layer_cache pointers and freed device buffers, which segfaults on
    // its next decode.
    ggml_backend_sched_t  sched = nullptr;

    // Set by every sweep pass over THIS cache, cleared when llama_moe_refill_vram_caches refills it. The
    // process-global g_moe_sweep_dirty only says "some cache swept somewhere"; with two models in the
    // process that made one model's sweep force a full refill of the other model's caches (7+ GiB) on the
    // next decode token.
    bool sweep_dirty = false;

    // Built during the recap's measurement reserve, so only slot 0 was prewarmed - slots 1..capacity-1 hold
    // garbage and no expert but 0 is marked resident. Normally the recap frees this cache before anything
    // computes with it; llama_moe_cache_defer_prewarm(false) finishes the prewarm of any cache that
    // survives (a recap that declined after all).
    bool prewarm_pending = false;

    // Device handles for the pinned-staging H2D path (LLAMA_MOE_PINNED_STAGE). The pinned arena itself is
    // a single GLOBAL buffer shared by every layer (see g_moe_stage_* below), not one per layer - all
    // staging happens in parallel_load, which runs one layer at a time under g_moe_layer_mutex, so a
    // single arena is race-free and avoids replicating ~250 MiB x 80 layers of pinned host memory (which
    // Windows counts as "Shared GPU memory" since pinned host RAM is GPU-addressable).
    ggml_backend_dev_t    stage_dev  = nullptr; // device whose host-buffer type backs the pinned arena
    ggml_backend_t        dev_backend = nullptr; // device backend handle (for async H2D + one synchronize)

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
    // capacity: 1 while the background loader is writing this slot's bytes with g_moe_layer_mutex RELEASED.
    // Every victim search and every stale-slot choice must skip such a slot: its expert is unmapped and its
    // bytes are half-written, so routing anything to it would read torn quant blocks.
    std::vector<char>     slot_loading;
    int                   ring_next = 0; // grace-period eviction cursor
    uint64_t              tick = 0;
    uint64_t              pin_gen = 0;   // current step's pin generation; bumped by each remap callback
    float                 sync_threshold = 0.5f; // miss fraction above which the remap sync-loads
    int                   sync_budget = 0; // >0: sync-load at most this many top-weight missing experts/step
    float                 sync_cover = 0.0f; // >0: coverage mode - stop syncing once (cached+synced) gate-
                                             // weight sum reaches this fraction, capped at sync_budget misses.
                                             // Needs per-step weight values (set by the cov_cb via step_wexp).
    std::vector<float>    step_wexp;         // n_expert: this step's normalized gate weight per selected
                                             // expert (0 for unselected); filled by cov_cb, read in remap_cb.
    float                 step_cover = 0.0f; // this step's active coverage threshold (0 = off); set by cov_cb.
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

    // fork audit probe (LLAMA_MOE_EVICTDBG): census of the background loader's FORCED evictions. The
    // loader's ring scan normally skips a slot whose occupant out-scores the incoming expert, but the
    // last candidate is accepted unconditionally - so a low-score expert can displace the high-score
    // core exactly when the cache already holds a better set. Records the step at which an expert was
    // force-evicted, so a later critical-path sync-load of that same expert is countable as a boomerang
    // (i.e. the speculative eviction cost us a reload). Read-only: never affects placement.
    std::vector<uint64_t> fevict_step; // n_expert: diag_steps when force-evicted (0 = never), 0 if off
};

static std::mutex g_moe_layer_mutex;
static std::unordered_map<const ggml_tensor *, llama_moe_layer_cache *> g_moe_layer_caches;

// Set while the one-shot LLAMA_MOE_PREFILL is running (holds g_moe_layer_mutex the whole time). The
// background loader's fill_ram does its slow disk read OUTSIDE the mutex, so without this flag it could
// write a RAM slot the prefill is also filling. fill_ram checks this at the top of each tick and yields
// entirely while it is true - the prefill is a few seconds, one time, so the loader losing those ticks is
// harmless (the pool is full by the time it resumes).
static std::atomic<bool> g_moe_prefill_running{false};


// fork: measured NEGATIVE, kept only as a marker of a tried-and-rejected idea. Suspending the whole loader
// for the duration of a sweep looks right - the sweep cycles each cache through the entire expert index
// space, so any VRAM promotion the loader makes mid-sweep is evicted by the next group - but it also stops
// llama_moe_loader_fill_ram, and the RAM pool is NOT cycled by the sweep, so that warming is useful. Killing
// it left the loader with a BIGGER backlog to work off during decode: lock wait 1.68ms -> 2.03ms per
// callback, decode 6.00 -> 4.94 tok/s. If revisited, gate ONLY the VRAM promotion phase, never fill_ram.
static std::atomic<bool> g_moe_sweep_suspend{false};
// Non-zero while a loader tick is inside its UNLOCKED disk read (fill_ram reads outside g_moe_layer_mutex).
// The prefill sets g_moe_prefill_running, then spins until this is 0, so no in-flight loader read overlaps
// its writes. Handshake: the loader bumps this then re-checks the flag before reading (bail if set); the
// prefill sets the flag then waits for this to drain. seq_cst on both sides makes the ordering airtight.
static std::atomic<int> g_moe_loader_reading{0};

// Single GLOBAL pinned host staging arena shared by all layers (LLAMA_MOE_PINNED_STAGE). Expert reads on
// the no-mmap path land here before the H2D copy so the transfer runs at full PCIe rate. Only ever touched
// inside parallel_load, which runs one layer at a time under g_moe_layer_mutex, so one arena suffices - a
// per-layer arena would replicate ~250 MiB x 80 layers of pinned RAM (all counted as Windows "Shared GPU
// memory"). Sized to its high-water (io-threads x max-projection-stride) and freed in llama_moe_cache_shutdown.
static ggml_backend_buffer_t g_moe_stage_buf     = nullptr; // owns the pinned bytes
static char *                g_moe_stage_base    = nullptr; // g_moe_stage_buf base ptr
static size_t                g_moe_stage_slot    = 0;        // bytes per staging slot (== max proj stride seen)
static int                   g_moe_stage_nthread = 0;        // io threads the arena is currently sized for
static int                   g_moe_stage_nproj   = 0;        // projections the arena is currently sized for

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
// fork: the three phases of moe_layer_load_into_slot, separated so the SLOW part (the H2D copies) can run
// with g_moe_layer_mutex released. Measured motivation: the loader held the mutex across its H2D, and the
// decode MoE callback (44 of them per token) blocked on it - lock wait 0.10ms -> 1.68ms per callback after a
// prefill sweep, x44 layers = +64ms/token, i.e. the whole post-sweep decode regression. Freezing the loader
// drove that wait to exactly 0.0000ms, which is what identified it.
//
// Safety, mirroring the RAM tier's existing lock-free write (see llama_moe_loader_fill_ram): while the bytes
// are in flight NOTHING may point at the slot. claim() therefore (a) sends the outgoing expert to the zero
// sentinel in slot_table, (b) redirects every stale_table entry that names this slot elsewhere - note
// moe_stale_on_evict is deliberately a no-op, so evicted experts otherwise keep pointing at exactly the slot
// we are about to overwrite - and (c) marks the slot `loading` so no victim search or stale choice picks it.
static void moe_layer_claim_slot(llama_moe_layer_cache * c, int slot) {
    const int old_e = c->slot_expert[(size_t) slot];
    if (old_e >= 0 && old_e < c->n_expert) {
        ggml_backend_tensor_set(c->slot_table, &c->capacity, (size_t) old_e * sizeof(int32_t), sizeof(int32_t));
        c->expert_slot[(size_t) old_e] = -1;
    }
    c->slot_expert[(size_t) slot] = -1;
    // move any stale reference away from this slot (the pure-GPU decode path reads stale_table directly)
    if ((int) c->stale_of_expert.size() == c->n_expert) {
        int safe = c->last_settled_slot;
        if (safe == slot) { safe = (slot + 1) % (c->capacity > 0 ? c->capacity : 1); }
        for (int x = 0; x < c->n_expert; ++x) {
            if (c->stale_of_expert[(size_t) x] == slot) {
                c->stale_of_expert[(size_t) x] = safe;
                ggml_backend_tensor_set(c->stale_table, &safe, (size_t) x * sizeof(int32_t), sizeof(int32_t));
            }
        }
    }
    c->slot_loading[(size_t) slot] = 1;
}

// Byte copies only - safe to call with the mutex RELEASED once claim() has unmapped the slot.
static void moe_layer_write_slot(llama_moe_layer_cache * c, int e, int slot, std::vector<char> & ldbuf,
                                 const char * const * staged) {
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
}

// Publish under the lock: the slot becomes routable only now, after its bytes are in place.
static void moe_layer_publish_slot(llama_moe_layer_cache * c, int e, int slot) {
    c->slot_expert[(size_t) slot] = e;
    c->expert_slot[(size_t) e]    = slot;
    c->slot_age[(size_t) slot]    = ++c->tick;
    ggml_backend_tensor_set(c->slot_table, &slot, (size_t) e * sizeof(int32_t), sizeof(int32_t));
    moe_stale_on_load(c, e, slot);
    c->last_settled_slot = slot;
    c->slot_loading[(size_t) slot] = 0;
}

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

// LLAMA_MOE_PUBLISH_BATCH (default ON, set =0 to opt out): skip the write-only slot_table device writes and
// publish stale_table once per step instead of once per loaded expert. The layer cache's slot_table has no
// reader - no graph op consumes it, nothing calls ggml_backend_tensor_get on it, and its accessor has no
// callers - so those writes are dead work. stale_table IS read (by the pure-GPU decode remap), but the host
// mirror stale_of_expert is the source of truth, so one whole-table upload at the end of the step lands
// identical bytes. Measured: device syncs in the publish path down 3.66x (26 -> 6.9 per token), output
// byte-identical on both the budget>0 (CPU remap) and budget==0 (GPU get_rows over stale_table) paths.
static bool llama_moe_publish_batch(void) {
    static const bool on = moe_env_on("LLAMA_MOE_PUBLISH_BATCH", true);
    return on;
}
// device set_tensor calls issued by the publish path. Each is a cudaMemcpyAsync + cudaStreamSynchronize, so
// this COUNT (not a timer) is the acceptance criterion when batching them.
static std::atomic<uint64_t> g_diag_pub_cnt{0};

// decode-path timing breakdown (under g_moe_layer_mutex); printed by the diag block, then reset.
static uint64_t g_diag_cb_ns    = 0; // total time inside the remap callback
static uint64_t g_diag_pl_ns    = 0; // of which, in parallel_load (disk fread + H2D of synced experts)
static uint64_t g_diag_sync_cnt = 0; // experts sync-loaded (per layer, summed)
static uint64_t g_diag_ram_hit  = 0; // of those, how many were served from the RAM pool (no disk)
static uint64_t g_diag_lockwait_ns = 0; // time the remap callback waited to ACQUIRE g_moe_layer_mutex
static uint64_t g_diag_publish_ns = 0; // time in the slot_table/stale_table publish sync sets (batching target)
// read/h2d are summed across the io threads, so they are THREAD-SUMS: with N threads they can exceed
// the wall time of the parallel_load they are inside. Never subtract them from a wall figure.
static std::atomic<uint64_t> g_diag_read_ns{0};     // parallel_load: disk fread, thread-sum
static std::atomic<uint64_t> g_diag_h2d_ns{0};      // parallel_load: staged per-expert H2D, thread-sum
static std::atomic<uint64_t> g_diag_h2d_wall_ns{0}; // parallel_load: async H2D batch, WALL (issuing thread)
// The background loader calls parallel_load too, on its own thread, and g_diag_pl_ns does NOT cover it.
// Its reads therefore must not land in the counters above, or "read" ends up larger than the
// parallel_load it looks like a subset of - which is what made the old breakdown unreadable.
static std::atomic<uint64_t> g_diag_ldr_read_ns{0}; // background loader: disk fread, thread-sum
static std::atomic<uint64_t> g_diag_ldr_h2d_ns{0};  // background loader: staged H2D, thread-sum
static std::atomic<uint64_t> g_diag_ldr_cnt{0};     // background loader: expert-projections it moved
// WALL time of the whole read phase (spawn -> join, or the inline call at n_threads==1) plus the bytes
// actually freaded in it. These two are what a "did the io get faster" question needs: the wall figure is
// comparable across runs and the byte count normalises it, so read-phase bandwidth does not depend on
// which experts the generated text happened to route to. The thread-sums above cannot do either.
static std::atomic<uint64_t> g_diag_rdphase_ns{0};     // read phase, WALL (calling thread)
static std::atomic<uint64_t> g_diag_read_bytes{0};     // bytes freaded in it
static std::atomic<uint64_t> g_diag_ldr_rdphase_ns{0}; // same, background loader
static std::atomic<uint64_t> g_diag_ldr_bytes{0};
static uint64_t g_diag_sweep_lockwait_ns = 0; // sweep-path lock wait, kept out of g_diag_lockwait_ns so
                                              // the decode print cannot report prefill contention
static const int g_diag_steps = 16; // decode steps covered by one timing print

// MoE layer count, for diag step accounting only. Every expert tensor is registered at load time,
// before any layer cache exists, so this is exact from the first call. Do not hardcode a layer count:
// dsv4 has 43 and hy3 has 80, and a wrong divisor silently rescales every per-step figure.
static int llama_moe_diag_n_layers(const llama_moe_layer_cache * c) {
    const size_t np = (c && !c->proj.empty()) ? c->proj.size() : 0;
    if (np == 0 || g_moe_expert_files.size() < np) { return 1; }
    const int n = (int) (g_moe_expert_files.size() / np);
    return n > 0 ? n : 1;
}

// fork audit probe (LLAMA_MOE_SLOTDBG): read-only census of how each routing position's slot id was
// resolved in the dst loop. HIT = the routed expert is resident; SENTINEL = zeroed drop slot; SPARE =
// fallback to an arbitrary RESIDENT (wrong) expert because the sentinel slots were exhausted this
// token; STALE = decode-only per-position reuse. Counts prefill (n_tokens>1) and decode separately.
static std::atomic<uint64_t> g_slot_hit[2]      = {{0}, {0}};
static std::atomic<uint64_t> g_slot_sentinel[2] = {{0}, {0}};
static std::atomic<uint64_t> g_slot_spare[2]    = {{0}, {0}};
static std::atomic<uint64_t> g_slot_stale[2]    = {{0}, {0}};
static std::atomic<uint64_t> g_slot_tokens[2]   = {{0}, {0}};
static std::atomic<uint64_t> g_slot_hist[2][33]; // per-token hit count histogram (index = hits, capped)

// fork: per-LAYER decode residency profile (LLAMA_MOE_LAYERDBG). g_slot_* above aggregates all layers into
// one number, which hides the fact that layers are not equally hard. DeepSeek-V4 routes its first
// dsv4_hash_layer_count layers through a static token-id -> expert table (deepseek4.cpp: selected_experts =
// get_rows(ffn_gate_tid2eid, inp_tokens)), so on those layers the resident set can only track the OUTPUT
// TOKEN distribution, not hidden-state locality - a structurally different regime that this profile makes
// visible. Decode positions only (prefill would swamp it). 160 = LLAMA_MAX_LAYERS headroom.
#define MOE_LAYERDBG_MAX 160
static std::atomic<uint64_t> g_lay_hit[MOE_LAYERDBG_MAX];
static std::atomic<uint64_t> g_lay_stale[MOE_LAYERDBG_MAX];
static std::atomic<uint64_t> g_lay_sentinel[MOE_LAYERDBG_MAX];
static std::atomic<uint64_t> g_lay_spare[MOE_LAYERDBG_MAX];
static std::atomic<uint64_t> g_lay_tokens[MOE_LAYERDBG_MAX];

// fork audit probe (LLAMA_MOE_EVICTDBG): is the background loader's forced-eviction fallback actually
// evicting the high-score core? All read-only counters; the eviction policy itself is untouched.
static std::atomic<uint64_t> g_fev_admit{0};      // loader admissions (a slot was found and filled)
static std::atomic<uint64_t> g_fev_forced{0};     // of those, the score guard was bypassed on the last try
static std::atomic<uint64_t> g_fev_skips{0};      // slots skipped because the occupant out-scored the incoming
static std::atomic<uint64_t> g_fev_nofit{0};      // scans that found no slot at all (every slot pinned)
static std::atomic<uint64_t> g_fev_gap_x1000{0};  // sum of (victim_score - incoming_score) x1000 on forced
static std::atomic<uint64_t> g_fev_sync{0};       // remap_cb critical-path sync-loads (boomerang denominator)
static std::atomic<uint64_t> g_fev_boom{0};       // sync-loads of an expert the loader had force-evicted
static std::atomic<uint64_t> g_fev_boom8{0};      // ... within 8 steps of that eviction (hot boomerang)

// fork: read-only gate-weight distribution probe (LLAMA_MOE_WEIGHT_DIAG). Answers "how often is hy3's
// top-2 gate weight actually low (flat distribution), where forcing a top-2 sync-load buys little?" by
// binning the normalized top-1 weight per decode token. Purely observational - no effect on compute or
// residency. Accumulated across all layers, printed periodically.
static std::atomic<uint64_t> g_wdiag_tokens{0};   // decode token-layers observed
static std::atomic<uint64_t> g_wdiag_top1_lt20{0}; // of those, top-1 normalized weight < 0.20 (flat-ish)
static std::atomic<uint64_t> g_wdiag_top1_lt15{0}; // < 0.15 (very flat: top-1 barely above uniform 1/8=0.125)
static std::atomic<uint64_t> g_wdiag_top1_gt40{0}; // > 0.40 (peaked: top-1 dominates, sync clearly matters)
static std::atomic<uint64_t> g_wdiag_top2sum_x1000{0}; // running sum of (top1+top2) x1000, for a mean

// coverage-mode diagnostic: how many experts the coverage rule WOULD sync vs the fixed budget=2, and the
// already-cached weight fraction seen at decision time. Accumulated when LLAMA_MOE_SYNC_COVER is active.
static std::atomic<uint64_t> g_cov_tokens{0};       // decode token-layers seen by the coverage callback
static std::atomic<uint64_t> g_cov_synced{0};       // experts the coverage rule chose to sync (sum)
static std::atomic<uint64_t> g_cov_cached_x1000{0}; // sum of already-cached weight fraction x1000 (for mean)
static std::atomic<uint64_t> g_cov_zero_sync{0};    // token-layers where coverage needed ZERO sync (all covered)
// steady-state early-stop diagnostic (LLAMA_MOE_COVDBG): measured in remap_cb over the SAME order_n (top-2)
// that drives the early-stop, so we can see the actual cov_have vs step_cover the rule compares - the cov_cb
// numbers above sum over all n_used and cannot show why a top-2 early-stop fired.
static std::atomic<uint64_t> g_covss_tokens{0};       // token-layers where the early-stop path was active
static std::atomic<uint64_t> g_covss_have_x1000{0};   // sum of initial cov_have (top-2 cached weight) x1000
static std::atomic<uint64_t> g_covss_earlystop{0};    // misses skipped by the early-stop rule (sum)
static std::atomic<uint64_t> g_covss_cover_x1000{0};  // sum of the active step_cover x1000 (for mean)

static std::thread       g_moe_loader_thread;
static std::atomic<bool> g_moe_loader_run{false};

// Persisted expert-score warm start (LLAMA_MOE_SCORE_FILE). Cold-start residency learning is slow: the
// loader only learns which experts are hot by observing routing, so it fills the RAM/VRAM pool by expert
// index until enough tokens accrue expert_score - measured ~1400 tokens to climb from 1% to the ~85%
// steady-state RAM-hit. Persisting the learned per-layer scores and preloading them on the next run lets
// the pool fill with the RIGHT experts from tick 0, skipping the warm-up. Keyed by the expert tensor name
// (e.g. "blk.5.ffn_gate_exps.weight"), which is stable across runs. Purely a residency HINT: wrong/stale
// scores only change WHICH experts warm first, never correctness (SYNC_BUDGET still guarantees the real
// top-N each step). File format (binary): repeated [u32 name_len][name bytes][u32 n_expert][f32 x n_expert].
static std::unordered_map<std::string, std::vector<float>> g_moe_loaded_scores; // name -> scores, read once
static std::once_flag g_moe_scores_loaded_flag;

static const char * llama_moe_score_file(void) {
    return getenv("LLAMA_MOE_SCORE_FILE");
}

// Read the score file into g_moe_loaded_scores once (idempotent). Missing/absent file => empty map (no-op).
static void llama_moe_scores_load_once(void) {
    std::call_once(g_moe_scores_loaded_flag, []() {
        const char * path = llama_moe_score_file();
        if (!path) { return; }
        FILE * f = fopen(path, "rb");
        if (!f) { return; } // first run: no file yet
        for (;;) {
            uint32_t nlen = 0;
            if (fread(&nlen, sizeof(nlen), 1, f) != 1) { break; }
            if (nlen == 0 || nlen > GGML_MAX_NAME) { break; } // corrupt; stop
            std::string name((size_t) nlen, '\0');
            if (fread(&name[0], 1, nlen, f) != nlen) { break; }
            uint32_t ne = 0;
            if (fread(&ne, sizeof(ne), 1, f) != 1) { break; }
            if (ne == 0 || ne > 100000) { break; } // sanity
            std::vector<float> sc((size_t) ne);
            if (fread(sc.data(), sizeof(float), ne, f) != ne) { break; }
            g_moe_loaded_scores[name] = std::move(sc);
        }
        fclose(f);
        if (!g_moe_loaded_scores.empty()) {
            LLAMA_LOG_WARN("MoE stream: preloaded expert scores for %zu layers from %s (warm-start residency).\n",
                           g_moe_loaded_scores.size(), path);
        }
    });
}

// If a persisted score vector exists for this layer (by expert tensor name), copy it into `dst`. Returns
// true if seeded. Called at cache creation so the loader fills the hot set immediately.
static bool llama_moe_scores_seed(const ggml_tensor * exps, std::vector<float> & dst, int n_expert) {
    if (!exps || !llama_moe_score_file()) { return false; }
    llama_moe_scores_load_once();
    auto it = g_moe_loaded_scores.find(std::string(exps->name));
    if (it == g_moe_loaded_scores.end() || (int) it->second.size() != n_expert) { return false; }
    dst = it->second;
    return true;
}

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
static int llama_moe_loader_fill_ram(std::vector<char> & scratch) {
    (void) scratch;
    // Yield entirely while the one-shot startup prefill holds the pool: its slow reads run under the mutex
    // but write the same RAM slots fill_ram would, so overlapping the unlocked read below would race.
    if (g_moe_prefill_running.load(std::memory_order_acquire)) { return 0; }
    int promoted = 0; // experts promoted this call (drives the loader's adaptive backoff when warm)
    // Experts promoted into the locked RAM pool per loader tick, across ALL caches. The slow disk read
    // runs WITHOUT the mutex (see below), so a large budget does not lengthen any lock-hold; it only
    // fills the pool faster. The old default of 64/tick across ~80 layers was <1 expert/layer/tick, so
    // a large model's RAM pool took thousands of ticks to fill and mostly sat empty during a run (the
    // callback then missed to disk instead of RAM). 1024 fills a big pool within the first seconds.
    const int fill_budget_env = getenv("LLAMA_MOE_RAM_FILL") ? atoi(getenv("LLAMA_MOE_RAM_FILL")) : 1024;
    int fill_budget = fill_budget_env > 0 ? fill_budget_env : 1024; // experts promoted per loader tick (all caches)

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

            // slow part, NO lock held: read the expert's bytes into the RAM pool slot for every proj.
            // Guard against the one-shot prefill: announce we are about to read, then re-check the prefill
            // flag. If the prefill has started, bail (leave the slot empty, published below) so its writes
            // to this slot cannot race ours. Otherwise the prefill's spin on g_moe_loader_reading waits for
            // this read to finish before it touches any slot.
            bool ok = true;
            g_moe_loader_reading.fetch_add(1, std::memory_order_seq_cst);
            if (g_moe_prefill_running.load(std::memory_order_seq_cst)) {
                ok = false; // prefill owns the pool now; abandon this fill cleanly
            } else {
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
            }
            g_moe_loader_reading.fetch_sub(1, std::memory_order_seq_cst);

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
            promoted++;
        }
    }
    return promoted;
}

// fork: one-shot multi-threaded RAM-pool prefill (LLAMA_MOE_PREFILL). The background loader's fill_ram is
// SERIAL (single fseek+fread per slab, one loader thread, 5ms per-tick sleep) so it does NOT saturate the
// NVMe - the pool takes ~a minute to warm while decode misses to disk, and score-preload alone did not
// help because the loader still fills at serial-read speed. This fills every cache's RAM pool to capacity
// UP FRONT with a pool of io threads doing positioned reads (the same 16-thread pattern parallel_load uses
// to hit ~2 GB/s), no compute/sync interleaved, no lock (runs before decode, no concurrent pool access).
// Experts chosen by expert_score if seeded (LLAMA_MOE_SCORE_FILE), else by index. Pure I/O; correctness
// unaffected (just pre-populates the RAM tier the loader would fill anyway). No-op if no RAM pool / no
// no-mmap file handles (needs pr.fpath to open per-thread handles).
static void llama_moe_prefill_ram_pools(void) {
    const int64_t t0 = ggml_time_us();
    int n_threads = 16;
    if (const char * e = getenv("LLAMA_MOE_IO_THREADS")) { const int v = atoi(e); if (v > 0) { n_threads = v; } }

    // Fence out the background loader's UNLOCKED reads: set the flag, then wait for any in-flight loader
    // read to drain. After this, fill_ram bails at the top of every tick (and re-checks after announcing),
    // so no loader write can land on a slot we are about to fill. Cleared on exit.
    g_moe_prefill_running.store(true, std::memory_order_seq_cst);
    while (g_moe_loader_reading.load(std::memory_order_seq_cst) != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Hold g_moe_layer_mutex for the whole prefill: it is a one-time startup barrier, so the background
    // loader and the (blocked) first decode step must wait until the pool is warm. The threaded file reads
    // below write to distinct pr.ram slots this function has already claimed, so they need no inner lock.
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    std::vector<llama_moe_layer_cache *> caches;
    caches.reserve(g_moe_layer_caches.size());
    for (auto & kv : g_moe_layer_caches) { caches.push_back(kv.second); }

    uint64_t total_bytes = 0;
    int filled_caches = 0;
    for (llama_moe_layer_cache * c : caches) {
        if (!c || c->ram_capacity <= 0 || c->proj.empty() || !c->proj[0].ram) { continue; }
        // need per-thread file handles (no-mmap path); skip if not registered
        if (c->proj[0].fpath.empty()) { continue; }
        const int rc = c->ram_capacity;
        const int ne = c->n_expert;

        // choose the rc experts to resident: highest expert_score first, else lowest index. This is the
        // same target set fill_ram would converge to, computed once here.
        std::vector<int> pick; pick.reserve((size_t) rc);
        {
            std::vector<int> idx(ne);
            for (int e = 0; e < ne; ++e) { idx[e] = e; }
            const std::vector<float> & sc = c->expert_score;
            const bool have_scores = ((int) sc.size() == ne);
            std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
                const float sa = have_scores ? sc[(size_t) a] : 0.0f;
                const float sb = have_scores ? sc[(size_t) b] : 0.0f;
                if (sa != sb) { return sa > sb; } // highest score first
                return a < b;                      // tie / cold: lowest index
            });
            for (int i = 0; i < rc && i < ne; ++i) { pick.push_back(idx[(size_t) i]); }
        }

        // assign RAM slots 0..rc-1 to the picked experts, publish the maps (no decode running yet)
        for (int s = 0; s < (int) pick.size(); ++s) {
            c->ram_expert[(size_t) s] = pick[(size_t) s];
            c->ram_score[(size_t) s]  = ((int) c->expert_score.size() == ne) ? c->expert_score[(size_t) pick[(size_t) s]] : 0.0f;
            c->ram_slot[(size_t) pick[(size_t) s]] = s;
        }

        // multi-threaded positioned reads: each thread opens its OWN handle per projection and reads a
        // contiguous slice of the picked experts into their RAM slots. No lock, no shared file position.
        const int np = (int) c->proj.size();
        auto range = [&](int lo, int hi) {
            std::vector<FILE *> fp((size_t) np, nullptr);
            for (int pi = 0; pi < np; ++pi) {
                if (!c->proj[(size_t) pi].fpath.empty()) { fp[(size_t) pi] = fopen(c->proj[(size_t) pi].fpath.c_str(), "rb"); }
            }
            for (int s = lo; s < hi; ++s) {
                const int e = pick[(size_t) s];
                for (int pi = 0; pi < np; ++pi) {
                    llama_moe_proj_store & pr = c->proj[(size_t) pi];
                    if (!pr.ram || !fp[(size_t) pi]) { continue; }
                    char * dst = (char *) pr.ram + (size_t) s * pr.stride;
                    llama_moe_pread(fp[(size_t) pi], pr.foff0 + (uint64_t) e * pr.stride, dst, pr.stride);
                }
            }
            for (FILE * f : fp) { if (f) { fclose(f); } }
        };
        const int nl = (int) pick.size();
        int nt = n_threads; if (nt > nl) { nt = nl; }
        if (nt <= 1) {
            range(0, nl);
        } else {
            std::vector<std::thread> pool;
            for (int t = 0; t < nt; ++t) {
                const int lo = (int) ((long long) t * nl / nt);
                const int hi = (int) ((long long) (t + 1) * nl / nt);
                if (lo < hi) { pool.emplace_back(range, lo, hi); }
            }
            for (auto & th : pool) { th.join(); }
        }
        for (int pi = 0; pi < np; ++pi) { total_bytes += (uint64_t) pick.size() * c->proj[(size_t) pi].stride; }
        filled_caches++;
    }

    const double ms = (ggml_time_us() - t0) / 1000.0;
    const double gb = total_bytes / (1024.0*1024.0*1024.0);
    LLAMA_LOG_WARN("MoE stream: prefilled RAM pools - %.1f GiB across %d layers in %.1fs (%.2f GiB/s) "
                   "with %d io threads (warm-start; skips the serial loader ramp).\n",
                   gb, filled_caches, ms/1000.0, ms > 0 ? gb/(ms/1000.0) : 0.0, n_threads);

    // Host-memory pressure report. The pools are all allocated by now, so this is the final split of
    // pinned vs pageable vs refused pool bytes, next to the system commit charge - the only figure here
    // that can show the OS pushing part of the working set into the pagefile. Reading it back from there
    // costs more than reading the same experts from the model file, and the write itself lands on the
    // pagefile's drive, so a charge well above physical RAM means LLAMA_MOE_RAM_CAP is too high for this
    // box no matter what available RAM said at sizing time. "refused" is the healthy outcome for bytes
    // that could not be pinned: they were handed back instead of becoming pagefile traffic.
    {
        uint64_t charge = 0, phys = 0;
        llama_moe_commit_charge(&charge, &phys);
        const double gib = 1024.0*1024.0*1024.0;
        LLAMA_LOG_WARN("MoE stream: host pool %.1f GiB pinned / %.1f GiB pageable / %.1f GiB refused "
                       "(unpinnable), avail phys %.1f GiB, system commit charge %.1f GiB of %.1f GiB "
                       "physical.\n",
                       g_moe_ram_lock_ok.load(std::memory_order_relaxed)      / gib,
                       g_moe_ram_lock_fail.load(std::memory_order_relaxed)    / gib,
                       g_moe_ram_lock_refused.load(std::memory_order_relaxed) / gib,
                       llama_moe_avail_ram() / gib, charge / gib, phys / gib);
    }

    // VRAM slot-cache prefill: the RAM pool is now warm, so fill each layer's VRAM slots (0..capacity-1)
    // with its top-`capacity` experts BEFORE decode, sourcing bytes from the RAM pool (~25 GB/s) instead of
    // letting the first tokens warm VRAM one miss at a time. Same score-based pick as the RAM tier; VRAM is a
    // subset of the RAM set, so this only front-loads H2D copies decode would do anyway - it does not gamble
    // on future routing (a wrong pick is just evicted by normal LRU, no worse than a cold start). Uses the
    // stock moe_layer_load_into_slot, which also publishes slot_table/stale_table so decode sees the experts
    // immediately. Opt-out with LLAMA_MOE_PREFILL_VRAM=0. Held under the same mutex, before any decode step.
    if (moe_env_on("LLAMA_MOE_PREFILL_VRAM", true)) {
        const int64_t tv0 = ggml_time_us();
        uint64_t vram_bytes = 0;
        int vram_caches = 0;
        std::vector<char> ldbuf;
        for (llama_moe_layer_cache * c : caches) {
            if (!c || c->capacity <= 0 || c->proj.empty() || !c->proj[0].dev) { continue; }
            const int cap = c->capacity;
            const int ne  = c->n_expert;
            // pick the top-cap experts (highest expert_score, else lowest index) - same ranking as RAM
            std::vector<int> idx(ne);
            for (int e = 0; e < ne; ++e) { idx[e] = e; }
            const std::vector<float> & sc = c->expert_score;
            const bool have_scores = ((int) sc.size() == ne);
            std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
                const float sa = have_scores ? sc[(size_t) a] : 0.0f;
                const float sb = have_scores ? sc[(size_t) b] : 0.0f;
                if (sa != sb) { return sa > sb; }
                return a < b;
            });
            for (int s = 0; s < cap && s < ne; ++s) {
                const int e = idx[(size_t) s];
                if (c->expert_slot[(size_t) e] >= 0) { continue; } // already resident (shouldn't happen pre-decode)
                moe_layer_load_into_slot(c, e, s, ldbuf); // sources from warm RAM pool; publishes slot/stale tables
                for (auto & pr : c->proj) { vram_bytes += (uint64_t) pr.stride; }
            }
            vram_caches++;
        }
        const double vms = (ggml_time_us() - tv0) / 1000.0;
        const double vgb = vram_bytes / (1024.0*1024.0*1024.0);
        LLAMA_LOG_WARN("MoE stream: prefilled VRAM caches - %.1f GiB across %d layers in %.1fs (%.2f GiB/s) "
                       "from the warm RAM pool (skips per-token VRAM warm-up).\n",
                       vgb, vram_caches, vms/1000.0, vms > 0 ? vgb/(vms/1000.0) : 0.0);
    }

    // pool is warm; let the background loader resume its maintenance ticks (it re-checks the flag, and
    // still has to take the mutex we hold until this function returns, so it cannot act until we are done)
    g_moe_prefill_running.store(false, std::memory_order_seq_cst);
}

// fork: one-shot trigger for the prefill above. The once_flag MUST be SHARED by every callback that can be
// the first to run, which is why it lives here and not in a caller. Two callbacks can reach it - the
// single-pass remap and the prefill sweep's group remap - and a run whose PREFILL takes one path while its
// DECODE takes the other used to own a private flag per callback and so prefilled TWICE: with
// LLAMA_MOE_PREFILL_SWEEP=1 the warmup step (n_tokens>1) consumed the group callback's flag, then the first
// decode token was the single-pass callback's first call and re-read the whole pool. Measured on dsv4 IQ2:
// 6.6-7.1s of RAM pool + 1.2-2.2s of VRAM cache, holding g_moe_layer_mutex throughout, landing in the eval
// window in 17 of 17 sweep runs - i.e. charged to the reported decode tok/s, which is exactly what the
// comment at the single-pass call site says this prefill exists to avoid. llama_moe_prefill_ram_pools is not
// idempotent (it re-picks and re-reads every slot unconditionally), so the second pass was a full re-read.
static void llama_moe_prefill_once(void) {
    if (!moe_env_on("LLAMA_MOE_PREFILL", true)) {
        return;
    }
    static std::once_flag flag;
    std::call_once(flag, []() { llama_moe_prefill_ram_pools(); });
}

// fork: queue an opportunistic prefetch of `ids` for the MoE layer at index `il`. Non-blocking: it only
// appends to the layer cache's prefetch_want, which the background loader drains at the top of its next pass.
// H2D on the calling thread is NOT an option - ggml_backend_tensor_set synchronizes per call, so doing the
// loads here would just pay the whole cost serially, up front, which is strictly worse than letting the
// remap callback's sync budget handle the misses.
//
// The point of this entry is DeepSeek-V4's static token-id-routed layers: for il < dsv4_hash_layer_count the
// selection is get_rows(ffn_gate_tid2eid, inp_token) (see deepseek4.cpp), i.e. a pure function of the token
// id, so the exact expert set is known before the graph is even built - a prefetch oracle with no prediction
// error, unlike every hidden-state-based scheme tried before it. Measured motivation: those layers run at
// ~43-51% VRAM hit / ~47-55% stale versus ~64-70% / ~22-27% for every other layer, in every configuration
// tested (with and without the prefill sweep, on repetitive and on varied output).
void llama_moe_layer_prefetch(int il, const int32_t * ids, int n_ids) {
    if (il < 0 || !ids || n_ids <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    for (auto & kv : g_moe_layer_caches) {
        llama_moe_layer_cache * c = kv.second;
        if (!c || c->il != il || c->capacity <= 0) {
            continue;
        }
        c->prefetch_keep.assign(ids, ids + n_ids);
        for (int i = 0; i < n_ids; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= c->n_expert) { continue; }
            if (c->expert_slot[(size_t) e] >= 0) {
                g_moe_pf_resident.fetch_add(1, std::memory_order_relaxed);
                continue; // already resident: nothing to do
            }
            bool dup = false;
            for (const int32_t q : c->prefetch_want) { if (q == e) { dup = true; break; } }
            if (!dup && (int) c->prefetch_want.size() < 2 * c->n_used) {
                c->prefetch_want.push_back(e);
                g_moe_pf_req.fetch_add(1, std::memory_order_relaxed);
                g_moe_pf_pending.fetch_add(1, std::memory_order_relaxed);
            }
        }
        break; // one cache per layer index
    }
}

// Service queued prefetches. Called by the loader thread at the top of each pass AND between caches, so a
// request waits at most one cache's worth of work rather than a whole pass. Same claim (locked) -> write
// (unlocked H2D) -> publish (locked) split as the main path.
//
// Victim choice deliberately IGNORES expert_score: unlike the loader's speculative promotions these experts
// are known to be needed by the token now entering decode, so they outrank whatever the score ranking wants.
// What must NOT be evicted is another expert of the same token (prefetch_keep), plus the usual
// loading/pinned guards.
static void llama_moe_layer_parallel_load(llama_moe_layer_cache * c, const std::vector<int> & load_e,
                                          const std::vector<int> & load_slot, bool allow_pinned,
                                          bool from_loader);

static void moe_drain_prefetch(std::vector<char> & ldbuf) {
    (void) ldbuf;
    if (g_moe_pf_pending.load(std::memory_order_relaxed) <= 0) {
        return;
    }
    // Collect per cache, then service caches in ascending layer order: the value is entirely in landing
    // before that layer's FFN, and layer 0 executes first (and so has the least horizon).
    struct moe_pf_group {
        llama_moe_layer_cache * c = nullptr;
        std::vector<int>        e;
        std::vector<int32_t>    keep;
    };
    std::vector<moe_pf_group> groups;
    {
        std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
        for (auto & kv : g_moe_layer_caches) {
            llama_moe_layer_cache * c = kv.second;
            if (!c || c->prefetch_want.empty()) { continue; }
            moe_pf_group g;
            g.c    = c;
            g.keep = c->prefetch_keep;
            for (const int32_t e : c->prefetch_want) {
                if (e >= 0 && e < c->n_expert && c->expert_slot[(size_t) e] < 0) {
                    g.e.push_back((int) e);
                }
            }
            g_moe_pf_pending.fetch_sub((int) c->prefetch_want.size(), std::memory_order_relaxed);
            c->prefetch_want.clear();
            if (!g.e.empty()) { groups.push_back(std::move(g)); }
        }
    }
    if (groups.empty()) {
        return;
    }
    std::stable_sort(groups.begin(), groups.end(), [](const moe_pf_group & a, const moe_pf_group & b) {
        return a.c->il < b.c->il;
    });

    for (moe_pf_group & g : groups) {
        llama_moe_layer_cache * c = g.c;
        std::vector<int> load_e, load_slot;
        // Claim every slot for this layer's batch in ONE critical section, so the unlocked transfer below
        // can move the whole batch with 16-way reads and a single device synchronize instead of one
        // synchronous H2D per expert.
        {
            std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
            for (const int e : g.e) {
                if (c->expert_slot[(size_t) e] >= 0) {
                    continue; // landed some other way since collection
                }
                int slot = -1;
                for (int tries = 0; tries < c->capacity; ++tries) {
                    const int cand = c->ring_next;
                    c->ring_next   = (c->ring_next + 1) % c->capacity;
                    if (c->slot_loading[(size_t) cand]) { continue; }
                    if (c->slot_pin[(size_t) cand] == c->pin_gen) { continue; }
                    const int occ = c->slot_expert[(size_t) cand];
                    if (occ >= 0) {
                        bool occ_needed = false;
                        for (const int32_t k : g.keep) { if (k == occ) { occ_needed = true; break; } }
                        if (occ_needed) { continue; } // the token needs this one too
                    }
                    slot = cand;
                    break;
                }
                if (slot < 0) { continue; }
                moe_layer_claim_slot(c, slot);
                load_e.push_back(e);
                load_slot.push_back(slot);
            }
        }
        if (load_e.empty()) {
            continue;
        }
        // UNLOCKED: nothing points at a claimed slot, so the bytes can land without the mutex. Must skip
        // the global pinned arena (see the note on llama_moe_layer_parallel_load) precisely because this
        // runs unlocked and the decode callback can be inside parallel_load at the same time.
        llama_moe_layer_parallel_load(c, load_e, load_slot, /*allow_pinned=*/false, /*from_loader=*/true);
        {
            std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
            for (size_t i = 0; i < load_e.size(); ++i) {
                moe_layer_publish_slot(c, load_e[i], load_slot[i]);
                // Pin for the rest of the current generation: without this the loader's own promotion of
                // the PREVIOUS token's selection recycles the slot before the FFN reads it. The generation
                // is bumped by this layer's next callback, which then pins its own hits.
                c->slot_pin[(size_t) load_slot[i]] = c->pin_gen;
            }
        }
        g_moe_pf_landed.fetch_add((uint64_t) load_e.size(), std::memory_order_relaxed);
    }
}

// least-recently-used slot and load the expert into it across ALL projections, then
// publish slot_table and finally resident_flag. resident_flag is written LAST so the
// compute never treats an expert as resident before its slot and data are in place.
static void llama_moe_loader_main() {
    const bool no_loader = getenv("LLAMA_MOE_NOLOADER") != nullptr; // diagnostic: freeze cache at prewarm
    // Opt-in; the caller side (llama_context::decode) gates on the same variable, so with it unset nothing
    // ever reaches prefetch_want and this drain is a single empty-vector check per pass.
    const bool prefetch_reqs_disabled = !moe_env_on("LLAMA_MOE_HASH_PREFETCH", false);
    std::vector<int32_t> sel;
    std::vector<char>    ldbuf; // scratch for reading an expert from file (no-mmap) before H2D
    std::vector<std::vector<char>> ldstage; // per-(miss,proj) staging buffers, pre-read outside the lock
    // Adaptive backoff (LLAMA_MOE_LOADER_BACKOFF, opt-in): once the pool is warm the loader keeps churning
    // full-speed (VRAM promote + RAM fill every ~100us). The idea was that its continuous disk/PCIe traffic
    // competes with the critical-path top-2 sync-load in the decode callback, so when a pass promotes NOTHING
    // (warm pool) it sleeps progressively longer (cap 20ms) and snaps back the instant a pass finds work. The
    // quality-bearing top-2 sync is never throttled - only this speculative background warming backs off.
    // Default OFF: measured ~neutral. It cuts loader disk %-busy 55->42% (external typeperf, stable), but a
    // colder RAM pool raises the critical-path top-2 sync read ~11% (148->164ms/640-calls) and drops RAM-hit
    // ~88->70%; the two roughly cancel so tps can't distinguish them. Idleness-based backoff starves exactly
    // the re-warming that keeps the top-2 in RAM. Kept as opt-in for a future contention-based variant.
    static const bool loader_backoff = []() { const char * e = getenv("LLAMA_MOE_LOADER_BACKOFF"); return e && atoi(e) != 0; }();
    static const bool evictdbg = getenv("LLAMA_MOE_EVICTDBG") != nullptr; // read-only forced-eviction census
    int idle_passes = 0;
    while (g_moe_loader_run.load(std::memory_order_relaxed)) {
        if (no_loader) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        int pass_promoted = 0; // VRAM slot promotions this pass (+ RAM fills below); 0 => pool warm, back off
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

        // Priority prefetch: serviced before any score-driven work in this pass, and again between caches
        // below, because a request that lands after its layer's FFN has already run is simply wasted.
        if (!prefetch_reqs_disabled) { moe_drain_prefetch(ldbuf); }

        for (llama_moe_layer_cache * c : ldcaches) {
            // Poll for latency-critical prefetch work between caches (one relaxed atomic load when idle).
            if (!prefetch_reqs_disabled) { moe_drain_prefetch(ldbuf); }
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
                std::unique_lock<std::mutex> lock(g_moe_layer_mutex);
                std::vector<const char *> staged(n_proj, nullptr);
                for (size_t k = 0; k < miss_e.size(); ++k) {
                    const int32_t e = miss_e[k];
                    if (e < 0 || e >= c->n_expert || c->expert_slot[(size_t) e] >= 0) {
                        continue; // became resident (e.g. the callback loaded it) since Phase A
                    }
                    const float in_score = (c->weighted_evict && (int) c->expert_score.size() == c->n_expert)
                                            ? c->expert_score[(size_t) e] : 0.0f;
                    int   slot   = -1;
                    bool  forced = false;  // audit: accepted a slot whose occupant OUT-SCORES the incoming
                    float vic_sc = 0.0f;   // audit: that occupant's score
                    for (int tries = 0; tries < c->capacity; ++tries) {
                        const int cand = c->ring_next;
                        c->ring_next   = (c->ring_next + 1) % c->capacity;
                        if (c->slot_loading[(size_t) cand]) {
                            continue; // another in-flight unlocked write owns this slot
                        }
                        if (c->slot_pin[(size_t) cand] == c->pin_gen) {
                            continue; // pinned by the current step, never overwrite
                        }
                        if (c->weighted_evict) {
                            const int32_t se = c->slot_expert[(size_t) cand];
                            const float   sc = (se >= 0 && se < c->n_expert && (int) c->expert_score.size() == c->n_expert)
                                                ? c->expert_score[(size_t) se] : 0.0f;
                            if (se >= 0 && sc > in_score) {
                                if (tries < c->capacity - 1) {
                                    if (evictdbg) { g_fev_skips.fetch_add(1, std::memory_order_relaxed); }
                                    continue;
                                }
                                forced = true; // last try: the guard is bypassed and a better expert dies
                                vic_sc = sc;
                            }
                        }
                        slot = cand;
                        break;
                    }
                    if (slot < 0) {
                        if (evictdbg) { g_fev_nofit.fetch_add(1, std::memory_order_relaxed); }
                        break; // every slot pinned by the current step: leave misses dropped for now
                    }
                    if (evictdbg) {
                        g_fev_admit.fetch_add(1, std::memory_order_relaxed);
                        if (forced) {
                            g_fev_forced.fetch_add(1, std::memory_order_relaxed);
                            g_fev_gap_x1000.fetch_add((uint64_t) ((vic_sc - in_score) * 1000.0f), std::memory_order_relaxed);
                            const int32_t out_e = c->slot_expert[(size_t) slot];
                            if (out_e >= 0 && (int) c->fevict_step.size() == c->n_expert) {
                                c->fevict_step[(size_t) out_e] = c->diag_steps + 1; // +1 so 0 stays "never"
                            }
                        }
                    }
                    const char * const * staged_ptr = nullptr;
                    if (miss_disk[k]) {
                        for (size_t pi = 0; pi < n_proj; ++pi) { staged[pi] = ldstage[k * n_proj + pi].data(); }
                        staged_ptr = staged.data();
                    }
                    // claim under the lock, copy the bytes with it RELEASED, publish under it again
                    moe_layer_claim_slot(c, slot);
                    lock.unlock();
                    moe_layer_write_slot(c, e, slot, ldbuf, staged_ptr);
                    lock.lock();
                    moe_layer_publish_slot(c, e, slot);
                    pass_promoted++;
                }
            }
        }
        // RAM residency tier fill (separate phase, outside the per-cache lock above). Keep the
        // highest-score experts resident in each layer's locked host pool so VRAM misses read from
        // RAM (~25 GB/s) instead of faulting the model file from disk. The slow part - the fread of
        // an expert's bytes - is done WITHOUT holding g_moe_layer_mutex (so it never stalls decode's
        // publish or the CPU remap); the mutex is taken only briefly to pick a target and to publish.
        pass_promoted += llama_moe_loader_fill_ram(ldbuf);
        if (pass_promoted > 0) { g_moe_loader_idle.store(0, std::memory_order_release); }
        else                   { g_moe_loader_idle.fetch_add(1, std::memory_order_acq_rel); }
        // Adaptive backoff: if this pass did real work, stay on the fast cadence (100us). If it promoted
        // nothing (warm pool), ramp the sleep up so the loader stops competing with decode for disk/PCIe -
        // but keep waking periodically (cap ~20ms) to catch routing drift. Snaps back to fast the moment a
        // pass finds work. Disabled with LLAMA_MOE_LOADER_BACKOFF=0 (restores the old always-100us churn).
        if (loader_backoff && pass_promoted == 0) {
            if (idle_passes < 200) { idle_passes++; }
            int sleep_us = 100 * idle_passes;      // linear ramp: 100us -> 20ms over ~200 idle passes
            if (sleep_us > 20000) { sleep_us = 20000; }
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        } else {
            idle_passes = 0;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
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

// Auto-pick the RAM residency-pool size (experts/layer) from available system RAM, so users need not
// tune LLAMA_MOE_RAM_CAP by hand. The pool holds high-weight experts in locked host RAM (~25 GB/s) so
// VRAM-cache misses avoid a disk fault (~2 GB/s); bigger is faster UNTIL it starves the OS + the mmap
// working set, at which point contention costs more than it saves. Budgeting mirrors
// llama_moe_auto_capacity: this runs at the FIRST MoE layer's cache creation, so the other
// (n_moe_layers-1) pools are NOT yet allocated - size the TOTAL pool footprint (all layers) from avail
// RAM up front, then divide by layers. per_expert = sum of projection strides (bytes one expert costs
// across gate/up/down). Returns experts/layer in [0, n_expert]; 0 if unmeasurable (tier stays off).
//   LLAMA_MOE_RAM_CAP set (>0)   -> honored verbatim, this is not called
//   LLAMA_MOE_RAM_FRAC=F         -> hard ceiling as a fraction of avail; only binds when the reserve is
//                                   set too low to be safe on its own (default 0.97)
//   LLAMA_MOE_RAM_RESERVE_MB=N   -> MiB of available RAM left free; exact (default 5120 = 5 GiB)
static int llama_moe_auto_ram_capacity(const std::vector<llama_moe_proj_store> & proj, int n_expert) {
    if (proj.empty() || n_expert <= 0) { return 0; }
    // Compute ONCE and reuse for every layer (same reason as llama_moe_auto_capacity): this is called
    // per layer as caches are created, and avail RAM shrinks as each layer's pool allocates - so
    // recomputing would shrink the cap for later layers and under-fill RAM. Cache the first (max-avail)
    // result so all layers get a uniform pool size.
    static int cached_rc = 0;
    if (cached_rc > 0) { return cached_rc; }

    const uint64_t avail = llama_moe_avail_ram();
    if (avail == 0) { return 0; }

    uint64_t per_expert = 0; // bytes one resident expert costs across all projections of a layer
    for (const auto & pr : proj) { per_expert += (uint64_t) pr.stride; }
    if (per_expert == 0) { return 0; }

    // Derive the MoE-layer count from the registered expert files (one entry per exps tensor, n_proj per
    // layer). At the first layer's cache creation only this cache exists, but all expert files are
    // registered up front at load time, so this is the true layer count - letting us budget for every
    // layer's pool before the rest are allocated.
    int n_moe_layers = proj.empty() ? 0 : (int) (g_moe_expert_files.size() / proj.size());
    if (n_moe_layers < 1) { n_moe_layers = 1; }

    double frac = 0.97;
    if (const char * fe = getenv("LLAMA_MOE_RAM_FRAC")) {
        const double f = atof(fe);
        if (f > 0.05 && f < 0.98) { frac = f; }
    }
    // The reserve is EXACT: the pool leaves this much AVAILABLE RAM (not total) free, so the OS and the
    // mmap working set of the model file - which grows as experts are touched - keep room. frac is only a
    // backstop for a reserve set too low to be safe on its own.
    uint64_t reserve = (uint64_t) 5120 * 1024 * 1024;
    if (const char * re = getenv("LLAMA_MOE_RAM_RESERVE_MB")) {
        const long long r = atoll(re);
        if (r >= 0) { reserve = (uint64_t) r * 1024 * 1024; }
    }
    const uint64_t by_reserve = avail > reserve ? avail - reserve : 0;
    uint64_t       usable     = std::min(by_reserve, (uint64_t) (frac * (double) avail));
    if (usable == 0) { return 0; }

    // Second bound, and the one that actually enforces "never pageable": ask the OS to raise this
    // process's minimum working set enough to LOCK the whole pool, and clamp to what it grants. Taking the
    // min of the two can only ever SHRINK the pool relative to the avail-based rule above, so this cannot
    // introduce an oversizing regression - it removes exactly the part that would have been pageable, i.e.
    // the part whose only eviction target is the pagefile.
    const uint64_t granted = llama_moe_ram_quota_reserve(usable);
    if (granted < usable) { usable = granted; }
    if (usable == 0) { return 0; }

    const uint64_t denom = (uint64_t) n_moe_layers * per_expert;
    long long cap = (long long) (usable / denom);
    if (cap < 0)        { cap = 0; }
    if (cap > n_expert) { cap = n_expert; }
    static bool logged = false;
    if (!logged && cap > 0) {
        logged = true;
        const double gib = 1024.0*1024.0*1024.0;
        const double pool_gib = (double) ((uint64_t) cap * denom) / gib;
        LLAMA_LOG_WARN("MoE stream: auto RAM_CAP=%lld -> ~%.1f GiB host pool across %d MoE layers "
                       "(avail %.1f GiB, reserve %.1f GiB for OS/mmap, frac %.2f, %.1f MiB/expert). "
                       "Override with LLAMA_MOE_RAM_CAP; tune LLAMA_MOE_RAM_FRAC / LLAMA_MOE_RAM_RESERVE_MB.\n",
                       cap, pool_gib, n_moe_layers, avail/gib, reserve/gib, frac, per_expert/(1024.0*1024.0));
    }
    if (cap > 0) { cached_rc = (int) cap; } // reuse for all remaining layers
    return (int) cap;
}

// Look up an EXISTING per-layer cache by its first projection tensor (the map key), without creating
// one. Used by the cross-layer prefetch to reach layer L+1's cache from layer L's graph build. Returns
// null if that layer's cache has not been created yet (e.g. during the reserve pass).
llama_moe_layer_cache * llama_moe_layer_cache_lookup(const ggml_tensor * exps0) {
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    auto it = g_moe_layer_caches.find(exps0);
    return it != g_moe_layer_caches.end() ? it->second : nullptr;
}

void llama_moe_set_defer_caches(bool v) {
    g_moe_defer_caches.store(v, std::memory_order_release);
}

void llama_moe_set_compute_reserve(size_t bytes) {
    g_moe_compute_reserve.store(bytes, std::memory_order_release);
}

llama_moe_layer_cache * llama_moe_layer_cache_get(ggml_backend_sched_t sched,
                                                  ggml_tensor * const * exps_list,
                                                  int                   n_proj,
                                                  ggml_tensor *         selected_experts,
                                                  int                   capacity) {
    // measurement pass: no cache yet, so build_moe_ffn takes the compaction path (see g_moe_defer_caches)
    if (g_moe_defer_caches.load(std::memory_order_acquire)) {
        return nullptr;
    }    if (n_proj <= 0 || !exps_list || !exps_list[0]) {
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
    //
    // The prefill group sweep NEEDS n_used sentinels and defaults to that: in a given group pass every
    // position routed outside the group must resolve to a genuine zero slab, and a token can have all
    // n_used positions outside the group. With fewer sentinels the overflow falls back to a spare
    // RESIDENT slot, i.e. a wrong expert's real weights, which would break the sweep's losslessness
    // (that contribution must be exactly 0 for the per-group sum to reconstruct the full output).
    // Number of physical zero sentinel slots. Resolved by llama_moe_sentinel_count so the
    // auto-capacity budget (which must subtract the same number) cannot disagree with what is
    // actually allocated here.
    const int n_sentinel = llama_moe_sentinel_count(n_used);

    // How many slots the prewarm below actually fills. The recap (llama_moe_recap_from_compute_reserve)
    // frees every cache built during its measurement reserve pass, so filling all `capacity` slots there
    // copies `capacity` experts per layer into VRAM for a cache nothing ever computes with - measured
    // 14.1 GiB / 2.7s of startup on qwen3.8-flash-next (page cache -> VRAM at ~3.2 GiB/s; the bytes are
    // already resident, so it does not show up as disk reads). During that pass fill slot 0 only; the slot
    // count, buffer size and graph shape are untouched, so the compute-buffer size the recap is measuring
    // stays bit-identical. slot 0 is still filled because stale_table requires every expert to map to a
    // REAL settled slot. If the recap declines after all, llama_moe_cache_defer_prewarm(false) fills the
    // rest.
    const int  n_prewarm       = (g_moe_defer_prewarm.load(std::memory_order_acquire) && capacity > 1) ? 1 : capacity;
    const bool prewarm_pending = n_prewarm < capacity;

    llama_moe_layer_cache * c = new llama_moe_layer_cache();
    c->sched      = sched; // owner tag, see the field comment
    c->n_expert   = n_expert;
    c->capacity   = capacity;
    c->n_used     = n_used;
    c->n_sentinel = n_sentinel;
    c->prewarm_pending = prewarm_pending;
    // Layer index, parsed from the key tensor's "blk.<il>." prefix. Diagnostics only (LLAMA_MOE_LAYERDBG):
    // the cache is otherwise layer-agnostic, but a per-layer residency profile is the only way to see that
    // some layers are structurally harder than others - e.g. DeepSeek-V4's first dsv4_hash_layer_count
    // layers route by a static token-id -> expert table, so their locality is the OUTPUT TOKEN's locality
    // rather than the hidden-state locality every other layer enjoys.
    if (key && key->name[0]) {
        int parsed = -1;
        if (sscanf(key->name, "blk.%d.", &parsed) == 1 && parsed >= 0) { c->il = parsed; }
    }
    // Weight-aware residency: the background loader protects high-score (recurring top-1/top-2)
    // experts from eviction so the stable high-weight core stays resident on the pure-GPU decode
    // path (rare misses drop to the sentinel). On by default; disable with LLAMA_MOE_WEIGHTED_EVICT=0.
    c->weighted_evict = true;
    if (const char * w_env = getenv("LLAMA_MOE_WEIGHTED_EVICT")) { c->weighted_evict = atoi(w_env) != 0; }
    c->slot_expert.assign((size_t) capacity, -1);
    c->slot_age.assign((size_t) capacity, 0);
    c->slot_pin.assign((size_t) capacity, 0);
    c->slot_loading.assign((size_t) capacity, 0);
    c->expert_slot.assign((size_t) n_expert, -1);
    c->expert_score.assign((size_t) n_expert, 0.0f);
    if (getenv("LLAMA_MOE_EVICTDBG")) { c->fevict_step.assign((size_t) n_expert, 0); }
    // warm-start residency: if a persisted score file (LLAMA_MOE_SCORE_FILE) has this layer's learned
    // scores, seed them so the loader fills the hot experts from tick 0 instead of by-index (skips the
    // ~1400-token cold learning ramp). Hint only - correctness is unaffected.
    llama_moe_scores_seed(exps_list[0], c->expert_score, n_expert);
    // seed each routing position u with a distinct real prewarm slot (slot u holds expert u after
    // prewarm below), so stale reuse starts from real experts and the n_used positions never collide.
    c->pos_slot.assign((size_t) n_used, 0);
    for (int u = 0; u < n_used; ++u) { c->pos_slot[(size_t) u] = (u < n_prewarm) ? u : 0; }
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
    if (moe_env_on("LLAMA_MOE_NOMMAP", true)) {
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
    // disk. LLAMA_MOE_RAM_CAP = experts/layer to hold in RAM. Unset/0 => auto-sized from available
    // system RAM (see llama_moe_auto_ram_capacity); a disk-streamed model always benefits from as large
    // a RAM tier as fits, so auto is the sensible default rather than "off". Clamp to n_expert.
    {
        const char * rc_env = getenv("LLAMA_MOE_RAM_CAP");
        int rc = rc_env ? atoi(rc_env) : 0;
        if (!rc_env || rc <= 0) {
            rc = llama_moe_auto_ram_capacity(c->proj, n_expert); // 0 if unmeasurable => tier stays off
        }
        if (rc > n_expert) { rc = n_expert; }
        if (rc > 0) {
            // Reserve the working-set quota for EVERY layer's pool up front, not just this one: pools are
            // created a layer at a time but the lock budget is process-wide, so asking per layer would let
            // the early layers spend it all and leave the late ones unpinnable (and therefore refused).
            // Idempotent, so this is a no-op when llama_moe_auto_ram_capacity already asked - it matters on
            // the explicit LLAMA_MOE_RAM_CAP path, which skips the auto sizing entirely.
            {
                uint64_t per_expert = 0;
                for (const auto & pr : c->proj) { per_expert += (uint64_t) pr.stride; }
                int n_moe_layers = n_proj > 0 ? (int) (g_moe_expert_files.size() / (size_t) n_proj) : 1;
                if (n_moe_layers < 1) { n_moe_layers = 1; }
                (void) llama_moe_ram_quota_reserve((uint64_t) rc * per_expert * (uint64_t) n_moe_layers);
            }
            bool ok         = true;
            bool all_pinned = true;
            for (int i = 0; i < n_proj && ok; ++i) {
                bool pinned = false;
                c->proj[(size_t) i].ram = llama_moe_ram_alloc((size_t) rc * (size_t) c->proj[(size_t) i].stride, &pinned);
                if (!c->proj[(size_t) i].ram) { ok = false; }
                if (!pinned) { all_pinned = false; }
            }
            if (ok) {
                c->ram_capacity = rc;
                c->ram_slot.assign((size_t) n_expert, -1);
                c->ram_expert.assign((size_t) rc, -1);
                c->ram_score.assign((size_t) rc, 0.0f);
                // Once an expert is in the RAM pool, drop its mmap page-cache copy so both do not occupy
                // RAM - but ONLY if that pool is actually PINNED. On an unpinned pool the trade runs
                // backwards: it destroys a copy the OS could have evicted for free (and re-read from the
                // model file) in favour of one the OS can only evict to the pagefile. Reachable only with
                // LLAMA_MOE_RAM_REQUIRE_PIN=0. Force either way with LLAMA_MOE_RAM_EVICT.
                c->evict_after_ram = all_pinned;
                if (const char * ev = getenv("LLAMA_MOE_RAM_EVICT")) { c->evict_after_ram = atoi(ev) != 0; }
            } else {
                // partial alloc failed: free whatever we got, run without the RAM tier
                for (int i = 0; i < n_proj; ++i) {
                    if (c->proj[(size_t) i].ram) {
                        llama_moe_ram_free(c->proj[(size_t) i].ram, (size_t) rc * (size_t) c->proj[(size_t) i].stride);
                        c->proj[(size_t) i].ram = nullptr;
                    }
                }
                LLAMA_LOG_WARN("MoE stream: no RAM residency pool for this layer (wanted %d experts/layer); "
                               "misses read the expert from the model file. If a 'refusing the host expert "
                               "pool' line appeared above, the allocation succeeded but could not be PINNED - "
                               "lower LLAMA_MOE_RAM_CAP or raise LLAMA_MOE_RAM_RESERVE_MB.\n", rc);
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
    // 0..n_prewarm-1 with experts 0..n_prewarm-1; the loader adapts to the real working set from there
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
        for (int s = 0; s < n_prewarm; ++s) {
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
        table[(size_t) e] = (e < n_prewarm) ? e : capacity; // missing -> zero sentinel slot
    }
    ggml_backend_tensor_set(c->slot_table, table.data(), 0, (size_t) n_expert * sizeof(int32_t));

    // stale_table: every expert maps to a REAL slot (never the sentinel). Resident experts (e<n_prewarm,
    // prewarmed into slot e) map to their own slot; the rest borrow slot (e%n_prewarm), which after
    // prewarm holds a real expert. Decode's get_rows reads this, so a miss reuses a real (stale) expert
    // instead of zeroing the branch. The loader/boundary sync keep it current via moe_stale_on_*.
    c->stale_of_expert.assign((size_t) n_expert, 0);
    for (int e = 0; e < n_expert; ++e) {
        c->stale_of_expert[(size_t) e] = (e < n_prewarm) ? e : (n_prewarm > 0 ? e % n_prewarm : 0);
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

bool llama_moe_layer_cache_backend_is_cuda(llama_moe_layer_cache * c) {
    if (!c || !c->dev_backend) {
        return false;
    }
    const char * name = ggml_backend_name(c->dev_backend);
    return name && strstr(name, "CUDA") != nullptr;
}

// fork: read-only probe callback on the normalized gate weights. `a` is [1, n_expert_used, n_tokens],
// each token's n_expert_used weights sum to 1 and are in descending order (argsort). Bins the top-1 and
// top-1+top-2 weight to measure how "flat" hy3's routing is. dst is unused (no-op op).
static void llama_moe_weight_diag_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) dst; (void) nth; (void) userdata;
    if (ith != 0) { return; }
    const int64_t n_used   = a->ne[1];
    const int64_t n_tokens = a->ne[2];
    if (n_used < 1 || a->type != GGML_TYPE_F32 || !a->data) { return; }
    const char * base = (const char *) a->data;
    for (int64_t t = 0; t < n_tokens; ++t) {
        // read this token's n_used weights; element (0, u, t) at offset u*nb[1] + t*nb[2]. The weights may
        // carry an arbitrary post-norm scale (hy3 expert_weights_scale=2.826), so normalize by their sum
        // here to get a scale-invariant fraction (top-1's SHARE of the routed mass, in [0,1]).
        double sum = 0.0; float w0 = 0.0f, w1 = 0.0f;
        for (int64_t u = 0; u < n_used; ++u) {
            const float w = *(const float *) (base + u * a->nb[1] + t * a->nb[2]);
            sum += w;
            if (u == 0) { w0 = w; }
            if (u == 1) { w1 = w; }
        }
        if (sum <= 0.0) { continue; }
        const float f0 = (float) (w0 / sum);          // top-1 share of routed weight
        const float f01 = (float) ((w0 + w1) / sum);  // top-1+top-2 share
        g_wdiag_tokens.fetch_add(1, std::memory_order_relaxed);
        if (f0 < 0.20f) { g_wdiag_top1_lt20.fetch_add(1, std::memory_order_relaxed); }
        if (f0 < 0.15f) { g_wdiag_top1_lt15.fetch_add(1, std::memory_order_relaxed); }
        if (f0 > 0.40f) { g_wdiag_top1_gt40.fetch_add(1, std::memory_order_relaxed); }
        g_wdiag_top2sum_x1000.fetch_add((uint64_t) (f01 * 1000.0f), std::memory_order_relaxed);
    }
    // print every ~2560 token-layers (=32 tokens x 80 layers) so it shows without spamming
    const uint64_t tok = g_wdiag_tokens.load(std::memory_order_relaxed);
    static std::atomic<uint64_t> last_print{0};
    uint64_t lp = last_print.load(std::memory_order_relaxed);
    if (tok - lp >= 2560 && last_print.compare_exchange_strong(lp, tok)) {
        const uint64_t n = tok ? tok : 1;
        LLAMA_LOG_WARN("MoE weight-diag: %llu token-layers | top1<0.20: %.0f%%  top1<0.15: %.0f%%  "
                       "top1>0.40: %.0f%%  mean(top1+top2)=%.2f\n",
                       (unsigned long long) tok,
                       100.0 * (double) g_wdiag_top1_lt20.load() / (double) n,
                       100.0 * (double) g_wdiag_top1_lt15.load() / (double) n,
                       100.0 * (double) g_wdiag_top1_gt40.load() / (double) n,
                       (double) g_wdiag_top2sum_x1000.load() / 1000.0 / (double) n);
    }
}

void llama_moe_weight_diag_probe(ggml_context * ctx0, ggml_cgraph * gf, ggml_tensor * weights) {
    if (!getenv("LLAMA_MOE_WEIGHT_DIAG") || !weights || !gf) { return; }
    // map_custom1 reads weights and writes a throwaway dst; expand it directly so it runs. Its output is
    // never consumed, so it cannot affect the model's computation - purely observational.
    ggml_tensor * probe = ggml_map_custom1(ctx0, weights, llama_moe_weight_diag_cb, 1, nullptr);
    ggml_build_forward_expand(gf, probe);
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

void llama_moe_layer_cache_prefetch(llama_moe_layer_cache * next_c,
                                    ggml_context *          ctx0,
                                    ggml_cgraph *           gf,
                                    ggml_tensor *           pred) {
    if (!next_c || !next_c->sel_buf || !pred) {
        return;
    }
    // Publish the predicted selection into the NEXT layer's sel_buf so its background loader warms the
    // candidates during THIS layer's compute. Copy the leading sel_buf->ne[0] rows of `pred` (the
    // highest-weight predicted experts, since pred is argsort output). pred is [k, n_tokens]; take a
    // view of its first n_used rows to match sel_buf's width. Column count must fit sel_buf.
    const int64_t n_used = next_c->sel_buf->ne[0];
    if (pred->ne[0] < n_used || pred->ne[1] > next_c->sel_buf->ne[1]) {
        return;
    }
    ggml_tensor * src = pred;
    if (pred->ne[0] != n_used) {
        src = ggml_view_2d(ctx0, pred, n_used, pred->ne[1], pred->nb[1], 0);
    }
    ggml_tensor * dst = next_c->sel_buf;
    if (pred->ne[1] != next_c->sel_buf->ne[1]) {
        dst = ggml_view_2d(ctx0, next_c->sel_buf, n_used, pred->ne[1], next_c->sel_buf->nb[1], 0);
    }
    ggml_tensor * cpy = ggml_cpy(ctx0, src, dst);
    ggml_build_forward_expand(gf, cpy);
}

// Ensure the cache's pinned host staging arena is allocated for `n_threads` io threads (one stride-sized
// slot per (thread, projection)). Backend-agnostic: uses the device's host-buffer type, which is real
// page-locked memory on CUDA. Returns the base ptr and per-slot stride via the cache fields, or leaves
// stage_base=nullptr on failure (caller falls back to a pageable heap buffer). Caller holds the mutex.
// page-locked memory on CUDA. Grows the single GLOBAL arena (g_moe_stage_*) to fit this layer's needs and
// leaves g_moe_stage_base=nullptr on failure (caller falls back to a pageable heap buffer). Caller holds
// g_moe_layer_mutex, and parallel_load runs one layer at a time under it, so the shared arena is race-free.
static void llama_moe_ensure_stage(llama_moe_layer_cache * c, int n_threads) {
    if (!c->stage_dev || n_threads < 1) { return; }
    // per-slot bytes = max projection stride (so any proj's expert fits)
    size_t slot = 0;
    for (auto & pr : c->proj) { if (pr.stride > slot) { slot = (size_t) pr.stride; } }
    if (slot == 0) { return; }
    const int n_proj = (int) c->proj.size();
    // High-water sizing: keep whichever of slot / n_threads / n_proj is largest across all layers, so the
    // single global arena is reused rather than reallocated per layer (and never shrinks under a layer).
    if (g_moe_stage_base &&
        g_moe_stage_slot >= slot && g_moe_stage_nthread >= n_threads && g_moe_stage_nproj >= n_proj) {
        return; // already big enough
    }
    const size_t new_slot   = slot        > g_moe_stage_slot    ? slot        : g_moe_stage_slot;
    const int    new_thread = n_threads   > g_moe_stage_nthread ? n_threads   : g_moe_stage_nthread;
    const int    new_proj   = n_proj      > g_moe_stage_nproj   ? n_proj      : g_moe_stage_nproj;
    if (g_moe_stage_buf) { ggml_backend_buffer_free(g_moe_stage_buf); g_moe_stage_buf = nullptr; g_moe_stage_base = nullptr; }
    ggml_backend_buffer_type_t hbuft = ggml_backend_dev_host_buffer_type(c->stage_dev);
    if (!hbuft) { return; } // device has no pinned host buffer type; caller uses pageable fallback
    const size_t n_slots = (size_t) new_thread * (size_t) new_proj;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(hbuft, n_slots * new_slot);
    if (!buf) { return; }
    g_moe_stage_buf     = buf;
    g_moe_stage_base    = (char *) ggml_backend_buffer_get_base(buf);
    g_moe_stage_slot    = new_slot;
    g_moe_stage_nthread = new_thread;
    g_moe_stage_nproj   = new_proj;
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
// allow_pinned=false forces the pageable per-thread read buffer instead of the GLOBAL pinned staging arena.
// The arena is only race-free because every other caller runs parallel_load under g_moe_layer_mutex; the
// opportunistic prefetch drain deliberately runs UNLOCKED (its whole point is not to stall the decode
// callbacks), so it must not touch the arena. It keeps the two things that actually make this fast: the
// 16-way private-handle reads and the async-H2D-plus-one-synchronize batching.
static void llama_moe_layer_parallel_load(llama_moe_layer_cache *   c,
                                          const std::vector<int> &  load_e,
                                          const std::vector<int> &  load_slot,
                                          bool                      allow_pinned = true,
                                          bool                      from_loader  = false) {
    const size_t nl = load_e.size();
    if (nl == 0) {
        return;
    }
    const bool diag = getenv("LLAMA_MOE_DIAG") != nullptr; // split read vs H2D timing (parallel_load audit)
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

    // The work unit is (expert, projection), not (expert). At decode a layer loads 0-1 experts per token,
    // so splitting by expert made n_threads collapse to 1 and the n_proj slabs of that one expert were
    // read back to back on a single thread. Splitting by unit lets even a single-expert load use n_proj
    // threads. Prefill is unaffected in kind: nu is just n_proj times larger and still clamps to the env.
    //
    // Going FINER than one projection slab was tried and is a dead end: striping a slab into chunks made
    // read-phase bandwidth worse in 24 of 24 paired windows at both 512 KiB and 256 KiB, because each chunk
    // costs its own seek plus its own synchronous H2D (a full cudaStreamSynchronize), and a 2.4 MiB request
    // already amortises that. One projection slab is the right granularity.
    const size_t np = c->proj.size() ? c->proj.size() : 1;
    const size_t nu = nl * np;
    const char * env = getenv("LLAMA_MOE_IO_THREADS");
    int n_threads = env ? atoi(env) : 16;
    if (n_threads < 1) { n_threads = 1; }
    if (n_threads > (int) nu) { n_threads = (int) nu; }

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

    // pinned staging arena (global, shared across layers); if it fails, threads use a pageable buffer
    if (fast && allow_pinned) { llama_moe_ensure_stage(c, n_threads); }
    const bool use_pinned = fast && allow_pinned && g_moe_stage_base != nullptr;

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
            // opened lazily below: a unit-split thread usually touches ONE projection, so eagerly opening
            // all of them cost n_proj fopen per thread per call for handles that were never used.
            tfp.resize(c->proj.size(), nullptr);
        }
        for (size_t u = lo; u < hi; ++u) {
            const size_t idx = u / np;
            const size_t pi  = u % np;
            const int e    = load_e[order[idx]];
            const int slot = load_slot[order[idx]];
            {
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
                    stg = g_moe_stage_base + si * g_moe_stage_slot;
                } else {
                    scratch.resize(stride);
                    stg = scratch.data();
                }
                if (par_read && !tfp[pi] && !pr.fpath.empty()) {
                    tfp[pi] = fopen(pr.fpath.c_str(), "rb"); // first unit of this projection on this thread
                }
                bool ok;
                const int64_t t_rd0 = diag ? ggml_time_us() : 0;
                if (par_read && tfp[pi]) {
                    ok = llama_moe_pread(tfp[pi], pr.foff0 + (uint64_t) e * stride, stg, stride);
                } else {
                    // single-thread / no private handle: use the shared read path (holds compute fp)
                    ok = llama_moe_read_expert(c, pi, e, stg);
                }
                if (diag) {
                    const uint64_t dt = (uint64_t) (ggml_time_us() - t_rd0);
                    const uint64_t by = ok ? (uint64_t) stride : 0;
                    if (from_loader) { g_diag_ldr_read_ns += dt; g_diag_ldr_cnt += 1; g_diag_ldr_bytes  += by; }
                    else            { g_diag_read_ns     += dt; g_diag_read_bytes += by; }
                }
                if (ok) {
                    // NOTE: in async mode a pinned staging slot is reused per (thread,proj) across
                    // experts, so we must issue its H2D before the next expert overwrites it. With the
                    // pinned arena the async copy reads the slot immediately on enqueue on the same
                    // stream, but to be safe against reuse we issue staged H2D synchronously here and
                    // only async-batch the direct-source (RAM/mmap, stable pointer) copies above.
                    const int64_t t_h0 = diag ? ggml_time_us() : 0;
                    ggml_backend_tensor_set(pr.dev, stg, (size_t) slot * stride, stride);
                    if (diag) {
                        const uint64_t dt = (uint64_t) (ggml_time_us() - t_h0);
                        if (from_loader) { g_diag_ldr_h2d_ns += dt; }
                        else            { g_diag_h2d_ns     += dt; }
                    }
                }
            }
        }
        for (FILE * f : tfp) { if (f) { fclose(f); } }
        if (async_h2d && !local.empty()) {
            std::lock_guard<std::mutex> lk(pend_mu);
            pend.insert(pend.end(), local.begin(), local.end());
        }
    };

    const int64_t t_rp0 = diag ? ggml_time_us() : 0;
    if (n_threads <= 1) {
        load_range(0, nu, 0);
    } else {
        // spawn n_threads-1 and run the last range on the calling thread: one fewer std::thread per
        // parallel_load call, which matters because decode calls this once per MoE layer per token.
        std::vector<std::thread> pool;
        pool.reserve((size_t) n_threads - 1);
        for (int t = 0; t < n_threads - 1; ++t) {
            const size_t lo = (size_t) ((long long) t       * (long long) nu / n_threads);
            const size_t hi = (size_t) ((long long) (t + 1) * (long long) nu / n_threads);
            if (lo < hi) {
                pool.emplace_back(load_range, lo, hi, t);
            }
        }
        {
            const int    t  = n_threads - 1;
            const size_t lo = (size_t) ((long long) t       * (long long) nu / n_threads);
            const size_t hi = (size_t) ((long long) (t + 1) * (long long) nu / n_threads);
            if (lo < hi) { load_range(lo, hi, t); }
        }
        for (auto & th : pool) {
            th.join();
        }
    }
    // WALL time of the read phase just completed (spawn + reads + staged H2D + join). Together with the
    // H2D batch below this closes parallel_load, so the printed decomposition is a real sum.
    if (diag) {
        const uint64_t dt = (uint64_t) (ggml_time_us() - t_rp0);
        if (from_loader) { g_diag_ldr_rdphase_ns += dt; }
        else             { g_diag_rdphase_ns     += dt; }
    }

    // async H2D batch: issue all direct-source copies on the device stream from THIS thread, then a
    // single synchronize (instead of one cudaStreamSynchronize per copy). Sources are stable host
    // pointers (RAM pool / mmap), safe to read after the reads completed above.
    if (async_h2d && !pend.empty()) {
        const int64_t t_b0 = diag ? ggml_time_us() : 0;
        for (const auto & p : pend) {
            ggml_backend_tensor_set_async(c->dev_backend, p.dev, p.src, p.off, p.bytes);
        }
        ggml_backend_synchronize(c->dev_backend);
        // wall time on the issuing thread, so it goes to its own counter: g_diag_h2d_ns holds thread-sums
        // from the staged path and mixing the two units made the printed H2D figure uninterpretable.
        if (diag) {
            const uint64_t dt = (uint64_t) (ggml_time_us() - t_b0);
            if (from_loader) { g_diag_ldr_h2d_ns += dt; }
            else             { g_diag_h2d_wall_ns += dt; }
        }
    }
}

// map_custom1 callback: remap this step's routing ids [n_used,n_tokens] to cache-slot ids,
// running for BOTH prefill and decode. Correctness contract (CUDA single-GPU): a map_custom op
// is always CPU-assigned, so the scheduler runs it as its own split that fully completes before
// the downstream mul_mat_id GPU split is enqueued; the synchronous ggml_backend_tensor_set here
// blocks the host until the H2D copy lands, so every slot this callback resolves is resident in
// VRAM before the matmul that reads these ids runs. See llama_moe_cache_remap_cb for the same
static void llama_moe_layer_remap_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// fork: coverage-mode remap callback (LLAMA_MOE_SYNC_COVER). `a` = selected_experts [n_used, 1] (argsort,
// descending gate weight); `b` = normalized weights [1, n_used, 1]. Computes this step's per-expert weight
// share into c->step_wexp and sets c->step_cover, then delegates to remap_cb, which applies the rule:
// sync missing experts weight-descending but STOP once (already-cached + synced) weight coverage reaches
// the threshold, AND never sync more than sync_budget (top-2) misses. So the sync count is
// min(top-2, experts-needed-to-reach-coverage): the top-2 cap guarantees the proven quality floor, and
// coverage lets a step whose cached experts already cover enough weight skip even those loads. Directly
// handles the user's case (cached experts' weight already high -> load fewer/none) without ever exceeding
// the safe budget.
static void llama_moe_layer_remap_cov_cb(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
                                         int ith, int nth, void * userdata) {
    (void) nth;
    llama_moe_layer_cache * c = (llama_moe_layer_cache *) userdata;
    if (ith == 0 && c && b && b->type == GGML_TYPE_F32 && b->data && a->ne[1] == 1) {
        const int64_t n_used = a->ne[0];
        const int32_t * sel = (const int32_t *) a->data;
        const char * wbase = (const char *) b->data;
        double wsum = 0.0;
        for (int64_t u = 0; u < n_used; ++u) { wsum += *(const float *) (wbase + u * b->nb[1]); }
        if ((int) c->step_wexp.size() != c->n_expert) { c->step_wexp.assign((size_t) c->n_expert, 0.0f); }
        else { std::fill(c->step_wexp.begin(), c->step_wexp.end(), 0.0f); }
        double cached_cover = 0.0;
        if (wsum > 0.0) {
            for (int64_t u = 0; u < n_used; ++u) {
                const int32_t e = sel[u];
                if (e < 0 || e >= c->n_expert) { continue; }
                const float w = (float) (*(const float *) (wbase + u * b->nb[1]) / wsum);
                c->step_wexp[(size_t) e] = w;
                if (c->expert_slot[(size_t) e] >= 0) { cached_cover += w; }
            }
        }
        c->step_cover = c->sync_cover; // active threshold for remap_cb this step
        // diagnostic (LLAMA_MOE_COVDBG only - coverage is default-on now, so this must not spam the
        // default path): how many the coverage rule will sync (min(budget, needed-to-reach-cover))
        if (getenv("LLAMA_MOE_COVDBG")) {
            g_cov_tokens.fetch_add(1, std::memory_order_relaxed);
            g_cov_cached_x1000.fetch_add((uint64_t) (cached_cover * 1000.0), std::memory_order_relaxed);
            if (cached_cover >= c->sync_cover) { g_cov_zero_sync.fetch_add(1, std::memory_order_relaxed); }
            const uint64_t tok = g_cov_tokens.load(std::memory_order_relaxed);
            static std::atomic<uint64_t> last{0};
            uint64_t lp = last.load(std::memory_order_relaxed);
            if (tok - lp >= 2560 && last.compare_exchange_strong(lp, tok)) {
                const uint64_t nn = tok ? tok : 1;
                LLAMA_LOG_WARN("MoE cover-diag: %llu token-layers | cached-cover mean=%.0f%% | "
                               "already-covered(zero-sync) %.0f%% of tokens | synced mean=%.2f/token\n",
                               (unsigned long long) tok,
                               100.0 * (double) g_cov_cached_x1000.load() / 1000.0 / (double) nn,
                               100.0 * (double) g_cov_zero_sync.load() / (double) nn,
                               (double) g_cov_synced.load() / (double) nn);
            }
        }
    }
    llama_moe_layer_remap_cb(dst, a, ith, nth, userdata);
}

// fork: per-token freshness stride (LLAMA_MOE_SYNC_STRIDE=N, N<=1 = off). Every N-th single-token decode
// step keeps the ordinary synchronous expert budget; the N-1 steps between it inspect no routing position
// and fall through to the existing per-position stale reuse (pos_slot). State lives here and is written
// ONCE PER DECODE STEP from llama_context::decode(); the remap callback runs once per MoE layer and only
// reads it - a counter advanced in the callback would stride LAYERS inside one token, which is a
// different experiment that still produces plausible ms/token numbers.
static std::atomic<bool>     g_stride_freefly{false};
static std::atomic<uint64_t> g_stride_applied{0}; // layer-steps that really ran free-fly (no-op detector)
static std::atomic<int>      g_stride_keep_il{0}; // layers with il < this never free-fly

void llama_moe_stride_begin_step(bool single_token, int32_t tok, int32_t stop_tok) {
    static const int stride = []() {
        const char * e = getenv("LLAMA_MOE_SYNC_STRIDE");
        const int    v = e ? atoi(e) : 1;
        return v > 1 ? v : 1; // <=1 disables; 0 and negatives are not a "skip everything" mode
    }();
    static const bool think_only = []() {
        const char * e = getenv("LLAMA_MOE_SYNC_STRIDE_THINK_ONLY");
        return e && atoi(e) != 0;
    }();
    static const int keep_il = []() {
        const char * e = getenv("LLAMA_MOE_SYNC_STRIDE_KEEP_LAYERS");
        const int    v = e ? atoi(e) : 0;
        return v > 0 ? v : 0; // 0 = stride every layer (default)
    }();
    static uint64_t step        = 0;
    static bool     think_ended = false;

    g_stride_keep_il.store(keep_il, std::memory_order_relaxed);

    // Any multi-token step is a prompt or re-prompt: disarm and restart the phase. This is the only
    // thing that keeps prompt processing off the stride (the callback's own n_tokens is 1 for every
    // ubatch of a -ub 1 prompt) and the only thing that makes the first generated token of turn 2 and
    // later fresh, since c->pos_init is never cleared again after cache creation.
    if (stride <= 1 || !single_token) {
        step        = 0;
        think_ended = false;
        g_stride_freefly.store(false, std::memory_order_relaxed);
        return;
    }
    // reasoning-close token was just fed back in: the answer span starts here, stay fresh from now on
    if (think_only && stop_tok >= 0 && tok == stop_tok) {
        think_ended = true;
    }
    if (think_only && think_ended) {
        g_stride_freefly.store(false, std::memory_order_relaxed);
        return;
    }
    // Steps 0 and 1 always keep the budget: llama_moe_refill_vram_caches() consumes the post-sweep dirty
    // flag after step 0 and rewrites up to cap experts per layer WITHOUT honouring slot_pin, so a
    // free-fly step 1 would reuse donors that pos_slot recorded before the refill moved them.
    const uint64_t s = step++;
    g_stride_freefly.store(s >= 2 && (s % (uint64_t) stride) != 0, std::memory_order_relaxed);

    if (s == 16) {
        const uint64_t applied = g_stride_applied.load(std::memory_order_relaxed);
        if (applied == 0) {
            LLAMA_LOG_WARN("MoE stride: LLAMA_MOE_SYNC_STRIDE=%d is set but no free-fly layer-step "
                           "engaged - the per-layer CPU remap is not on this decode graph (it needs "
                           "LLAMA_MOE_SYNC_BUDGET>0)\n", stride);
        } else {
            LLAMA_LOG_WARN("MoE stride: N=%d keep_il=%d - %llu free-fly layer-steps in the first 16 "
                           "decode steps\n", stride, g_stride_keep_il.load(std::memory_order_relaxed),
                           (unsigned long long) applied);
        }
    }
}

bool llama_moe_stride_freefly(void) {
    return g_stride_freefly.load(std::memory_order_relaxed);
}

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

    // One-shot RAM-pool prefill (LLAMA_MOE_PREFILL): fire on the VERY FIRST remap callback, which is the
    // first prompt-eval layer (n_tokens>1) - by then every layer cache and its RAM pool already exist (the
    // whole graph is built, so all 80 caches registered, before any compute runs). Filling here charges the
    // ~4s one-time read to the prompt-eval/startup window instead of the first decode token, so the reported
    // decode tok/s reflects the warm steady state a user actually feels, not the startup cost. Runs once,
    // before this callback takes the mutex. Skips the ~minute-long serial background-loader warm-up ramp.
    // The flag is shared with the sweep's group callback - see llama_moe_prefill_once.
    llama_moe_prefill_once();

    std::vector<int32_t> flat;
    llama_moe_flatten_ids(a, flat);

    const int64_t diag_lock0 = ggml_time_us();
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    const int64_t diag_t0 = ggml_time_us();
    if (getenv("LLAMA_MOE_PFDBG") && n_tokens == 1 && c->il >= 0 && c->il < 3 && flat.size() >= 6) {
        int res_n = 0;
        for (int i = 0; i < 6; ++i) {
            const int32_t e = flat[(size_t) i];
            if (e >= 0 && e < c->n_expert && c->expert_slot[(size_t) e] >= 0) { res_n++; }
        }
        LLAMA_LOG_WARN("MoE PFDBG callback il=%d ids=%d,%d,%d,%d,%d,%d resident=%d/6\n", c->il,
                       flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], res_n);
    }
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
        // "latent" because diag_miss counts EVERY unique missing expert of the step, while the set that
        // actually reaches VRAM is truncated by sync_budget and by coverage early-stop. The two are not
        // the same quantity, so this figure must not be read as "the cache is too small".
        LLAMA_LOG_WARN("MoE diag [cache %p]: %llu steps, latent miss frac %.0f%% (all unique misses, NOT "
                       "the loaded set: budget=%d cover=%.2f truncate it), capacity=%d | top-2 core: "
                       "%d distinct experts ever in top-2, hottest %d cover %.0f%% of all top-2 picks\n",
                       (void *) c, (unsigned long long) c->diag_steps,
                       100.0 * (double) c->diag_miss / (double) (c->diag_unique ? c->diag_unique : 1),
                       c->sync_budget, c->sync_cover,
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

    // Per-token freshness stride (LLAMA_MOE_SYNC_STRIDE; phase decided once per step in
    // llama_moe_stride_begin_step): on a free-fly step inspect nothing. The residency loop below then
    // resolves and loads nothing, so every position takes the second pass's pos_slot donor. Zeroing
    // order_n rather than sync_budget is deliberate: sync_budget also selects the ordering array above,
    // decides whether the clamp applies, and gates the threshold branch below, so writing it at runtime
    // is a mode switch, not a count change - and non-budget mode does not clamp order_n, so it would
    // load MORE. Layers below keep_il stay fresh (hash-routed front layers have no donor locality), and
    // a layer with fewer sentinels than routed positions refuses: there an unusable donor falls through
    // to next_spare, i.e. a live WRONG expert applied at the intended expert's full gate weight.
    if (!first_decode && order_n > 0 && llama_moe_stride_freefly() &&
        c->il >= g_stride_keep_il.load(std::memory_order_relaxed)) {
        if (c->n_sentinel >= (int) n_used) {
            order_n = 0;
            g_stride_applied.fetch_add(1, std::memory_order_relaxed);
        } else {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true, std::memory_order_relaxed)) {
                LLAMA_LOG_WARN("MoE stride: refusing to free-fly, n_sentinel=%d < n_used=%d - an unusable "
                               "donor would fall back to a live wrong expert at full gate weight. The "
                               "expert-group sweep raises n_sentinel to n_used; do not disable it.\n",
                               c->n_sentinel, (int) n_used);
            }
        }
    }

    // Coverage early-stop (LLAMA_MOE_SYNC_COVER, set via cov_cb into step_cover/step_wexp): sync missing
    // experts weight-descending but STOP once (already-cached + synced) gate-weight coverage reaches the
    // threshold. Still capped by order_n (=sync_budget, top-2), so the sync count is min(top-2, needed-for-
    // coverage): the cap guarantees the proven quality floor, coverage lets a step whose cached experts
    // already cover enough weight skip loads. cov_active gates it; when off, behaviour is exactly as before.
    // gate coverage off on first_decode (that step intentionally loads all 8 for a correct baseline) and
    // only sum coverage over the SAME experts the sync loop will inspect (order_n, = top-2 budget in steady
    // state). Summing the full first-decode order (8) wrongly reached the threshold and skipped real loads.
    const bool cov_active = (c->step_cover > 0.0f) && ((int) c->step_wexp.size() == c->n_expert)
                            && (n_tokens == 1) && !first_decode;
    double cov_have = 0.0; // weight already covered by cached experts among the inspected order
    if (cov_active) {
        for (int r = 0; r < order_n; ++r) {
            const int32_t e = order[(size_t) r];
            if (e >= 0 && e < c->n_expert && c->expert_slot[(size_t) e] >= 0) {
                cov_have += c->step_wexp[(size_t) e];
            }
        }
    }
    // COVDBG: capture the initial top-2 cached coverage vs the active threshold, before any sync this step.
    const bool covdbg = cov_active && getenv("LLAMA_MOE_COVDBG");
    int covdbg_earlystop = 0;
    if (covdbg) {
        g_covss_tokens.fetch_add(1, std::memory_order_relaxed);
        g_covss_have_x1000.fetch_add((uint64_t) (cov_have * 1000.0), std::memory_order_relaxed);
        g_covss_cover_x1000.fetch_add((uint64_t) (c->step_cover * 1000.0), std::memory_order_relaxed);
    }

    static const bool evictdbg = getenv("LLAMA_MOE_EVICTDBG") != nullptr; // forced-eviction census (read-only)

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
        // coverage early-stop: if the already-cached experts (plus misses synced so far this step) already
        // cover >= step_cover of the routed weight, skip syncing this (lower-weight) miss - leave it stale.
        // This is what lets a well-covered step load fewer than the budget. The budget cap (order_n) still
        // bounds the max, so we never sync MORE than the proven-safe top-2.
        if (cov_active && cov_have >= (double) c->step_cover) {
            if (covdbg) { covdbg_earlystop++; }
            continue;
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
            if (c->slot_loading[(size_t) s]) {
                continue; // loader is writing this slot with the mutex released
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
            if (!llama_moe_publish_batch()) {
                ggml_backend_tensor_set(c->slot_table, &c->capacity, (size_t) old_e * sizeof(int32_t), sizeof(int32_t));
                g_diag_pub_cnt += 1;
            }
            c->expert_slot[(size_t) old_e] = -1;
            moe_stale_on_evict(c, old_e, victim); // keep stale_table real for the pure-GPU decode path
        }
        c->slot_expert[(size_t) victim] = e;
        c->expert_slot[(size_t) e]      = victim;
        c->slot_age[(size_t) victim]    = ++c->tick;
        c->slot_pin[(size_t) victim]    = gen;
        load_e.push_back(e);
        load_slot.push_back(victim);
        // audit (LLAMA_MOE_EVICTDBG): this is a CRITICAL-PATH load. If the loader had force-evicted this
        // same expert earlier, that speculative eviction is what put us here - count it as a boomerang.
        if (evictdbg) {
            g_fev_sync.fetch_add(1, std::memory_order_relaxed);
            if ((int) c->fevict_step.size() == c->n_expert && c->fevict_step[(size_t) e] != 0) {
                g_fev_boom.fetch_add(1, std::memory_order_relaxed);
                if (c->diag_steps + 1 - c->fevict_step[(size_t) e] <= 8) {
                    g_fev_boom8.fetch_add(1, std::memory_order_relaxed);
                }
                c->fevict_step[(size_t) e] = 0; // count each eviction at most once
            }
        }
        if (cov_active) {
            cov_have += c->step_wexp[(size_t) e]; // this synced expert now contributes to coverage
            g_cov_synced.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (covdbg) {
        g_covss_earlystop.fetch_add((uint64_t) covdbg_earlystop, std::memory_order_relaxed);
        const uint64_t tk = g_covss_tokens.load(std::memory_order_relaxed);
        static std::atomic<uint64_t> last{0};
        uint64_t lp = last.load(std::memory_order_relaxed);
        if (tk - lp >= 2560 && last.compare_exchange_strong(lp, tk)) {
            const double nn = tk ? (double) tk : 1.0;
            LLAMA_LOG_WARN("MoE COVDBG: %llu tok-layers | cover thr mean=%.3f | initial cov_have (top-%d cached) "
                           "mean=%.3f | early-stopped misses mean=%.3f/tok\n",
                           (unsigned long long) tk,
                           (double) g_covss_cover_x1000.load() / 1000.0 / nn, order_n,
                           (double) g_covss_have_x1000.load() / 1000.0 / nn,
                           (double) g_covss_earlystop.load() / nn);
        }
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
    c->step_cover = 0.0f; // consumed for this step; cov_cb re-sets it next step if coverage mode is on
    g_diag_pl_ns    += (uint64_t) (ggml_time_us() - diag_pl0);
    g_diag_sync_cnt += (uint64_t) load_e.size();
    // publish the freshly loaded slots into the device slot_table only after their data is in
    // place (keeps slot_table consistent for the background loader's bookkeeping)
    const int64_t diag_pub0 = ggml_time_us();
    const bool pub_batch = llama_moe_publish_batch();
    for (size_t i = 0; i < load_e.size(); ++i) {
        if (!pub_batch) {
            ggml_backend_tensor_set(c->slot_table, &load_slot[i], (size_t) load_e[i] * sizeof(int32_t), sizeof(int32_t));
            moe_stale_on_load(c, load_e[i], load_slot[i]); // mirror into stale_table for the pure-GPU decode path
            g_diag_pub_cnt += 2;
        } else if (c->stale_table && load_e[i] >= 0 && load_e[i] < c->n_expert &&
                   (int) c->stale_of_expert.size() == c->n_expert) {
            c->stale_of_expert[(size_t) load_e[i]] = load_slot[i]; // host mirror only; uploaded once below
        }
        c->last_settled_slot = load_slot[i];
    }
    if (pub_batch && !load_e.empty() && c->stale_table &&
        (int) c->stale_of_expert.size() == c->n_expert) {
        // one whole-table upload instead of 2 per loaded expert. Must stay inside the mutex: the background
        // loader writes the same host mirror, and after parallel_load so no slot is published early.
        ggml_backend_tensor_set(c->stale_table, c->stale_of_expert.data(), 0,
                                (size_t) c->n_expert * sizeof(int32_t));
        g_diag_pub_cnt += 1;
    }
    if (getenv("LLAMA_MOE_DIAG")) { g_diag_publish_ns += (uint64_t) (ggml_time_us() - diag_pub0); }

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
    const bool slotdbg = getenv("LLAMA_MOE_SLOTDBG") != nullptr;
    const bool laydbg  = getenv("LLAMA_MOE_LAYERDBG") != nullptr;
    const int  sdb_i   = stale_reuse ? 1 : 0; // 0 = prefill, 1 = decode
    std::vector<char> used_slot((size_t) n_slots);
    for (int64_t t = 0; t < n_tokens; ++t) {
        std::fill(used_slot.begin(), used_slot.end(), 0);
        int sdb_hit = 0, sdb_sen = 0, sdb_spa = 0, sdb_sta = 0;
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
                sdb_hit++;
                if (stale_reuse && u < (int64_t) c->pos_slot.size()) { c->pos_slot[(size_t) u] = slot; }
            } else {
                // miss: prefer the stale slot this position used last step (decode), else a zero
                // sentinel (prefill); if that choice is already used this token, take any free slot.
                int32_t cand = -1;
                if (stale_reuse && u < (int64_t) c->pos_slot.size()) {
                    cand = c->pos_slot[(size_t) u];
                    if (cand < 0 || cand >= n_slots || used_slot[(size_t) cand]) { cand = -1; }
                    if (cand >= 0 && cand < c->capacity && c->slot_loading[(size_t) cand]) { cand = -1; }
                }
                if (cand >= 0) { sdb_sta++; }
                if (cand < 0) {
                    while (next_sentinel < n_slots && used_slot[(size_t) next_sentinel]) { next_sentinel++; }
                    if (next_sentinel < n_slots) {
                        cand = next_sentinel;
                        sdb_sen++;
                    } else {
                        while (next_spare < n_slots && used_slot[(size_t) next_spare]) { next_spare++; }
                        cand = next_spare; // exists: at most n_used-1 slots used so far
                        sdb_spa++;
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
        if (slotdbg) {
            g_slot_hit[sdb_i]      += (uint64_t) sdb_hit;
            g_slot_sentinel[sdb_i] += (uint64_t) sdb_sen;
            g_slot_spare[sdb_i]    += (uint64_t) sdb_spa;
            g_slot_stale[sdb_i]    += (uint64_t) sdb_sta;
            g_slot_tokens[sdb_i]++;
            g_slot_hist[sdb_i][(size_t) (sdb_hit < 33 ? sdb_hit : 32)]++;
        }
        if (laydbg && sdb_i == 1 && c->il >= 0 && c->il < MOE_LAYERDBG_MAX) {
            const size_t L = (size_t) c->il;
            g_lay_hit[L]      += (uint64_t) sdb_hit;
            g_lay_stale[L]    += (uint64_t) sdb_sta;
            g_lay_sentinel[L] += (uint64_t) sdb_sen;
            g_lay_spare[L]    += (uint64_t) sdb_spa;
            g_lay_tokens[L]++;
        }
    }
    if (slotdbg) {
        static std::atomic<uint64_t> last_print[2] = {{0}, {0}};
        const uint64_t tk = g_slot_tokens[sdb_i].load();
        uint64_t lp = last_print[sdb_i].load();
        if (tk - lp >= 20480 && last_print[sdb_i].compare_exchange_strong(lp, tk)) {
            const double nn = tk ? (double) tk : 1.0;
            const uint64_t tot = g_slot_hit[sdb_i] + g_slot_sentinel[sdb_i] + g_slot_spare[sdb_i] + g_slot_stale[sdb_i];
            const double td = tot ? (double) tot : 1.0;
            char hist[512]; int off = 0;
            for (int h = 0; h <= n_used && h < 33; ++h) {
                off += snprintf(hist + off, sizeof(hist) - (size_t) off, "%s%d:%.1f%%", h ? " " : "", h,
                                100.0 * (double) g_slot_hist[sdb_i][(size_t) h].load() / nn);
            }
            LLAMA_LOG_WARN("MoE SLOTDBG %s: %llu token-layers, cap=%d n_sentinel=%d | per-position: "
                           "HIT %.1f%% SENTINEL %.1f%% SPARE(wrong-expert) %.1f%% STALE %.1f%% | "
                           "per-token mean hit %.2f/%d sentinel %.2f spare %.2f | hits/token hist %s\n",
                           sdb_i ? "DECODE" : "PREFILL", (unsigned long long) tk, c->capacity, c->n_sentinel,
                           100.0 * (double) g_slot_hit[sdb_i].load() / td,
                           100.0 * (double) g_slot_sentinel[sdb_i].load() / td,
                           100.0 * (double) g_slot_spare[sdb_i].load() / td,
                           100.0 * (double) g_slot_stale[sdb_i].load() / td,
                           (double) g_slot_hit[sdb_i].load() / nn, (int) n_used,
                           (double) g_slot_sentinel[sdb_i].load() / nn,
                           (double) g_slot_spare[sdb_i].load() / nn, hist);
        }
    }

    // LAYERDBG report: per-layer decode residency, dumped by layer 0 so it prints once per interval. The
    // question it answers is whether some layers are structurally harder than the aggregate suggests - in
    // particular DeepSeek-V4's static token-id-routed layers, whose hit rate can only follow the output
    // token distribution. STALE is the interesting column: a stale position computes a WRONG expert.
    if (laydbg && sdb_i == 1 && c->il == 0) {
        static std::atomic<uint64_t> lay_steps{0};
        const uint64_t ls = lay_steps.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((ls % 128) == 0) {
            char line[4096]; int off = 0;
            off += snprintf(line + off, sizeof(line) - (size_t) off,
                            "MoE LAYERDBG after %llu decode tokens (cap=%d n_used=%d) prefetch req=%llu landed=%llu entry-resident=%llu il:HIT%%/STALE%%\n",
                            (unsigned long long) ls, c->capacity, (int) n_used,
                            (unsigned long long) g_moe_pf_req.load(),
                            (unsigned long long) g_moe_pf_landed.load(),
                            (unsigned long long) g_moe_pf_resident.load());
            for (int L = 0; L < MOE_LAYERDBG_MAX; ++L) {
                const uint64_t tkn = g_lay_tokens[(size_t) L].load();
                if (tkn == 0) { continue; }
                const uint64_t h = g_lay_hit[(size_t) L].load(), s = g_lay_stale[(size_t) L].load();
                const uint64_t sn = g_lay_sentinel[(size_t) L].load(), sp = g_lay_spare[(size_t) L].load();
                const double   tt = (double) (h + s + sn + sp ? h + s + sn + sp : 1);
                if (off < (int) sizeof(line) - 24) {
                    off += snprintf(line + off, sizeof(line) - (size_t) off, "%s%d:%.0f/%.0f",
                                    L ? " " : "", L, 100.0 * (double) h / tt, 100.0 * (double) s / tt);
                }
            }
            LLAMA_LOG_WARN("%s\n", line);
        }
    }

    // EVICTDBG report: the discriminating numbers for "is the loader's forced-eviction fallback eating the
    // high-score core?". forced/admit = how often the score guard is bypassed; boomerang = how many of those
    // victims the critical path then had to sync-load back. Near-zero forced => policy is fine and the cache
    // is simply capacity-bound; high forced WITH boomerangs => the loader is evicting experts it needs back.
    if (evictdbg && n_tokens == 1) {
        static std::atomic<uint64_t> ev_steps{0};
        const uint64_t es = ev_steps.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((es % 20480) == 0) {
            const uint64_t adm = g_fev_admit.load(), fcd = g_fev_forced.load();
            const uint64_t syn = g_fev_sync.load(), boo = g_fev_boom.load();
            const double   ad  = adm ? (double) adm : 1.0;
            const double   tl  = (double) es; // token-layers observed by this counter
            LLAMA_LOG_WARN("MoE EVICTDBG: %llu decode token-layers | loader admits %llu (%.3f/tok-layer) "
                           "FORCED %llu (%.1f%% of admits, %.3f/tok-layer) score-skips %llu no-fit %llu | "
                           "mean forced score gap %.2f | critical-path sync-loads %llu, of which BOOMERANG "
                           "%llu (%.1f%%) within-8-steps %llu\n",
                           (unsigned long long) es, (unsigned long long) adm, (double) adm / tl,
                           (unsigned long long) fcd, 100.0 * (double) fcd / ad, (double) fcd / tl,
                           (unsigned long long) g_fev_skips.load(), (unsigned long long) g_fev_nofit.load(),
                           fcd ? (double) g_fev_gap_x1000.load() / 1000.0 / (double) fcd : 0.0,
                           (unsigned long long) syn, (unsigned long long) boo,
                           syn ? 100.0 * (double) boo / (double) syn : 0.0,
                           (unsigned long long) g_fev_boom8.load());
        }
    }

    // timing breakdown: accumulate total callback time; print + reset every g_diag_steps DECODE STEPS
    // when LLAMA_MOE_DIAG is set. The period is derived from the real MoE layer count so the printed
    // totals always cover exactly that many steps on any architecture.
    // Units: callback, parallel_load, publish, LOCK-WAIT and the derived remap are WALL time on this
    // thread. read and H2D-staged are THREAD-SUMS over the io threads, so they are not on the same scale
    // and no residual can be derived from them; H2D-batch is wall on the issuing thread. The LOADER line
    // is the background thread's own disk work over the same window: it is NOT part of parallel_load, so
    // it must be read as competition for the device and the disk, never as a component of the callback.
    g_diag_cb_ns += (uint64_t) (ggml_time_us() - diag_t0);

    // Always-on spill watch (NOT gated on LLAMA_MOE_DIAG). The startup audit only sees the state right
    // after allocation; the driver can start spilling later, once the graph's transients and the KV cache
    // have grown into whatever headroom was left. So resample from the eval path and WARN the first time
    // non-local usage crosses the threshold.
    //
    // Deliberately NOT gated on n_tokens == 1, unlike the timing print below: this has nothing to do with
    // the single-token fast path, and with speculative decoding a verify step carries n_tokens > 1 - so
    // that gate would only ever see the draft step's ONE layer, which is far too few callbacks to reach
    // any sample period. Prefill samples too, which is wanted: the sweep is when VRAM pressure peaks.
    // Always sample the FIRST call so the reading exists even on a short generation, then every 8 steps'
    // worth of callbacks (a DXGI query is a COM call, not a register read).
    if (c->dev_backend) {
        static uint64_t spill_calls = 0;
        const uint64_t  spill_period = (uint64_t) llama_moe_diag_n_layers(c) * 8;
        const uint64_t  spill_n      = ++spill_calls;
        if (spill_n == 1 || (spill_period > 0 && spill_n % spill_period == 0)) {
            size_t sdev_free = 0, sdev_total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(c->dev_backend), &sdev_free, &sdev_total);
            llama_moe_vram_seg seg;
            if (sdev_total > 0 && llama_moe_vram_spill_sample((uint64_t) sdev_total, &seg)) {
                const double   mib  = 1024.0*1024.0;
                const uint64_t over = seg.local_used > seg.local_budget
                                    ? seg.local_used - seg.local_budget : 0;
                // Report the first sample, then again whenever the over-budget figure grows by another
                // 512 MiB. Keyed on OVER-BUDGET, not on the non-local figure: measured on a 4090D, a
                // healthy fast run already sits at 162-310 MiB non-local (that is the pinned H2D staging
                // arena, which is host memory the GPU can see) while local ran 4.3 GiB over budget. So the
                // over-budget delta is what tracks the driver's paging pressure; non-local only says how
                // much of it happens to be parked in host RAM at this instant.
                // The startup audit cannot see this at all: it runs before the KV cache and the graph's
                // transients have grown into whatever headroom was left. The first sample normally lands in
                // prefill (the sweep is when pressure peaks), hence "in eval", not "in decode".
                static std::atomic<uint64_t> reported{UINT64_MAX}; // UINT64_MAX = nothing reported yet
                const uint64_t prev = reported.load(std::memory_order_relaxed);
                const uint64_t step = 512ull * 1024 * 1024;
                if (prev == UINT64_MAX || over >= prev + step) {
                    reported.store(over, std::memory_order_relaxed);
                    LLAMA_LOG_WARN("MoE stream: VRAM SEGMENTS in eval - local %.0f / %.0f MiB budget "
                                   "(over by %.0f MiB), spilled to host (non-local) %.0f MiB.\n",
                                   seg.local_used/mib, seg.local_budget/mib, over/mib,
                                   seg.nonlocal_used/mib);
                }
                // 256 MiB over budget: comfortably above driver bookkeeping, comfortably below the 4+ GiB
                // seen when the cache genuinely does not fit. The threshold is a judgement call - the
                // printed number is the thing to act on, not the fact that the line appeared. Note that
                // being over budget is NOT by itself a throughput collapse: measured 8.4 tok/s decode at
                // 4.3 GiB over. It means the driver is paging, not that paging is dominating.
                static std::atomic<bool> warned{false};
                if (over > 256ull * 1024 * 1024 && !warned.exchange(true)) {
                    LLAMA_LOG_WARN("MoE stream: VRAM SPILL - this process has committed %.0f MiB more "
                                   "video memory than the driver's budget (%.0f MiB), so the driver is "
                                   "paging the excess over PCIe; %.0f MiB is in host RAM right now. Lower "
                                   "LLAMA_MOE_CACHE_CAP or raise LLAMA_MOE_VRAM_RESERVE_MB.\n",
                                   over/mib, seg.local_budget/mib, seg.nonlocal_used/mib);
                }
            }
        }
    }
    if (getenv("LLAMA_MOE_DIAG") && n_tokens == 1) {
        const uint64_t period = (uint64_t) llama_moe_diag_n_layers(c) * (uint64_t) g_diag_steps;
        static uint64_t calls = 0;
        if (period > 0 && ++calls % period == 0) {
            const double cb_ms  = g_diag_cb_ns / 1000.0;
            const double pl_ms  = g_diag_pl_ns / 1000.0;
            const double lw_ms  = g_diag_lockwait_ns / 1000.0;
            const double pub_ms = g_diag_publish_ns / 1000.0;
            const double rd_ms  = g_diag_read_ns.load() / 1000.0;
            const double h2d_ms = g_diag_h2d_ns.load() / 1000.0;
            const double h2dw_ms = g_diag_h2d_wall_ns.load() / 1000.0;
            const double lrd_ms = g_diag_ldr_read_ns.load() / 1000.0;
            const double lh2_ms = g_diag_ldr_h2d_ns.load() / 1000.0;
            const double rp_ms  = g_diag_rdphase_ns.load() / 1000.0;
            const double rb_mib = g_diag_read_bytes.load() / (1024.0 * 1024.0);
            const double rbw    = rp_ms > 0.0 ? rb_mib / rp_ms / 1024.0 * 1000.0 : 0.0; // GiB/s
            const uint64_t sc = g_diag_sync_cnt ? g_diag_sync_cnt : 1;
            LLAMA_LOG_WARN("MoE timing/%d steps [wall]: callback %.0fms = parallel_load %.0f + publish %.0f "
                           "+ remap %.0f; parallel_load %.0f = setup %.0f + read-phase %.0f + H2D-batch %.0f; "
                           "LOCK-WAIT %.0fms; publish %llu device-syncs; read-phase moved %.0f MiB = %.2f GiB/s "
                           "| [thread-sums, not subtractable] read %.0fms, H2D-staged %.0fms | sync-loaded %llu "
                           "(RAM-hit %.0f%%, DISK %llu) | LOADER (separate thread, not in callback): "
                           "read %.0fms, H2D %.0fms, %llu expert-proj | VRAM spill peak %.0f MiB\n",
                           g_diag_steps, cb_ms, pl_ms, pub_ms, cb_ms - pl_ms - pub_ms,
                           pl_ms, pl_ms - rp_ms - h2dw_ms, rp_ms, h2dw_ms,
                           lw_ms, (unsigned long long) g_diag_pub_cnt.load(), rb_mib, rbw,
                           rd_ms, h2d_ms,
                           (unsigned long long) g_diag_sync_cnt,
                           100.0 * (double) g_diag_ram_hit / (double) sc,
                           (unsigned long long) (g_diag_sync_cnt - g_diag_ram_hit),
                           lrd_ms, lh2_ms, (unsigned long long) g_diag_ldr_cnt.load(),
                           g_moe_vram_spill_peak.load(std::memory_order_relaxed)/(1024.0*1024.0));
            g_diag_cb_ns = g_diag_pl_ns = g_diag_sync_cnt = g_diag_ram_hit = g_diag_lockwait_ns = g_diag_publish_ns = 0;
            g_diag_read_ns = 0; g_diag_h2d_ns = 0; g_diag_h2d_wall_ns = 0;
            g_diag_ldr_read_ns = 0; g_diag_ldr_h2d_ns = 0; g_diag_ldr_cnt = 0;
            g_diag_rdphase_ns = 0; g_diag_read_bytes = 0;
            g_diag_ldr_rdphase_ns = 0; g_diag_ldr_bytes = 0; g_diag_pub_cnt = 0;
        }
    }
}

// fork: host callback for the fused CUDA GGML_OP_MOE_FFN op. The CUDA op has already copied the routing
// ids to sel_host; this plans residency + sync-loads the missing top-N experts into the device cache and
// writes the resolved slot per position into ids_host. It reuses llama_moe_layer_remap_cb verbatim (the
// proven residency/eviction/load logic) by wrapping the host arrays in contiguous stand-in tensors - so
// there is a single source of truth for "which slot does each routing position map to". remap_cb takes
// g_moe_layer_mutex itself, so we must not hold it here. Runs on the host inside the CUDA op, before the
// expert matmuls, so the loaded slabs are in VRAM by the time mul_mat_id reads them (same stream).
static void llama_moe_ffn_sync(void * cache, const int32_t * sel_host, int32_t * ids_host,
                               int n_used, int n_tokens) {
    llama_moe_layer_cache * c = (llama_moe_layer_cache *) cache;
    if (!c) { return; }

    ggml_tensor a;   // stand-in for `selected_experts` (remap_cb reads a->data via nb strides)
    memset(&a, 0, sizeof(a));
    a.type = GGML_TYPE_I32;
    a.ne[0] = n_used; a.ne[1] = n_tokens; a.ne[2] = 1; a.ne[3] = 1;
    a.nb[0] = sizeof(int32_t);
    a.nb[1] = a.ne[0]*a.nb[0];
    a.nb[2] = a.ne[1]*a.nb[1];
    a.nb[3] = a.ne[2]*a.nb[2];
    a.data  = (void *) sel_host;

    ggml_tensor d = a; // stand-in for the map_custom dst (remap_cb writes slot ids into d->data)
    d.data = ids_host;

    llama_moe_layer_remap_cb(&d, &a, /*ith=*/0, /*nth=*/1, c);
}

ggml_moe_ffn_sync_fn llama_moe_layer_cache_sync_fn(void) {
    return llama_moe_ffn_sync;
}

// fork: CPU expert lane (Pulsar-style). For each routed position whose expert is resident in the host
// RAM pool, compute the FULL FFN on the CPU with ggml's AVX2 quantized vec_dot - no H2D upload, no VRAM
// slot - and accumulate its weighted contribution into partial_host. Zeroes the router weight of every
// handled position so the GPU weighted-sum excludes it (the caller uploads the modified weights and adds
// partial_host). Byte-identical math to the GPU path (same quantized weights, q8_K-quantized activations,
// silu(gate)*up). Threaded across (position) jobs. Returns the number of positions handled.
//
// Layout (verified): gate/up expert slab = n_ff rows of n_embd quantized elems; down slab = n_embd rows
// of n_ff quantized elems. One row = ggml_row_size(type, k). vec_dot(k, &s, 0, wrow, 0, actq, 0, 1).
static int llama_moe_cpu_lane(void * cache, const float * cur_host, float * weights_host,
                              const int32_t * sel_host, int32_t * /*ids_host*/, float * partial_host,
                              int n_used, int n_tokens, int n_embd, int n_ff, int /*gpu_skip_slot*/) {
    llama_moe_layer_cache * c = (llama_moe_layer_cache *) cache;
    if (!c || c->proj.size() < 3) { return 0; }

    // Projection order is fixed by build_moe_ffn's registration (llama-graph.cpp: projs[] =
    // {gate_exps, up_exps, down_exps} when gate_up_exps is null, which the fused-op guard requires), so
    // proj[0]=gate, proj[1]=up, proj[2]=down. gate/up share n_embd input dim; down takes the n_ff mid.
    const llama_moe_proj_store & gate_pr = c->proj[0];
    const llama_moe_proj_store & up_pr   = c->proj[1];
    const llama_moe_proj_store & down_pr = c->proj[2];

    const ggml_type wtype = gate_pr.src ? gate_pr.src->type : GGML_TYPE_COUNT;
    if (wtype == GGML_TYPE_COUNT) { return 0; }
    const ggml_type dtype = down_pr.src ? down_pr.src->type : GGML_TYPE_COUNT;
    const auto * tt_gu = ggml_get_type_traits_cpu(wtype);   // gate/up weight traits
    const auto * tt_dn = ggml_get_type_traits_cpu(dtype);   // down weight traits
    if (!tt_gu || !tt_gu->vec_dot || !tt_gu->from_float) { return 0; }
    if (!tt_dn || !tt_dn->vec_dot || !tt_dn->from_float) { return 0; }
    const ggml_type gu_vdt = tt_gu->vec_dot_type; // activation quant type for gate/up dots
    const ggml_type dn_vdt = tt_dn->vec_dot_type; // activation quant type for down dots

    const size_t gu_rowbytes = ggml_row_size(wtype, n_embd); // one gate/up weight row (n_embd elems)
    const size_t dn_rowbytes = ggml_row_size(dtype, n_ff);   // one down weight row (n_ff elems)

    // Build the CPU job list under the lock: each job is a (token, used-position) whose expert is
    // RAM-resident in ALL three projections. Zero its router weight so the GPU sum drops it. We snapshot
    // the RAM pointers here; the slots are pinned by the residency machinery until this step completes.
    struct Job { int t; int u; int e; float w; const char * gate; const char * up; const char * down; };
    std::vector<Job> jobs;
    {
        std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
        if (gate_pr.ram == nullptr) { return 0; } // RAM tier off => nothing to do on the CPU
        for (int t = 0; t < n_tokens; ++t) {
            for (int u = 0; u < n_used; ++u) {
                const int32_t e = sel_host[(size_t) (t * n_used + u)];
                if (e < 0 || e >= c->n_expert) { continue; }
                const int32_t rs = c->ram_slot[(size_t) e];
                if (rs < 0) { continue; } // not RAM-resident => leave it to the GPU
                Job j;
                j.t = t; j.u = u; j.e = e;
                j.w = weights_host[(size_t) (t * n_used + u)];
                j.gate = (const char *) gate_pr.ram + (size_t) rs * gate_pr.stride;
                j.up   = (const char *) up_pr.ram   + (size_t) rs * up_pr.stride;
                j.down = (const char *) down_pr.ram + (size_t) rs * down_pr.stride;
                jobs.push_back(j);
                weights_host[(size_t) (t * n_used + u)] = 0.0f; // GPU sum excludes this position
            }
        }
    }
    if (jobs.empty()) { return 0; }

    // per-token output accumulator is shared, so guard the final add; the heavy dots are lock-free.
    std::mutex out_mtx;
    const size_t act_gu_bytes = ggml_row_size(gu_vdt, n_embd);
    const size_t act_dn_bytes = ggml_row_size(dn_vdt, n_ff);

    int nthreads = 1;
    if (const char * e = getenv("LLAMA_MOE_IO_THREADS")) { nthreads = atoi(e); }
    if (nthreads < 1) { nthreads = 1; }
    if (nthreads > (int) jobs.size()) { nthreads = (int) jobs.size(); }

    auto worker = [&](int tid) {
        // per-thread scratch: quantized input row (shared across a token's experts), gate/up/mid buffers
        std::vector<char>  xq(act_gu_bytes);   // q8_K of the token's n_embd input
        std::vector<char>  midq(act_dn_bytes); // q8_K of the n_ff intermediate
        std::vector<float> mid((size_t) n_ff);
        std::vector<float> out((size_t) n_embd);
        int last_t = -1;
        for (size_t ji = (size_t) tid; ji < jobs.size(); ji += (size_t) nthreads) {
            const Job & j = jobs[ji];
            // quantize this token's input activations to the gate/up dot type (reuse across experts of
            // the same token handled by this thread)
            if (j.t != last_t) {
                tt_gu->from_float(cur_host + (size_t) j.t * n_embd, xq.data(), n_embd);
                last_t = j.t;
            }
            // gate/up: n_ff rows each -> silu(gate)*up
            for (int f = 0; f < n_ff; ++f) {
                float g = 0.0f, u = 0.0f;
                tt_gu->vec_dot(n_embd, &g, 0, j.gate + (size_t) f * gu_rowbytes, 0, xq.data(), 0, 1);
                tt_gu->vec_dot(n_embd, &u, 0, j.up   + (size_t) f * gu_rowbytes, 0, xq.data(), 0, 1);
                const float s = g / (1.0f + expf(-g)); // silu(gate)
                mid[(size_t) f] = s * u;
            }
            // quantize the intermediate to the down dot type
            tt_dn->from_float(mid.data(), midq.data(), n_ff);
            // down: n_embd rows -> out
            for (int i = 0; i < n_embd; ++i) {
                float o = 0.0f;
                tt_dn->vec_dot(n_ff, &o, 0, j.down + (size_t) i * dn_rowbytes, 0, midq.data(), 0, 1);
                out[(size_t) i] = o * j.w;
            }
            // accumulate into the shared per-token partial
            {
                std::lock_guard<std::mutex> lk(out_mtx);
                float * dst = partial_host + (size_t) j.t * n_embd;
                for (int i = 0; i < n_embd; ++i) { dst[i] += out[(size_t) i]; }
            }
        }
    };

    if (nthreads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> pool;
        for (int t = 1; t < nthreads; ++t) { pool.emplace_back(worker, t); }
        worker(0);
        for (auto & th : pool) { th.join(); }
    }
    return (int) jobs.size();
}

ggml_moe_ffn_cpu_fn llama_moe_layer_cache_cpu_fn(void) {
    return llama_moe_cpu_lane;
}

// prefill sweep census (LLAMA_MOE_SWEEPDBG). The sweep's losslessness reduces to one crisp,
// falsifiable invariant: over the n_groups passes of a single step, every routing position must
// resolve to its REAL expert slot exactly once and to a zero sentinel in every other pass. So
// hit / (hit + sentinel) must equal 1/n_groups, and `loads` per step must equal the step's unique
// expert count (not the capacity). Anything else means experts are being dropped or double-counted.
static std::atomic<uint64_t> g_sweep_hit{0};
static std::atomic<uint64_t> g_sweep_sentinel{0};
static std::atomic<uint64_t> g_sweep_loads{0};
static std::atomic<uint64_t> g_sweep_passes{0};

// Set by every sweep pass, consumed once by llama_moe_refill_vram_caches at the first decode token.
// A sweep walks the whole expert index space and leaves each cache holding whatever the LAST group
// was, which is an arbitrary index range with no relation to what decode will ask for - measured
// decode after a sweep is several times slower than after an ordinary prefill, which warms the cache
// on the experts it actually used. See llama_moe_refill_vram_caches.
static std::atomic<bool> g_moe_sweep_dirty{false};

// fork: one pass of the prefill expert-group sweep (see llama_moe_layer_cache_remap_group in
// llama-moe-stream.h for the design). Resolves residency for the experts of ONE static index group
// and writes this group's slot ids. Deliberately much simpler than llama_moe_layer_remap_cb: no
// threshold, no budget, no coverage early-stop, no stale reuse. The group is sized so its whole
// selected set fits in `capacity`, so every in-group expert is loaded and every out-of-group position
// gets a genuine zero sentinel. Nothing is dropped.
static void llama_moe_layer_remap_group_cb(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * dep,
                                           int ith, int nth, void * userdata) {
    (void) nth;
    (void) dep; // scheduling edge only - never read for its value; see the header note
    if (ith != 0) {
        return;
    }
    const llama_moe_group_ud * ud = (const llama_moe_group_ud *) userdata;
    llama_moe_layer_cache *    c  = ud->c;

    const int64_t n_used   = a->ne[0];
    const int64_t n_tokens = a->ne[1];
    const int64_t n        = n_used * n_tokens;

    llama_moe_prefill_once();

    std::vector<int32_t> flat;
    llama_moe_flatten_ids(a, flat);

    const int64_t diag_lock0 = ggml_time_us();
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    // sweep-path wait goes to the sweep counter. It used to land in g_diag_lockwait_ns, which only the
    // decode print reads and resets, so a whole prefill worth of contention showed up as decode LOCK-WAIT
    // in the first print after the prompt.
    if (getenv("LLAMA_MOE_DIAG")) { g_diag_sweep_lockwait_ns += (uint64_t) (ggml_time_us() - diag_lock0); }

    // New pin generation, which releases the PREVIOUS group's pins. This is only safe because `dep`
    // gave the scheduler a data edge from that group's FFN output into this callback, so its matmuls
    // have retired and their slabs may now be overwritten.
    const uint64_t gen = ++c->pin_gen;

    // Expert scoring is a per-STEP notion (it seeds the loader's eviction protection for decode), so
    // update it once per sweep - on the first group - over the whole selection.
    //
    // The DECAY GRANULARITY matters and is easy to get wrong. The single-token path decays once per
    // token, so its score is dominated by the most recent few dozen tokens; that recency is exactly
    // what makes the resident set predictive for the NEXT token. Decaying once per step instead yields
    // a flat histogram over the entire ubatch with no recency weighting at all, and a cache warmed from
    // that ranking leaves decode running cold (measured ~6x slower than after a single-token prefill).
    // So reproduce the per-token decay in closed form: age the carried-over score by d^n_tokens, then
    // add each token's contribution weighted by how recent it is. Tokens more than a few hundred back
    // underflow to zero on their own, which is the intended behaviour.
    if (c->weighted_evict && ud->e_lo == 0) {
        if ((int) c->expert_score.size() != c->n_expert) { c->expert_score.assign((size_t) c->n_expert, 0.0f); }
        const float d = 0.97f;
        float dn = 1.0f;
        for (int64_t t = 0; t < n_tokens; ++t) { dn *= d; }
        for (float & s : c->expert_score) { s *= dn; }
        float w = 1.0f; // weight of the newest token
        for (int64_t t = n_tokens - 1; t >= 0; --t) {
            for (int64_t u = 0; u < n_used; ++u) {
                const int32_t e = flat[(size_t) (t * n_used + u)];
                if (e >= 0 && e < c->n_expert) {
                    c->expert_score[(size_t) e] += w / (float) (u + 1);
                }
            }
            w *= d;
        }
    }

    // Mirror the single-pass remap's diagnostic counters (once per sweep, on the first group) so a sweep
    // run's MoE diag lines are comparable to a non-sweep run's. Without this the sweep's diag_steps and
    // top-2 histogram cover DECODE ONLY while the baseline's cover prompt+decode, which silently makes
    // any "working set concentration" comparison between the two meaningless.
    if (ud->e_lo == 0) {
        c->diag_steps += (uint64_t) n_tokens;
        if (getenv("LLAMA_MOE_DIAG")) {
            if ((int) c->diag_top2.size() != c->n_expert) { c->diag_top2.assign((size_t) c->n_expert, 0); }
            for (int64_t t = 0; t < n_tokens; ++t) {
                for (int64_t u = 0; u < 2 && u < n_used; ++u) {
                    const int32_t e = flat[(size_t) (t * n_used + u)];
                    if (e >= 0 && e < c->n_expert) { c->diag_top2[(size_t) e]++; }
                }
            }
        }
    }

    // Record the resident set an ordinary capacity-capped prefill would have produced for this ubatch,
    // so the post-sweep refill can restore exactly that (see llama_moe_layer_cache::freq_ranked). The
    // ranking must match llama_moe_plan_remap's comparator verbatim: descending selection frequency,
    // ties broken by lower expert index.
    if (ud->e_lo == 0) {
        std::vector<int> freq((size_t) c->n_expert, 0);
        for (int64_t p = 0; p < n; ++p) {
            const int32_t e = flat[(size_t) p];
            if (e >= 0 && e < c->n_expert) { freq[(size_t) e]++; }
        }
        std::vector<int32_t> ranked;
        ranked.reserve((size_t) c->n_expert);
        for (int e = 0; e < c->n_expert; ++e) {
            if (freq[(size_t) e] > 0) { ranked.push_back(e); }
        }
        std::sort(ranked.begin(), ranked.end(), [&](int32_t x, int32_t y) {
            return freq[(size_t) x] != freq[(size_t) y] ? freq[(size_t) x] > freq[(size_t) y] : x < y;
        });
        if ((int) ranked.size() > c->capacity) { ranked.resize((size_t) c->capacity); }
        c->freq_ranked = std::move(ranked);
    }

    // Which of this group's experts does this ubatch actually select? Skipping the unselected ones is
    // what keeps a short window (or a hash-routed layer, where only a few experts are reachable) from
    // loading more than the single-pass path would.
    std::vector<int> need;
    {
        const int span = ud->e_hi - ud->e_lo;
        need.reserve((size_t) span);
        std::vector<char> seen((size_t) span, 0);
        for (int64_t p = 0; p < n; ++p) {
            const int32_t e = flat[(size_t) p];
            if (e >= ud->e_lo && e < ud->e_hi && !seen[(size_t) (e - ud->e_lo)]) {
                seen[(size_t) (e - ud->e_lo)] = 1;
                need.push_back(e);
            }
        }
    }

    // Resolve residency for the whole group. Age-LRU victim choice (not the weighted policy the
    // single-pass path uses): across a sweep every slot must be recyclable, and protecting high-score
    // slots here would just starve the later groups.
    std::vector<int> load_e;
    std::vector<int> load_slot;
    for (const int e : need) {
        int slot = c->expert_slot[(size_t) e];
        if (slot >= 0) {
            c->slot_pin[(size_t) slot] = gen;
            c->slot_age[(size_t) slot] = ++c->tick;
            continue;
        }
        int      victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int s = 0; s < c->capacity; ++s) {
            if (c->slot_pin[(size_t) s] == gen) {
                continue; // pinned by an earlier expert of THIS group
            }
            if (c->slot_loading[(size_t) s]) {
                continue; // loader is writing this slot with the mutex released
            }
            if (c->slot_expert[(size_t) s] < 0) { victim = s; break; }
            if (c->slot_age[(size_t) s] < oldest) { oldest = c->slot_age[(size_t) s]; victim = s; }
        }
        // The pin-only argument (|need| <= group span <= capacity, only this group holds pins) guarantees
        // an UNPINNED slot remains - but the loop above also skips slots the background async loader is
        // mid-write on (slot_loading), and the loader is deliberately NOT suspended during a sweep. On a
        // full group (|need| == capacity) the single unpinned slot can be exactly the one the loader is
        // writing, leaving no victim. Mirror the single-pass remap (llama_moe_layer_remap_cb): stop here
        // and let the remaining in-group experts fall through to a zero sentinel for this ubatch (the dst
        // writer below already maps expert_slot < 0 to a sentinel). Spin-waiting would deadlock - the
        // loader's publish needs THIS mutex - so drop rather than wait: a little quality for one ubatch
        // step, never a torn read. (Was GGML_ASSERT(victim >= 0), which crashed on this benign race.)
        if (victim < 0) {
            break;
        }
        const int old_e = c->slot_expert[(size_t) victim];
        if (old_e >= 0 && old_e < c->n_expert) {
            ggml_backend_tensor_set(c->slot_table, &c->capacity, (size_t) old_e * sizeof(int32_t), sizeof(int32_t));
            c->expert_slot[(size_t) old_e] = -1;
            moe_stale_on_evict(c, old_e, victim);
        }
        c->slot_expert[(size_t) victim] = e;
        c->expert_slot[(size_t) e]      = victim;
        c->slot_age[(size_t) victim]    = ++c->tick;
        c->slot_pin[(size_t) victim]    = gen;
        load_e.push_back(e);
        load_slot.push_back(victim);
    }

    const int64_t diag_pl0 = ggml_time_us();
    llama_moe_layer_parallel_load(c, load_e, load_slot);
    g_diag_pl_ns    += (uint64_t) (ggml_time_us() - diag_pl0);
    g_diag_sync_cnt += (uint64_t) load_e.size();
    // RAM-tier hit accounting for the sweep path. g_diag_ram_hit is otherwise only touched by the
    // single-pass callback, so without this the sweep's prefill breakdown reports a meaningless 0% and
    // blames every load on disk. Counted BEFORE parallel_load runs, since it consumes the same residency.
    if (getenv("LLAMA_MOE_DIAG") && (int) c->ram_slot.size() == c->n_expert) {
        for (const int e : load_e) {
            if (e >= 0 && e < c->n_expert && c->ram_slot[(size_t) e] >= 0) { g_diag_ram_hit++; }
        }
    }
    for (size_t i = 0; i < load_e.size(); ++i) {
        ggml_backend_tensor_set(c->slot_table, &load_slot[i], (size_t) load_e[i] * sizeof(int32_t), sizeof(int32_t));
        moe_stale_on_load(c, load_e[i], load_slot[i]);
        c->last_settled_slot = load_slot[i];
    }

    // Write this group's slot per routing position: an in-group expert resolves to its real (now
    // guaranteed resident) slot, everything else to a distinct zero sentinel. Distinctness within a
    // token is the MMQ requirement; n_sentinel == n_used makes it always satisfiable without ever
    // falling back to a spare RESIDENT slot (which would inject a wrong expert and break losslessness).
    const int n_slots = c->capacity + c->n_sentinel;
    std::vector<char> used_slot((size_t) n_slots);
    static const bool sweepdbg = getenv("LLAMA_MOE_SWEEPDBG") != nullptr;
    uint64_t dbg_hit = 0, dbg_sen = 0;
    for (int64_t t = 0; t < n_tokens; ++t) {
        std::fill(used_slot.begin(), used_slot.end(), 0);
        for (int64_t u = 0; u < n_used; ++u) {
            const int32_t e = flat[(size_t) (t * n_used + u)];
            if (e < ud->e_lo || e >= ud->e_hi) {
                continue;
            }
            const int32_t slot = c->expert_slot[(size_t) e];
            if (slot >= 0) { used_slot[(size_t) slot] = 1; }
        }
        int next_sentinel = c->capacity;
        for (int64_t u = 0; u < n_used; ++u) {
            const int32_t e = flat[(size_t) (t * n_used + u)];
            int32_t slot = (e >= ud->e_lo && e < ud->e_hi) ? c->expert_slot[(size_t) e] : -1;
            if (slot < 0) {
                while (next_sentinel < n_slots && used_slot[(size_t) next_sentinel]) { next_sentinel++; }
                GGML_ASSERT(next_sentinel < n_slots); // guaranteed by n_sentinel == n_used
                slot = next_sentinel;
                dbg_sen++;
            } else {
                dbg_hit++;
            }
            used_slot[(size_t) slot] = 1;
            if (slot < c->capacity) { c->slot_pin[(size_t) slot] = gen; }
            *(int32_t *) ((char *) dst->data + t * dst->nb[1] + u * dst->nb[0]) = slot;
        }
    }

    // This layer's cache now holds this group's index range. Flag it (and the global fast-path gate) as
    // needing a refill before decode (consumed once, at the first single-token step). Per-cache and not
    // just global, so a second model's sweep does not force a full refill of this model's caches.
    c->sweep_dirty = true;
    g_moe_sweep_dirty.store(true, std::memory_order_relaxed);

    // Prefill breakdown (LLAMA_MOE_DIAG). The single-pass callback's timing print is gated to n_tokens==1,
    // and the sweep does not go through it at all, so prefill had no breakdown - the one number needed to
    // decide whether the sweep's cost is BYTES (every selected expert must reach VRAM once per ubatch step)
    // or COMPUTE (n_groups full-width FFN passes, of which (n_groups-1)/n_groups multiplies zero sentinel
    // slabs). Printed once per ubatch step: n_moe_layers x n_groups group-calls.
    if (getenv("LLAMA_MOE_DIAG")) {
        static std::atomic<uint64_t> gcalls{0};
        const uint64_t gc = gcalls.fetch_add(1, std::memory_order_relaxed) + 1;
        const int      n_groups = (c->n_expert + c->capacity - 1) / c->capacity;
        const int      n_lay    = llama_moe_diag_n_layers(c);
        const uint64_t per_step = (uint64_t) n_lay * (uint64_t) (n_groups > 0 ? n_groups : 1);
        if (per_step > 0 && (gc % per_step) == 0) {
            const double pl_ms  = g_diag_pl_ns / 1000.0;
            const double rd_ms  = g_diag_read_ns.load() / 1000.0;
            const double h2d_ms = g_diag_h2d_ns.load() / 1000.0;
            const double h2dw_ms = g_diag_h2d_wall_ns.load() / 1000.0;
            const double lw_ms  = g_diag_sweep_lockwait_ns / 1000.0;
            const double rp_ms  = g_diag_rdphase_ns.load() / 1000.0;
            const double rb_mib = g_diag_read_bytes.load() / (1024.0 * 1024.0);
            const double rbw    = rp_ms > 0.0 ? rb_mib / rp_ms / 1024.0 * 1000.0 : 0.0; // GiB/s
            const uint64_t sc   = g_diag_sync_cnt ? g_diag_sync_cnt : 1;
            LLAMA_LOG_WARN("MoE SWEEP timing/ubatch-step (%llu group-calls, %d layers x n_groups=%d): "
                           "[wall] parallel_load %.0fms = setup %.0f + read-phase %.0f + H2D-batch %.0f; "
                           "LOCK-WAIT %.0fms; read-phase moved %.0f MiB = %.2f GiB/s | "
                           "[thread-sums, not subtractable] read %.0fms, H2D-staged %.0fms | "
                           "experts loaded %llu (RAM-hit %.0f%%, DISK %llu)\n",
                           (unsigned long long) per_step, n_lay, n_groups,
                           pl_ms, pl_ms - rp_ms - h2dw_ms, rp_ms, h2dw_ms,
                           lw_ms, rb_mib, rbw,
                           rd_ms, h2d_ms, (unsigned long long) g_diag_sync_cnt,
                           100.0 * (double) g_diag_ram_hit / (double) sc,
                           (unsigned long long) (g_diag_sync_cnt - g_diag_ram_hit));
            g_diag_pl_ns = g_diag_sync_cnt = g_diag_ram_hit = g_diag_sweep_lockwait_ns = 0;
            g_diag_read_ns = 0; g_diag_h2d_ns = 0; g_diag_h2d_wall_ns = 0;
            g_diag_rdphase_ns = 0; g_diag_read_bytes = 0;
        }
    }

    if (sweepdbg) {
        g_sweep_hit      += dbg_hit;
        g_sweep_sentinel += dbg_sen;
        g_sweep_loads    += (uint64_t) load_e.size();        const uint64_t p  = ++g_sweep_passes;
        // 43 MoE layers x n_groups passes per ubatch step, so 128 lands inside the first step.
        // Also print the very first pass so the configuration is visible immediately.
        if (p == 1 || p % 128 == 0) {
            const uint64_t h = g_sweep_hit.load(), s = g_sweep_sentinel.load();
            const double   tot = (double) (h + s ? h + s : 1);
            LLAMA_LOG_WARN("MoE SWEEPDBG: %llu passes | HIT %.2f%% SENTINEL %.2f%% "
                           "(HIT must be 1/n_groups) | loads %llu (%.1f/pass) | cap=%d n_expert=%d "
                           "group=[%d,%d) n_sentinel=%d\n",
                           (unsigned long long) p, 100.0 * (double) h / tot, 100.0 * (double) s / tot,
                           (unsigned long long) g_sweep_loads.load(),
                           (double) g_sweep_loads.load() / (double) p,
                           c->capacity, c->n_expert, ud->e_lo, ud->e_hi, c->n_sentinel);
        }
    }
}

// map_custom1 shim for the first group, which has no predecessor to order against.
static void llama_moe_layer_remap_group_cb1(ggml_tensor * dst, const ggml_tensor * a,
                                            int ith, int nth, void * userdata) {
    llama_moe_layer_remap_group_cb(dst, a, nullptr, ith, nth, userdata);
}

bool llama_moe_prefill_sweep_enabled(void) {
    static const bool on = moe_env_on("LLAMA_MOE_PREFILL_SWEEP", false);
    return on;
}

int llama_moe_layer_cache_capacity(const llama_moe_layer_cache * c) {
    return c ? c->capacity : 0;
}

bool llama_moe_layer_cache_decoded(const llama_moe_layer_cache * c) {
    return c && c->pos_init; // set by the first single-token remap step for this cache
}

ggml_tensor * llama_moe_layer_cache_remap_group(llama_moe_layer_cache * c,
                                                ggml_context *          ctx0,
                                                ggml_tensor *           selected_experts,
                                                int                     group,
                                                ggml_tensor *           dep) {
    if (!c || c->capacity <= 0 || c->n_expert <= 0) {
        return nullptr;
    }
    const int64_t n_sel = selected_experts->ne[0] * selected_experts->ne[1];
    if (n_sel <= 0 || selected_experts->ne[2] != 1 || selected_experts->ne[3] != 1) {
        return nullptr;
    }
    // Losslessness needs one zero sentinel per routing position. Refuse rather than silently fall
    // back to a spare resident slot, which would inject a wrong expert's weights.
    if (c->n_sentinel < c->n_used) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("MoE prefill sweep unavailable: n_sentinel=%d < n_used=%d "
                           "(unset LLAMA_MOE_SENTINELS so the sweep can pick n_used)\n",
                           c->n_sentinel, c->n_used);
        }
        return nullptr;
    }
    const int n_groups = (c->n_expert + c->capacity - 1) / c->capacity;
    if (group < 0 || group >= n_groups) {
        return nullptr;
    }
    if ((int) c->group_ud.size() != n_groups) {
        c->group_ud.assign((size_t) n_groups, llama_moe_group_ud());
        for (int g = 0; g < n_groups; ++g) {
            c->group_ud[(size_t) g].c    = c;
            c->group_ud[(size_t) g].e_lo = g * c->capacity;
            c->group_ud[(size_t) g].e_hi = std::min((g + 1) * c->capacity, c->n_expert);
        }
    }
    void * ud = (void *) &c->group_ud[(size_t) group];
    if (dep) {
        return ggml_map_custom2(ctx0, selected_experts, dep, llama_moe_layer_remap_group_cb, 1, ud);
    }
    return ggml_map_custom1(ctx0, selected_experts, llama_moe_layer_remap_group_cb1, 1, ud);
}

// fork: one-shot VRAM audit (always on, WARN level). llama_moe_auto_capacity has to size the expert cache
// from dev_free measured at the FIRST MoE layer's cache creation - i.e. during graph BUILD, before the
// scheduler's compute buffer exists - so it can only hold back a flat reserve (default 512 MiB) for
// "KV/compute". For a model whose graph is large that guess is far too small: on dsv4 at n_ubatch 2048 the
// lightning-indexer score pair alone is ~512 * n_lid * n_ubatch bytes, i.e. gigabytes. The cache then
// over-commits and the driver silently backs the overflow with shared host memory, which costs real
// throughput with no error and no log line. Measured on this fork: default cap=44 (15.1 GiB cache) gave
// 11.6-12.9 tok/s decode while an explicit LLAMA_MOE_CACHE_CAP=30 (9.0 GiB) gave 14.4-16.7 - the default
// was the slower setting. This runs after everything is allocated and simply makes the situation visible,
// with the number needed to act on it. It deliberately does NOT change the capacity: picking the operating
// point is the user's call.
void llama_moe_vram_audit(void) {
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    ggml_backend_t backend = nullptr;
    for (auto & kv : g_moe_layer_caches) {
        if (kv.second && kv.second->dev_backend) { backend = kv.second->dev_backend; break; }
    }
    if (!backend) { return; }
    size_t dev_free = 0, dev_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(backend), &dev_free, &dev_total);
    if (dev_total == 0) { return; }
    const double gib = 1024.0*1024.0*1024.0;
    const double cache_gib = (double) g_moe_layer_vram_bytes / gib;
    const double free_gib  = (double) dev_free / gib;
    // A healthy steady state leaves room for the graph's transients. Under ~0.5 GiB free after everything
    // is allocated, the driver is almost certainly paging part of the working set over PCIe.
    // What this can and cannot tell you. On Windows/WDDM (and with CUDA's caching allocator) the free figure
    // reported here lands at ~0.00 GiB essentially regardless of how big the expert cache is: whatever
    // headroom auto capacity leaves gets absorbed afterwards. Measured on dsv4 IQ2 / 24 GiB by raising
    // LLAMA_MOE_VRAM_RESERVE_MB: budgeting 15.09 GiB of cache left 1.05 GiB unaccounted and 0.00 GiB free;
    // budgeting 14.19 GiB left 1.96 GiB unaccounted and STILL 0.00 GiB free. So a low free figure is NOT
    // evidence that the cache is spilling, and this audit must not claim that it is - an earlier version did,
    // and the claim was wrong. Capacity has to be chosen by measured throughput (llama_moe_auto_capacity's
    // reserve is the knob), not by this number.
    const size_t ac_free    = g_moe_autocap_free.load(std::memory_order_acquire);
    const size_t ac_planned = g_moe_autocap_planned.load(std::memory_order_acquire);
    LLAMA_LOG_WARN("MoE stream: VRAM AUDIT - expert cache %.2f GiB, %.2f GiB free of %.1f GiB after "
                   "allocation. Note the free figure absorbs whatever headroom is left and reads near zero "
                   "at any cache size on this platform, so it does NOT by itself indicate spilling; tune "
                   "LLAMA_MOE_CACHE_CAP / LLAMA_MOE_VRAM_RESERVE_MB against measured tok/s.\n",
                   cache_gib, free_gib, dev_total/gib);
    if (ac_free > 0) {
        LLAMA_LOG_WARN("MoE stream: VRAM AUDIT attribution - auto capacity saw %.2f GiB free and budgeted "
                       "%.2f GiB of cache, expecting to leave %.2f GiB; %.2f GiB free now, so %.2f GiB went "
                       "elsewhere after the capacity decision.\n",
                       ac_free/gib, ac_planned/gib, ((double) ac_free - (double) ac_planned)/gib,
                       free_gib, ((double) ac_free - (double) ac_planned - (double) dev_free)/gib);
    }
    // The figure the two lines above cannot give: whether the driver is backing part of this process's
    // video memory with host RAM. Printed unconditionally, including when it is zero - "no spill" is the
    // result worth confirming, and it is the only thing here that answers it.
    llama_moe_vram_seg seg;
    if (llama_moe_vram_spill_sample(dev_total, &seg)) {
        const double mib = 1024.0*1024.0;
        // local_used can exceed local_budget (and even the card's physical size): DXGI counts what this
        // process has COMMITTED to the segment, and WDDM lets a process over-commit and then pages the
        // excess out. So the over-budget delta is the actionable figure - it is how much the driver has
        // to keep shuffling - while nonlocal_used is only the part parked in host RAM at this instant.
        const double over = seg.local_used > seg.local_budget
                          ? (double) (seg.local_used - seg.local_budget) / mib : 0.0;
        LLAMA_LOG_WARN("MoE stream: VRAM SEGMENTS - local %.0f / %.0f MiB budget (over by %.0f MiB), "
                       "spilled to host (non-local) %.0f MiB of %.0f MiB budget. Over-budget or non-zero "
                       "spill means the working set does not fit and the driver is paging it over PCIe; a "
                       "small steady non-local figure is the pinned H2D staging arena, not the cache.\n",
                       seg.local_used/mib, seg.local_budget/mib, over, seg.nonlocal_used/mib,
                       seg.nonlocal_budget/mib);
    }
}

bool llama_moe_sweep_refill_pending(void) {
    return g_moe_sweep_dirty.load(std::memory_order_relaxed);
}

// fork: dump the resident expert set of every layer cache (LLAMA_MOE_RESDBG). The point is to settle a
// specific question by measurement rather than by comparing throughput: does a prefill sweep leave the
// VRAM caches holding a DIFFERENT set of experts than an ordinary prefill does? Run the same prompt with
// and without the sweep, diff these lines, and the answer is either "the sets differ, here is by how
// much" or "the sets match, so the decode difference is caused by something other than residency".
// Ordered and labelled by the registry's key tensor name (stable across runs, unlike pointers).
void llama_moe_dump_residency(const char * tag) {
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    std::vector<std::pair<std::string, llama_moe_layer_cache *>> cs;
    cs.reserve(g_moe_layer_caches.size());
    for (auto & kv : g_moe_layer_caches) {
        if (!kv.second) { continue; }
        const char * nm = (kv.first && kv.first->name[0]) ? kv.first->name : "?";
        cs.emplace_back(std::string(nm), kv.second);
    }
    std::sort(cs.begin(), cs.end(), [](const std::pair<std::string, llama_moe_layer_cache *> & a,
                                       const std::pair<std::string, llama_moe_layer_cache *> & b) {
        return a.first < b.first;
    });
    for (auto & pr : cs) {
        llama_moe_layer_cache * c = pr.second;
        std::vector<int> res;
        res.reserve((size_t) c->capacity);
        for (int s = 0; s < c->capacity; ++s) {
            const int e = c->slot_expert[(size_t) s];
            if (e >= 0) { res.push_back(e); }
        }
        std::sort(res.begin(), res.end());
        std::string ids;
        ids.reserve(res.size() * 4);
        char buf[16];
        for (size_t i = 0; i < res.size(); ++i) {
            snprintf(buf, sizeof(buf), "%s%d", i ? "," : "", res[i]);
            ids += buf;
        }
        LLAMA_LOG_WARN("MoE RESDBG %s %s cap=%d n=%d ids=%s\n",
                       tag, pr.first.c_str(), c->capacity, (int) res.size(), ids.c_str());
    }
}

void llama_moe_refill_vram_caches(void) {
    std::unique_lock<std::mutex> lock(g_moe_layer_mutex);
    if (!g_moe_sweep_dirty.exchange(false)) {
        return; // already consumed
    }
    const int64_t t0 = ggml_time_us();
    uint64_t bytes = 0;
    int n_caches = 0, n_loaded = 0;
    std::vector<char> ldbuf;

    for (auto & kv : g_moe_layer_caches) {
        llama_moe_layer_cache * c = kv.second;
        if (!c || c->capacity <= 0 || c->proj.empty() || !c->proj[0].dev) {
            continue;
        }
        if (!c->sweep_dirty) {
            continue; // never swept, or already refilled - do not re-pay another model's refill
        }
        c->sweep_dirty = false;
        const int cap = c->capacity;
        const int ne  = c->n_expert;

        // Same ranking the RAM tier and the startup VRAM prefill use: highest expert_score first,
        // ties by lowest index. The sweep updates expert_score on its first group of every step, so
        // by now the scores reflect the prompt that was just processed.
        std::vector<int> idx((size_t) ne);
        for (int e = 0; e < ne; ++e) { idx[(size_t) e] = e; }
        const std::vector<float> & sc = c->expert_score;
        const bool have_scores = ((int) sc.size() == ne);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
            const float sa = have_scores ? sc[(size_t) a] : 0.0f;
            const float sb = have_scores ? sc[(size_t) b] : 0.0f;
            if (sa != sb) { return sa > sb; }
            return a < b;
        });

        const int n_target = cap < ne ? cap : ne;
        std::vector<char> want((size_t) ne, 0);
        std::vector<int>  target;
        target.reserve((size_t) n_target);
        // Priority order matters, and NOT for the reason first assumed. Matching a non-sweep run's
        // resident set (which is what the frequency ranking does) is the wrong target: the resident set is
        // not what decode is waiting on. Measured root cause of the post-sweep decode regression is
        // g_moe_layer_mutex contention - the background loader drives the cache toward its EXPERT_SCORE
        // ranking and holds the mutex during H2D while doing so, and every decode MoE callback blocks on
        // it (lock wait 0.10ms -> 1.68ms per callback, x44 layers = +64ms/token). So the refill's job is to
        // leave the cache where the LOADER already wants it, draining its backlog: score first, frequency
        // only as filler.
        for (int t = 0; t < ne && (int) target.size() < n_target; ++t) {
            const int e = idx[(size_t) t];
            if (!want[(size_t) e]) {
                want[(size_t) e] = 1;
                target.push_back(e);
            }
        }
        for (const int32_t e : c->freq_ranked) {
            if ((int) target.size() >= n_target) { break; }
            if (e >= 0 && e < ne && !want[(size_t) e]) {
                want[(size_t) e] = 1;
                target.push_back(e);
            }
        }

        // Load every target expert that is not already resident, reusing slots whose current
        // occupant is not itself a target. moe_layer_load_into_slot evicts that occupant correctly
        // (drops it from slot_table/stale_table before overwriting the slab). A usable slot always
        // exists: with k targets already resident, at most k slots hold wanted experts, leaving
        // cap - k >= n_target - k free for the n_target - k that still need loading.
        int s_scan = 0;
        for (const int e : target) {
            if (c->expert_slot[(size_t) e] >= 0) {
                continue; // already resident, keep it where it is
            }
            while (s_scan < cap) {
                const int occ = c->slot_expert[(size_t) s_scan];
                if (!c->slot_loading[(size_t) s_scan] && (occ < 0 || !want[(size_t) occ])) { break; }
                s_scan++;
            }
            if (s_scan >= cap) {
                break; // unreachable given the counting argument above
            }
            moe_layer_load_into_slot(c, e, s_scan, ldbuf);
            for (auto & pr : c->proj) { bytes += (uint64_t) pr.stride; }
            n_loaded++;
            s_scan++;
        }
        n_caches++;
    }

    // Optional (LLAMA_MOE_PREFILL_DRAIN=1, DEFAULT OFF): drain the loader's backlog HERE, at the
    // prefill/decode boundary, so it is paid once in the prefill window rather than trickling through decode.
    // The mutex must be RELEASED while waiting - the loader needs it to make progress.
    //
    // Measured NET NEGATIVE on both axes and therefore off by default. Two things are wrong with it:
    //
    // 1. The convergence predicate cannot be reached. g_moe_loader_idle counts consecutive passes with
    //    pass_promoted == 0, and pass_promoted INCLUDES llama_moe_loader_fill_ram (see the loader main loop).
    //    fill_ram chases the score-top ram_capacity experts out of n_expert, a target the pool cannot hold
    //    and that keeps shifting, so after a sweep the loader essentially never reports an idle pass and this
    //    wait always runs to the deadline below. "The loader has nothing left to do" is not a state that
    //    occurs on this path.
    // 2. Its premise is stale. The backlog used to block decode because the loader held g_moe_layer_mutex
    //    across its H2D; the claim/write/publish split already moved that out of the lock, so a backlog no
    //    longer stalls the decode callbacks it was meant to protect.
    //
    // With it on, decode is SLOWER, not faster - the opposite of the intent - on top of the deadline's worth
    // of added wall clock. Mechanism for the slowdown is not isolated; the plausible candidates are that the
    // deadline's worth of full-cadence loader activity (the idleness backoff never engages, since no pass is
    // idle) re-shuffles the VRAM cache away from the placement this refill just made, and leaves the RAM pool
    // mid-thrash. Do not re-enable without measuring both decode tok/s AND total wall clock.
    const int64_t drain0 = ggml_time_us();
    if (moe_env_on("LLAMA_MOE_PREFILL_DRAIN", false)) {
        lock.unlock();
        g_moe_loader_idle.store(0, std::memory_order_release);
        const int64_t deadline = drain0 + 20LL * 1000 * 1000; // 20s cap
        while (g_moe_loader_idle.load(std::memory_order_acquire) < 3 && ggml_time_us() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        lock.lock();
    }
    const double drain_ms = (ggml_time_us() - drain0) / 1000.0;

    const double ms  = (ggml_time_us() - t0) / 1000.0;
    const double gib = bytes / (1024.0*1024.0*1024.0);
    LLAMA_LOG_WARN("MoE stream: refilled VRAM caches after the prefill sweep - %.1f GiB / %d experts "
                   "across %d layers in %.1fs (%.2f GiB/s), loader drain %.2fs. Without this decode runs on the "
                   "sweep's last expert group happened to be.\n",
                   gib, n_loaded, n_caches, ms/1000.0, ms > 0 ? gib/(ms/1000.0) : 0.0, drain_ms/1000.0);
}

ggml_tensor * llama_moe_layer_cache_remap(llama_moe_layer_cache * c,
                                          ggml_context *          ctx0,
                                          ggml_tensor *           selected_experts,
                                          ggml_tensor *           weights,
                                          float                   threshold,
                                          int                     sync_budget) {
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
    // working set (bounded by capacity via the threshold), so budget must not throttle it. The caller
    // (build_moe_ffn) already resolved the effective budget once (env value, else 2 on the async path,
    // else 0) so both the decode-path selection and this cache agree - do not re-read the env here. The
    // HYBRID_BUDGET probe still applies only when no budget was passed (the static per-layer hybrid runs
    // with SYNC_BUDGET unset so it does not force ALL layers onto the CPU path).
    c->sync_budget = 0;
    if (selected_experts->ne[1] <= 1) {
        if (sync_budget > 0) {
            c->sync_budget = sync_budget;
        } else if (const char * hb = getenv("LLAMA_MOE_HYBRID_BUDGET")) {
            c->sync_budget = atoi(hb);
            if (c->sync_budget < 0) { c->sync_budget = 0; }
        }
    }
    // Coverage mode (LLAMA_MOE_SYNC_COVER=F, decode only): size the sync set by weight coverage instead of
    // a fixed budget - sync missing top-K experts weight-descending but stop once (cached+synced) gate weight
    // reaches F, capped at sync_budget (top-2). On the async path this is ON by default at F=0.2 (set via
    // common.cpp set_if_unset when --moe-stream-async is passed); a cross-domain sweep found 0.2 gives ~+30%
    // decode on average (measured +2%..+51%) at coherent quality, while 0.15-0.19 is a seed-fragile cliff
    // (repeat loops), so 0.2 is the safe knee. Tuned on hy3 (top-2 weight ~0.34); other MoEs may want a
    // different F. LLAMA_MOE_SYNC_COVER=0 opts out (atof 0 -> f<=0 -> plain map_custom1 == fixed budget=2).
    // The earlier garbage bug (COV=0.9 dropping real top-2 loads) was cov_have summing over the full
    // first-decode order of 8 experts (weights sum to 1.0), which cleared any threshold <1 and fired spurious
    // early-stops; fixed by gating coverage off on first_decode and only summing over order_n (see the
    // early-stop block in remap_cb). Verify with LLAMA_MOE_COVDBG (early-stopped misses/tok vs threshold).
    c->sync_cover = 0.0f;
    if (weights && selected_experts->ne[1] == 1 && weights->ne[1] == selected_experts->ne[0]) {
        if (const char * cv = getenv("LLAMA_MOE_SYNC_COVER")) {
            const float f = (float) atof(cv);
            if (f > 0.0f) {
                c->sync_cover = f > 1.0f ? 1.0f : f;
                return ggml_map_custom2(ctx0, selected_experts, weights, llama_moe_layer_remap_cov_cb, 1, c);
            }
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

// fork: re-decide the expert-cache capacity once the REAL compute-buffer size is known.
//
// llama_moe_auto_capacity has to run at the first MoE layer's cache creation, i.e. during graph BUILD,
// before any compute buffer exists - so on its own it can only hold back a flat guess
// (LLAMA_MOE_VRAM_RESERVE_MB, 3 GiB). That guess cannot be right: the buffer is O(n_ubatch x n_kv) and
// measured 1170 MiB at n_ctx 16384 / ub 2048 but 34 GiB at n_ctx 131072 / ub 2048 before the
// lightning-indexer scoring was chunked. Too small a guess overcommits VRAM (the driver then pages part
// of the working set over PCIe, invisible in every VRAM counter and visible only as throughput); too
// large leaves whole capacity steps unused.
//
// So: let the first reserve pass run with the flat guess (it only has to BUILD, and the buffer size does
// not depend on the capacity - the slot pools live in their own buffer), then publish the measured size
// here, drop the caches, and let the next graph build re-create them against the real number. The
// earlier attempt at this deferred cache creation entirely, which made build_moe_ffn fall back to the
// compaction gather whose transient is the full n_expert-wide expert tensor - a graph that needs MORE
// compute buffer than the real one, so the measurement was useless. Keeping a real (flat-guess) cache
// for the measurement pass avoids that: same graph shape, same buffer.
//
// Returns true if the caps were invalidated and the caller must re-reserve. The RAM residency pools are
// dropped with the caches, but llama_moe_prefill_once has not run yet at this point in startup, so no
// prefill work is thrown away. The measurement pass's VRAM prewarm is skipped entirely when the caller
// wraps its reserve in llama_moe_cache_defer_prewarm (llama_context does), so the dropped caches cost
// allocation but no disk traffic.
bool llama_moe_recap_from_compute_reserve(ggml_backend_sched_t sched, size_t compute_bytes) {
    if (compute_bytes == 0) { return false; }
    if (getenv("LLAMA_MOE_CACHE_CAP"))          { return false; } // capacity pinned by hand
    if (!moe_env_on("LLAMA_MOE_AUTOCAP_RECAP", true)) { return false; }

    // One shot per PROCESS, not per context. The point of the recap is to replace the flat
    // pre-measurement guess with a measured compute-buffer size, and that only has to happen once.
    // g_moe_compute_reserve and g_moe_cached_cap are process-global, so letting a second context (a
    // draft/MTP or embedding context, created after the main one) recap here does two wrong things:
    //   - it overwrites the measured reserve of the context that actually sets the VRAM ceiling with its
    //     own, much smaller, compute buffer;
    //   - it clears g_moe_cached_cap and re-runs llama_moe_auto_capacity at a point where dev_free is
    //     already spoken for by the first context's caches, so it lands in the out-of-VRAM fallback
    //     instead of reusing the capacity that was just fitted to this device.
    // Measured on qwen3.8-flash-next + MTP: the draft context built a cap-100 cache, recapped, freed it,
    // and rebuilt at the fallback capacity. Reusing the first context's cap is both cheaper and correct.
    if (g_moe_compute_reserve.load(std::memory_order_acquire) != 0) { return false; }

    // Only the per-layer (async) caches are rebuilt here. If the per-tensor compaction pools are in use
    // the graph holds their device tensors too, and dropping them is not worth a second code path.
    if (!g_moe_cache_pools.empty()) { return false; }

    // Drop only the caches THIS context built (llama_moe_layer_cache::sched). g_moe_layer_caches is
    // process-global: a draft/embedding context constructed after the main one used to free the main
    // model's caches here and then re-reserve only its own graph, leaving the main context's graphs
    // pointing at deleted caches and freed device buffers - an instant segfault on its next decode.
    // Checked before the loader stop below, which is process-wide: a context that owns nothing must
    // leave the loader running for the one that does.
    {
        std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
        bool owned = false;
        for (auto & kv : g_moe_layer_caches) {
            if (kv.second && kv.second->sched == sched) { owned = true; break; }
        }
        if (!owned) { return false; }
    }

    if (g_moe_loader_run.exchange(false)) {
        if (g_moe_loader_thread.joinable()) {
            g_moe_loader_thread.join();
        }
    }

    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);

    for (auto it = g_moe_layer_caches.begin(); it != g_moe_layer_caches.end(); ) {
        llama_moe_layer_cache * c = it->second;
        if (!c || c->sched != sched) { ++it; continue; }
        for (auto & pr : c->proj) {
            if (pr.ram) {
                llama_moe_ram_free(pr.ram, (size_t) c->ram_capacity * (size_t) pr.stride);
                pr.ram = nullptr;
            }
            if (pr.fp)    { fclose(pr.fp);    pr.fp    = nullptr; }
            if (pr.fp_ld) { fclose(pr.fp_ld); pr.fp_ld = nullptr; }
        }
        if (c->buffer) {
            // each freed buffer was counted into g_moe_layer_vram_bytes when the cache was built, so take
            // it back out. Without this the second pass adds on top of the first and the VRAM AUDIT line
            // plus the "approaching device capacity" warning both report roughly double.
            const uint64_t sz = (uint64_t) ggml_backend_buffer_get_size(c->buffer);
            g_moe_layer_vram_bytes = g_moe_layer_vram_bytes > sz ? g_moe_layer_vram_bytes - sz : 0;
            ggml_backend_buffer_free(c->buffer);
            c->buffer = nullptr;
        }
        if (c->ctx)    { ggml_free(c->ctx);                   c->ctx    = nullptr; }
        delete c;
        it = g_moe_layer_caches.erase(it);
    }

    g_moe_compute_reserve.store(compute_bytes, std::memory_order_release);
    g_moe_cached_cap = 0;
    g_moe_autocap_logged = false;

    return true;
}

// Same gates as llama_moe_recap_from_compute_reserve, minus the two it cannot know before the reserve
// (a zero compute buffer, and whether this context ends up owning any cache). So "pending" means "the
// recap will fire unless one of those two declines it" - which is why llama_moe_cache_defer_prewarm(false)
// repairs the caches instead of assuming they are gone.
bool llama_moe_recap_pending(void) {
    if (getenv("LLAMA_MOE_CACHE_CAP"))                { return false; }
    if (!moe_env_on("LLAMA_MOE_AUTOCAP_RECAP", true)) { return false; }
    if (g_moe_compute_reserve.load(std::memory_order_acquire) != 0) { return false; }
    return g_moe_cache_pools.empty();
}

void llama_moe_cache_defer_prewarm(bool defer) {
    // LLAMA_MOE_DEFER_PREWARM=0 restores the old behaviour (the measurement pass fills all `capacity`
    // slots and the recap throws them away), so what deferring is worth - and that it costs nothing
    // downstream - can be A/B'd inside ONE session with one binary instead of across two builds.
    static const bool on = moe_env_on("LLAMA_MOE_DEFER_PREWARM", true);
    g_moe_defer_prewarm.store(defer && on, std::memory_order_release);
    if (defer) {
        return;
    }
    // Clearing the flag: finish the prewarm of any deferred cache that is still alive. Normally the recap
    // has already deleted all of them and this loop finds nothing; it matters only when the recap declined
    // after the caller had already deferred, where the alternative is a cache holding one real expert.
    std::lock_guard<std::mutex> lock(g_moe_layer_mutex);
    std::vector<char> ldbuf;
    for (auto & kv : g_moe_layer_caches) {
        llama_moe_layer_cache * c = kv.second;
        if (!c || !c->prewarm_pending) { continue; }
        c->prewarm_pending = false;
        for (int s = 0; s < c->capacity && s < c->n_expert; ++s) {
            // leave anything the loader already settled alone: loading expert s into slot s while s (or
            // slot s) is occupied elsewhere would leave a duplicate slot_expert entry behind.
            if (c->slot_expert[(size_t) s] >= 0 || c->expert_slot[(size_t) s] >= 0) { continue; }
            moe_layer_load_into_slot(c, s, s, ldbuf);
        }
    }
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

    // Persist the learned per-layer expert scores (LLAMA_MOE_SCORE_FILE) so the next run warm-starts
    // residency instead of re-learning the hot set over ~1400 tokens. Keyed by expert tensor name (stable
    // across runs). Written before the caches are torn down below.
    if (const char * spath = llama_moe_score_file()) {
        FILE * f = fopen(spath, "wb");
        if (f) {
            size_t written = 0;
            for (auto & kv : g_moe_layer_caches) {
                llama_moe_layer_cache * c = kv.second;
                if (!c || c->proj.empty() || !c->proj[0].src) { continue; }
                if ((int) c->expert_score.size() != c->n_expert || c->n_expert <= 0) { continue; }
                const char * nm = c->proj[0].src->name;
                const uint32_t nlen = (uint32_t) strnlen(nm, GGML_MAX_NAME);
                if (nlen == 0) { continue; }
                const uint32_t ne = (uint32_t) c->n_expert;
                fwrite(&nlen, sizeof(nlen), 1, f);
                fwrite(nm, 1, nlen, f);
                fwrite(&ne, sizeof(ne), 1, f);
                fwrite(c->expert_score.data(), sizeof(float), ne, f);
                written++;
            }
            fclose(f);
            LLAMA_LOG_WARN("MoE stream: saved expert scores for %zu layers to %s (next run warm-starts).\n",
                           written, spath);
        } else {
            LLAMA_LOG_WARN("MoE stream: could not open %s to save expert scores.\n", spath);
        }
    }

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
    }
    // free the single global pinned staging arena (loader stopped, mutex held: no concurrent access)
    if (g_moe_stage_buf) {
        ggml_backend_buffer_free(g_moe_stage_buf);
        g_moe_stage_buf     = nullptr;
        g_moe_stage_base    = nullptr;
        g_moe_stage_slot    = 0;
        g_moe_stage_nthread = 0;
        g_moe_stage_nproj   = 0;
    }
}
