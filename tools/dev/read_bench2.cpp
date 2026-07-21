// Cold-disk read benchmark v2: measure the headroom above buffered fread.
// Uses Win32 ReadFile so we can toggle FILE_FLAG_NO_BUFFERING (unbuffered DMA straight into our buffer,
// skipping the OS page-cache copy). Two patterns:
//   seq  : contiguous large blocks per thread (favorable; bulk fill upper bound)
//   scat : scattered `blockMB` reads at random aligned offsets (mimics the real loader reading experts
//          at foff0 + e*stride - the actual prefill / parallel_load access pattern)
// Usage: read_bench2 <file> <seq|scat> <threads> <blockMB> <buffered:0|1>
// Run after purge_standby for a cold read.
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <thread>
#include <vector>
#include <atomic>

static const uint64_t TOTAL = 16ull * 1024 * 1024 * 1024; // 16 GiB total across all threads
static const uint64_t SECTOR = 4096;                       // alignment for NO_BUFFERING

static uint64_t file_size(const char * path) {
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER s; GetFileSizeEx(h, &s); CloseHandle(h);
    return (uint64_t) s.QuadPart;
}

static double now_s() {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) f.QuadPart;
}

// read `bytes` from `path` in `blk`-sized chunks. seq: start at `off`, contiguous. scat: random aligned
// offsets across the whole file. unbuffered: FILE_FLAG_NO_BUFFERING (aligned buffer + aligned offsets).
static void worker(const char * path, uint64_t off, uint64_t bytes, uint64_t blk, bool scat, bool unbuf,
                   uint64_t fsz, uint32_t seed) {
    DWORD flags = FILE_ATTRIBUTE_NORMAL | (unbuf ? FILE_FLAG_NO_BUFFERING : 0);
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
    if (h == INVALID_HANDLE_VALUE) return;
    // page-aligned buffer (satisfies NO_BUFFERING sector alignment)
    char * buf = (char *) VirtualAlloc(NULL, blk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buf) { CloseHandle(h); return; }
    const uint64_t maxblk = (fsz / blk);
    uint64_t left = bytes;
    uint64_t cur = off;
    while (left > 0) {
        uint64_t o;
        if (scat) {
            seed = seed * 1664525u + 1013904223u;
            uint64_t bi = ((uint64_t) seed * 2654435761u) % (maxblk ? maxblk : 1);
            o = bi * blk;                       // block-aligned => sector-aligned since blk % SECTOR == 0
        } else {
            o = cur; cur += blk;
            if (o + blk > fsz) { o = 0; cur = blk; }
        }
        OVERLAPPED ov; memset(&ov, 0, sizeof(ov));
        ov.Offset     = (DWORD) (o & 0xFFFFFFFF);
        ov.OffsetHigh = (DWORD) (o >> 32);
        DWORD got = 0;
        if (!ReadFile(h, buf, (DWORD) blk, &got, &ov) || got == 0) break;
        left -= got;
    }
    VirtualFree(buf, 0, MEM_RELEASE);
    CloseHandle(h);
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s <file> <seq|scat> <threads> <blockMB> <buffered:0|1>\n", argv[0]); return 2; }
    const char *  path  = argv[1];
    const bool    scat  = strcmp(argv[2], "scat") == 0;
    const int     nt    = atoi(argv[3]) > 0 ? atoi(argv[3]) : 16;
    const uint64_t blk  = (uint64_t) atoi(argv[4]) * 1024 * 1024;
    const bool    unbuf = atoi(argv[5]) == 0;   // buffered:0 => unbuffered
    const uint64_t fsz = file_size(path);
    if (fsz < TOTAL || blk == 0 || (blk % SECTOR)) { fprintf(stderr, "bad args/file\n"); return 2; }

    const uint64_t per = TOTAL / nt;
    const uint64_t stride = nt > 1 ? (fsz - per) / (nt - 1) : 0;
    double t0 = now_s();
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t) {
        uint64_t o = scat ? 0 : (uint64_t) t * stride;
        if (!scat && o + per > fsz) o = fsz - per;
        pool.emplace_back(worker, path, o, per, blk, scat, unbuf, fsz, 0x9e3779b9u + (uint32_t) t * 2654435761u);
    }
    for (auto & th : pool) th.join();
    double dt = now_s() - t0;
    printf("%s t=%d blk=%lluMB %s: %.2f GiB in %.2fs = %.2f GiB/s\n",
           scat ? "scat" : "seq ", nt, blk / (1024*1024), unbuf ? "UNBUF" : "buf  ",
           TOTAL / 1073741824.0, dt, (TOTAL / 1073741824.0) / dt);
    return 0;
}
