// Cold-disk read benchmark: decompose "raw NVMe ceiling" vs "access-pattern / threading" effects.
// Reads a fixed volume from a large file in three patterns. Run after purge_standby for a cold read.
//   seq1   : single thread, sequential large blocks   (what the serial loader fill_ram approximates)
//   parN   : N threads, each a contiguous region       (what the prefill / parallel_load does)
//   randB  : single thread, random blocks of B bytes   (what scattered decode expert reads approximate)
// Usage: read_bench <file> <mode> [threads_or_blockKB]
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <thread>
#include <vector>
#include <atomic>

static const uint64_t TOTAL = 16ull * 1024 * 1024 * 1024; // read 16 GiB total
static const size_t   BLK   = 4ull * 1024 * 1024;         // 4 MiB block for seq/par

static uint64_t file_size(const char * path) {
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER s; GetFileSizeEx(h, &s); CloseHandle(h);
    return (uint64_t) s.QuadPart;
}

// read `bytes` starting at `off` in `blk`-sized chunks using a private handle
static void read_range(const char * path, uint64_t off, uint64_t bytes, size_t blk, std::atomic<uint64_t> * done) {
    FILE * f = fopen(path, "rb");
    if (!f) return;
    _fseeki64(f, (int64_t) off, SEEK_SET);
    std::vector<char> buf(blk);
    uint64_t left = bytes;
    while (left > 0) {
        size_t want = (size_t) (left < blk ? left : blk);
        size_t got  = fread(buf.data(), 1, want, f);
        if (got == 0) break;
        left -= got;
        if (done) done->fetch_add(got, std::memory_order_relaxed);
    }
    fclose(f);
}

static double now_s() {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) f.QuadPart;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <file> <seq1|parN|randKB> [n]\n", argv[0]); return 2; }
    const char * path = argv[1];
    const char * mode = argv[2];
    const int    n    = argc > 3 ? atoi(argv[3]) : 16;
    const uint64_t fsz = file_size(path);
    if (fsz < TOTAL) { fprintf(stderr, "file too small\n"); return 2; }

    double t0 = now_s();
    if (strcmp(mode, "seq1") == 0) {
        read_range(path, 0, TOTAL, BLK, nullptr);
    } else if (strncmp(mode, "par", 3) == 0) {
        const int nt = atoi(mode + 3) > 0 ? atoi(mode + 3) : n;
        // each thread reads a contiguous TOTAL/nt slice, slices spread across the file
        const uint64_t per = TOTAL / nt;
        const uint64_t stride = (fsz - per) / (nt > 1 ? nt - 1 : 1); // spread start offsets across the file
        std::vector<std::thread> pool;
        for (int t = 0; t < nt; ++t) {
            uint64_t off = (uint64_t) t * stride;
            if (off + per > fsz) off = fsz - per;
            pool.emplace_back(read_range, path, off, per, BLK, nullptr);
        }
        for (auto & th : pool) th.join();
        printf("threads=%d ", nt);
    } else if (strncmp(mode, "rand", 4) == 0) {
        const size_t bkb = atoi(mode + 4) > 0 ? (size_t) atoi(mode + 4) : 512;
        const size_t blk = bkb * 1024;
        FILE * f = fopen(path, "rb");
        if (!f) return 2;
        std::vector<char> buf(blk);
        const uint64_t nblocks = TOTAL / blk;
        const uint64_t maxstart = (fsz - blk) / blk;
        uint32_t seed = 0x9e3779b9u;
        for (uint64_t i = 0; i < nblocks; ++i) {
            seed = seed * 1664525u + 1013904223u;      // LCG, no Math.random needed
            uint64_t bi = ((uint64_t) seed * 2654435761u) % maxstart;
            _fseeki64(f, (int64_t) (bi * blk), SEEK_SET);
            fread(buf.data(), 1, blk, f);
        }
        fclose(f);
        printf("blockKB=%zu ", bkb);
    } else { fprintf(stderr, "bad mode\n"); return 2; }
    double dt = now_s() - t0;
    printf("%s: %.2f GiB in %.2fs = %.2f GiB/s\n", mode, TOTAL / 1073741824.0, dt, (TOTAL / 1073741824.0) / dt);
    return 0;
}
