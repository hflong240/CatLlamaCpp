// Deep-queue ASYNC read benchmark - the pattern that can actually saturate an NVMe (vs 16 blocking threads).
// Keeps QD overlapped reads in flight at all times via an array of OVERLAPPED + events, re-issuing on each
// completion. This is the read path a real high-throughput loader would use, and the one the earlier
// benchmarks (synchronous blocking fread/ReadFile) NEVER tested.
//
// Integrity fixes vs the earlier tool:
//  - reports GiB/s from ACTUAL bytes read, not an assumed total (the 58 GiB/s bug was a failed read that
//    still printed the nominal total).
//  - every read offset is sector-aligned (required for NO_BUFFERING); on any ReadFile error it PRINTS the
//    error and counts zero, so a broken run is obvious instead of looking fast.
//
// Usage: read_async <file> <QD> <blockMB> <buffered:0|1> [seq|scat]
// Run after purge_standby for a cold read.
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

static const uint64_t TOTAL  = 16ull * 1024 * 1024 * 1024; // read ~16 GiB of ACTUAL data
static const uint64_t SECTOR = 4096;

static double now_s() {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) f.QuadPart;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <file> <QD> <blockMB> <buffered:0|1> [seq|scat]\n", argv[0]); return 2; }
    const char *  path  = argv[1];
    const int     QD    = atoi(argv[2]) > 0 ? atoi(argv[2]) : 32;
    const uint64_t blk  = (uint64_t) atoi(argv[3]) * 1024 * 1024;
    const bool    unbuf = atoi(argv[4]) == 0;
    const bool    scat  = argc > 5 ? (strcmp(argv[5], "scat") == 0) : true;
    if (blk == 0 || (blk % SECTOR)) { fprintf(stderr, "blockMB must be a multiple of 4KB sectors\n"); return 2; }

    DWORD flags = FILE_FLAG_OVERLAPPED | (unbuf ? FILE_FLAG_NO_BUFFERING : 0);
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "open failed %lu\n", GetLastError()); return 2; }
    LARGE_INTEGER fs; GetFileSizeEx(h, &fs);
    const uint64_t fsz = (uint64_t) fs.QuadPart;
    const uint64_t maxblk = fsz / blk;

    std::vector<OVERLAPPED> ov(QD);
    std::vector<HANDLE>     ev(QD);
    std::vector<char*>      buf(QD);
    uint64_t next_off = 0;                 // for seq
    uint32_t seed = 0x9e3779b9u;
    auto next_offset = [&](int slot) -> uint64_t {
        if (scat) {
            seed = seed * 1664525u + 1013904223u;
            uint64_t bi = ((uint64_t) seed * 2654435761u + (uint32_t) slot * 40503u) % (maxblk ? maxblk : 1);
            return bi * blk;               // block-aligned => sector-aligned
        } else {
            uint64_t o = next_off; next_off += blk;
            if (o + blk > fsz) { o = 0; next_off = blk; }
            return o;
        }
    };

    for (int i = 0; i < QD; ++i) {
        buf[i] = (char*) VirtualAlloc(NULL, blk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        ev[i]  = CreateEventA(NULL, FALSE, FALSE, NULL);
    }

    uint64_t issued = 0, done_bytes = 0, errors = 0;
    double t0 = now_s();
    // prime QD reads
    for (int i = 0; i < QD; ++i) {
        memset(&ov[i], 0, sizeof(OVERLAPPED));
        uint64_t o = next_offset(i);
        ov[i].Offset = (DWORD)(o & 0xFFFFFFFF); ov[i].OffsetHigh = (DWORD)(o >> 32); ov[i].hEvent = ev[i];
        DWORD got = 0;
        if (!ReadFile(h, buf[i], (DWORD) blk, &got, &ov[i])) {
            DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) { errors++; ResetEvent(ev[i]); SetEvent(ev[i]); }
        }
        issued += blk;
    }
    // reap + re-issue until we've read TOTAL
    while (done_bytes < TOTAL) {
        DWORD idx = WaitForMultipleObjects(QD, ev.data(), FALSE, INFINITE);
        int i = (int) (idx - WAIT_OBJECT_0);
        if (i < 0 || i >= QD) break;
        DWORD got = 0;
        if (GetOverlappedResult(h, &ov[i], &got, FALSE) && got > 0) {
            done_bytes += got;
        } else {
            errors++;
        }
        if (done_bytes >= TOTAL) break;
        memset(&ov[i], 0, sizeof(OVERLAPPED));
        uint64_t o = next_offset(i);
        ov[i].Offset = (DWORD)(o & 0xFFFFFFFF); ov[i].OffsetHigh = (DWORD)(o >> 32); ov[i].hEvent = ev[i];
        DWORD g2 = 0;
        if (!ReadFile(h, buf[i], (DWORD) blk, &g2, &ov[i])) {
            DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) { errors++; SetEvent(ev[i]); }
        }
        issued += blk;
    }
    double dt = now_s() - t0;
    printf("%s QD=%d blk=%lluMB %s: read %.2f GiB in %.2fs = %.2f GiB/s  (errors=%llu)\n",
           scat ? "scat" : "seq ", QD, blk/(1024*1024), unbuf ? "UNBUF" : "buf  ",
           done_bytes / 1073741824.0, dt, (done_bytes / 1073741824.0) / dt, errors);

    for (int i = 0; i < QD; ++i) { if (buf[i]) VirtualFree(buf[i], 0, MEM_RELEASE); if (ev[i]) CloseHandle(ev[i]); }
    CloseHandle(h);
    return 0;
}
