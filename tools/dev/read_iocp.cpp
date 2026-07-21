// IOCP-based deep-queue async read benchmark - removes the 64-handle WaitForMultipleObjects limit so we
// can test QD up to 256+. This is the canonical Windows high-throughput read path (single completion port,
// N reads in flight, reap+reissue). If THIS cannot beat the ~7 GiB/s that blocking fread already gets, then
// 7 GiB/s is the real cold-scattered ceiling for this drive/pattern, not a software limit.
//
// Integrity: reports GiB/s from ACTUAL completed bytes; any read error is counted and printed.
// Usage: read_iocp <file> <QD> <blockMB> <buffered:0|1> [seq|scat]
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

static const uint64_t TOTAL  = 16ull * 1024 * 1024 * 1024;
static const uint64_t SECTOR = 4096;

struct Req { OVERLAPPED ov; char * buf; };

static double now_s() {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) f.QuadPart;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <file> <QD> <blockMB> <buffered:0|1> [seq|scat]\n", argv[0]); return 2; }
    const char *   path  = argv[1];
    const int      QD    = atoi(argv[2]) > 0 ? atoi(argv[2]) : 64;
    const uint64_t blk   = (uint64_t) atoi(argv[3]) * 1024 * 1024;
    const bool     unbuf = atoi(argv[4]) == 0;
    const bool     scat  = argc > 5 ? (strcmp(argv[5], "scat") == 0) : true;
    if (blk == 0 || (blk % SECTOR)) { fprintf(stderr, "blockMB must be multiple of 4KB\n"); return 2; }

    DWORD flags = FILE_FLAG_OVERLAPPED | (unbuf ? FILE_FLAG_NO_BUFFERING : 0);
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "open failed %lu\n", GetLastError()); return 2; }
    LARGE_INTEGER fs; GetFileSizeEx(h, &fs);
    const uint64_t fsz = (uint64_t) fs.QuadPart;
    const uint64_t maxblk = fsz / blk;

    HANDLE iocp = CreateIoCompletionPort(h, NULL, 0, 0);
    if (!iocp) { fprintf(stderr, "iocp failed %lu\n", GetLastError()); return 2; }

    std::vector<Req> reqs(QD);
    uint64_t next_off = 0; uint32_t seed = 0x9e3779b9u;
    auto next_offset = [&](int slot) -> uint64_t {
        if (scat) { seed = seed * 1664525u + 1013904223u;
            uint64_t bi = ((uint64_t) seed * 2654435761u + (uint32_t) slot * 40503u) % (maxblk ? maxblk : 1);
            return bi * blk; }
        uint64_t o = next_off; next_off += blk; if (o + blk > fsz) { o = 0; next_off = blk; } return o;
    };
    auto issue = [&](int i) {
        memset(&reqs[i].ov, 0, sizeof(OVERLAPPED));
        uint64_t o = next_offset(i);
        reqs[i].ov.Offset = (DWORD)(o & 0xFFFFFFFF); reqs[i].ov.OffsetHigh = (DWORD)(o >> 32);
        DWORD got = 0;
        BOOL ok = ReadFile(h, reqs[i].buf, (DWORD) blk, &got, &reqs[i].ov);
        if (!ok && GetLastError() != ERROR_IO_PENDING) return false;
        return true;
    };

    for (int i = 0; i < QD; ++i) reqs[i].buf = (char*) VirtualAlloc(NULL, blk, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);

    uint64_t done_bytes = 0, errors = 0;
    double t0 = now_s();
    for (int i = 0; i < QD; ++i) { if (!issue(i)) errors++; }
    while (done_bytes < TOTAL) {
        DWORD bytes = 0; ULONG_PTR key = 0; OVERLAPPED * po = NULL;
        BOOL ok = GetQueuedCompletionStatus(iocp, &bytes, &key, &po, INFINITE);
        if (!ok && !po) break;
        if (ok && bytes > 0) done_bytes += bytes; else errors++;
        if (done_bytes >= TOTAL) break;
        int i = (int) ((Req*) po - reqs.data());   // recover slot from OVERLAPPED address
        if (i >= 0 && i < QD) { if (!issue(i)) errors++; }
    }
    double dt = now_s() - t0;
    printf("%s QD=%d blk=%lluMB %s: read %.2f GiB in %.2fs = %.2f GiB/s  (errors=%llu)\n",
           scat ? "scat" : "seq ", QD, blk/(1024*1024), unbuf ? "UNBUF":"buf  ",
           done_bytes/1073741824.0, dt, (done_bytes/1073741824.0)/dt, errors);

    for (int i = 0; i < QD; ++i) if (reqs[i].buf) VirtualFree(reqs[i].buf, 0, MEM_RELEASE);
    CloseHandle(iocp); CloseHandle(h);
    return 0;
}
