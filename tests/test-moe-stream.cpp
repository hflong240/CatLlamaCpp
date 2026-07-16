// Standalone unit test for the MoE streaming cache core (offset math + LRU).
// Compile without CMake for a fast local check, e.g. (MSVC):
//   cl /EHsc /std:c++17 /I src tests\test-moe-stream.cpp src\llama-moe-stream.cpp
#include "llama-moe-stream.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL: %s (line %d)\n", #cond, __LINE__); g_fail++; } } while (0)

int main() {
    // synthetic model file: `n_expert` experts of `stride` bytes each, starting
    // at `base`; expert e is filled with the byte value (e + 1).
    const uint32_t n_expert = 4;
    const uint64_t stride   = 256;
    const uint64_t base     = 1024;

    const std::string path = "test-moe-stream.bin";
    {
        FILE * f = fopen(path.c_str(), "wb");
        CHECK(f != nullptr);
        const std::vector<uint8_t> pad(base, 0);
        fwrite(pad.data(), 1, pad.size(), f);
        for (uint32_t e = 0; e < n_expert; ++e) {
            const std::vector<uint8_t> blk(stride, (uint8_t) (e + 1));
            fwrite(blk.data(), 1, blk.size(), f);
        }
        fclose(f);
    }

    // offset math
    const llama_moe_expert_slab slab{ base, stride, n_expert };
    CHECK(llama_moe_expert_offset(slab, 0) == base);
    CHECK(llama_moe_expert_offset(slab, 3) == base + 3 * stride);

    FILE * fp = fopen(path.c_str(), "rb");
    CHECK(fp != nullptr);
    llama_moe_file_source src(fp);

    // budget holds exactly 2 experts
    llama_moe_stream_cache cache(2 * stride);

    auto check_expert = [&](uint32_t e) {
        const uint64_t  key = llama_moe_stream_key(0, LLAMA_MOE_PROJ_GATE, e);
        const uint8_t * p   = (const uint8_t *) cache.acquire(key, llama_moe_expert_offset(slab, e), stride, src);
        CHECK(p != nullptr);
        if (p) {
            bool ok = true;
            for (uint64_t i = 0; i < stride; ++i) {
                ok = ok && (p[i] == (uint8_t) (e + 1));
            }
            CHECK(ok);
        }
    };

    check_expert(0); // miss
    check_expert(1); // miss -> {1,0} resident
    check_expert(0); // hit  -> {0,1}
    CHECK(cache.get_stats().hits   == 1);
    CHECK(cache.get_stats().misses == 2);

    check_expert(2); // miss -> evict LRU (expert 1) -> {2,0}
    CHECK(cache.get_stats().evictions   == 1);
    CHECK(cache.resident_bytes()        == 2 * stride);

    check_expert(0); // still resident -> hit
    CHECK(cache.get_stats().hits == 2);

    check_expert(1); // evicted earlier -> miss
    CHECK(cache.get_stats().misses == 4);

    fclose(fp);
    remove(path.c_str());

    // --- expert compaction ---
    {
        // n_used=2, n_tokens=2, ids row-major (pos = t*n_used + e): {2,0, 2,1}
        const int32_t ids[] = { 2, 0, 2, 1 };
        int32_t slot_of_pos[4] = {};
        int32_t expert_of_slot[4] = {};
        const int k = llama_moe_compaction(ids, 2, 2, 8, slot_of_pos, expert_of_slot);
        CHECK(k == 3);                       // unique experts: 2,0,1
        CHECK(expert_of_slot[0] == 2);       // first-appearance order
        CHECK(expert_of_slot[1] == 0);
        CHECK(expert_of_slot[2] == 1);
        CHECK(slot_of_pos[0] == 0);          // expert 2 -> slot 0
        CHECK(slot_of_pos[1] == 1);          // expert 0 -> slot 1
        CHECK(slot_of_pos[2] == 0);          // expert 2 again -> slot 0
        CHECK(slot_of_pos[3] == 2);          // expert 1 -> slot 2
    }
    {
        // all-same expert collapses to a single slot
        const int32_t ids[] = { 5, 5, 5 };
        int32_t slot_of_pos[3] = {};
        int32_t expert_of_slot[3] = {};
        const int k = llama_moe_compaction(ids, 3, 1, 8, slot_of_pos, expert_of_slot);
        CHECK(k == 1);
        CHECK(expert_of_slot[0] == 5);
        CHECK(slot_of_pos[0] == 0 && slot_of_pos[1] == 0 && slot_of_pos[2] == 0);
    }

    // --- threshold remap planning (llama_moe_plan_remap) ---
    {
        const int n_expert = 8;
        // freq: experts 1,3,4,7 selected; 4 is hottest (3x), then 1 (2x), then 3,7 (1x)
        int freq[8] = { 0, 2, 0, 1, 3, 0, 0, 1 };

        // residency: experts 1 and 4 resident (slots 0,2); 3 and 7 are misses => 2/4 = 0.5 miss
        int32_t expert_slot[8] = { -1, 0, -1, -1, 2, -1, -1, -1 };
        int n_unique = 0, n_miss = 0, n_ranked = 0;
        int32_t ranked[8] = {};

        // threshold 0.5: miss fraction (0.5) is NOT > 0.5 => no sync
        bool sync = llama_moe_plan_remap(freq, n_expert, expert_slot, 0.5f,
                                         &n_unique, &n_miss, ranked, &n_ranked);
        CHECK(n_unique == 4);
        CHECK(n_miss == 2);
        CHECK(n_ranked == 4);
        CHECK(sync == false);
        // ranked by descending freq, ties by ascending id: 4(3),1(2),3(1),7(1)
        CHECK(ranked[0] == 4);
        CHECK(ranked[1] == 1);
        CHECK(ranked[2] == 3);
        CHECK(ranked[3] == 7);

        // threshold 0.4: 0.5 > 0.4 => sync
        sync = llama_moe_plan_remap(freq, n_expert, expert_slot, 0.4f,
                                    &n_unique, &n_miss, ranked, &n_ranked);
        CHECK(sync == true);

        // cold cache: nothing resident => 4/4 = 1.0 miss => sync at any threshold < 1
        int32_t cold[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
        sync = llama_moe_plan_remap(freq, n_expert, cold, 0.5f,
                                    &n_unique, &n_miss, ranked, &n_ranked);
        CHECK(n_miss == 4 && n_unique == 4);
        CHECK(sync == true);

        // fully resident => 0 misses => never sync
        int32_t hot[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        sync = llama_moe_plan_remap(freq, n_expert, hot, 0.0f,
                                    &n_unique, &n_miss, ranked, &n_ranked);
        CHECK(n_miss == 0);
        CHECK(sync == false); // n_miss (0) > 0.0*n_unique (0) is false

        // no experts selected => no sync, no unique
        int zero_freq[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        sync = llama_moe_plan_remap(zero_freq, n_expert, cold, 0.5f,
                                    &n_unique, &n_miss, ranked, &n_ranked);
        CHECK(n_unique == 0);
        CHECK(sync == false);
    }


    if (g_fail == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d CHECK(s) FAILED\n", g_fail);
    return 1;
}
