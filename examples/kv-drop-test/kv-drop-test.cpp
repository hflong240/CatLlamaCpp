#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

// fork: KV-drop quality probe. Verifies that deleting a MIDDLE range of the KV cache
// (seq_rm + seq_add compaction, no re-prefill) preserves a needle placed in the retained HEAD.
// A/B in one process: A = answer with full KV; B = answer after dropping MIDDLE. Same model load,
// greedy sampling, so any answer difference is due to the KV edit alone.

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n    %s -m model.gguf -c 4096 --moe-stream-async -fit off [--junk 200]\n\n", argv[0]);
}

// decode one token at position pos into seq 0, return nothing (logits requested)
static bool decode_one(llama_context * ctx, llama_batch & batch, llama_token tok, int pos) {
    common_batch_clear(batch);
    common_batch_add(batch, tok, pos, { 0 }, true);
    return llama_decode(ctx, batch) == 0;
}

// greedy-generate up to n_gen tokens starting after last decoded logits; returns the text
static std::string generate(llama_context * ctx, llama_sampler * smpl, llama_batch & batch,
                            const llama_vocab * vocab, int & n_past, int n_gen) {
    std::string out;
    for (int i = 0; i < n_gen; i++) {
        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) { break; }
        out += common_token_to_piece(ctx, id);
        if (!decode_one(ctx, batch, id, n_past++)) { break; }
    }
    return out;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_junk = 200;
    params.n_ctx  = 4096;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PASSKEY, print_usage)) {
        return 1;
    }

    const std::string needle = "BLUEHERON";
    const std::string head = "IMPORTANT: The secret project codename is " + needle +
        ". Remember this codename. When asked for the codename, answer with exactly " + needle + " and nothing else.\n\n";
    std::string middle;
    for (int i = 0; i < params.n_junk; i++) {
        middle += "Filler line " + std::to_string(i) +
            ": the weather is mild, the cat sat on the mat, and time passed slowly.\n";
    }
    const std::string q = "\nQuestion: What is the secret project codename? Answer with only the codename.\nAnswer:";

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) { LOG_ERR("%s: load failed\n", __func__); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { LOG_ERR("%s: ctx failed\n", __func__); return 1; }

    auto * mem = llama_get_memory(ctx);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // tokenize each segment separately so we know exact position boundaries.
    // add_special (BOS) only on the head; parse_special=true so any control tokens render.
    std::vector<llama_token> t_head = common_tokenize(ctx, head,   true,  true);
    std::vector<llama_token> t_mid  = common_tokenize(ctx, middle, false, false);
    std::vector<llama_token> t_q    = common_tokenize(ctx, q,      false, true);

    const int h = (int) t_head.size();               // [0, h)   HEAD (kept, holds needle)
    const int m = h + (int) t_mid.size();            // [h, m)   MIDDLE (to be dropped)
    LOG_INF("%s: HEAD=%d MIDDLE=%d Q=%d tokens; will drop MIDDLE [%d,%d)\n",
            __func__, h, (int) t_mid.size(), (int) t_q.size(), h, m);

    const int n_gen = 16;
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);

    // Path B verification: split across processes to model "big model exits, then restarts".
    //   KVDROP_SAVE=file : prefill HEAD+MIDDLE, save KV to file, exit (big model saves + exits).
    //   KVDROP_LOAD=file : skip prefill, load KV from file, then prune MIDDLE + answer (big model restarts).
    // Neither set: original single-process A/B (live-context prune).
    const char * kv_save = getenv("KVDROP_SAVE");
    const char * kv_load = getenv("KVDROP_LOAD");

    int n_past = m;
    if (kv_load) {
        // ---- big-model-restart path: load full saved KV (no prefill), then prune ----
        size_t n_tok_out = 0;
        std::vector<llama_token> dummy(m + 8);
        const size_t nread = llama_state_seq_load_file(ctx, kv_load, 0, dummy.data(), dummy.size(), &n_tok_out);
        if (nread == 0) { LOG_ERR("%s: KV load failed from %s\n", __func__, kv_load); return 1; }
        LOG_INF("%s: loaded KV from %s, %zu tokens, %zu bytes (NO prefill)\n", __func__, kv_load, n_tok_out, nread);
        n_past = (int) (llama_memory_seq_pos_max(mem, 0) + 1);
    } else {
        // ---- prefill HEAD+MIDDLE into seq 0 (batched, like passkey) ----
        std::vector<llama_token> pre = t_head;
        pre.insert(pre.end(), t_mid.begin(), t_mid.end());
        for (int i = 0; i < (int) pre.size(); i += params.n_batch) {
            common_batch_clear(batch);
            const int n = std::min((int) params.n_batch, (int) pre.size() - i);
            for (int j = 0; j < n; j++) {
                common_batch_add(batch, pre[i + j], i + j, { 0 }, false);
            }
            if (llama_decode(ctx, batch) != 0) { LOG_ERR("%s: prefill decode failed\n", __func__); return 1; }
        }
        n_past = m; // positions [0,m) filled

        if (kv_save) {
            const size_t nw = llama_state_seq_save_file(ctx, kv_save, 0, pre.data(), pre.size());
            LOG_INF("%s: saved KV to %s, %zu tokens, %zu bytes; exiting (big model would exit here)\n",
                    __func__, kv_save, pre.size(), nw);
            llama_sampler_free(smpl); llama_batch_free(batch);
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 0;
        }
    }

    // ================= A: baseline, full KV (skipped on the load path) =================
    std::string ansA = "(skipped: load path)";
    if (!kv_load) {
    for (int j = 0; j < (int) t_q.size(); j++) {
        const bool last = (j == (int) t_q.size() - 1);
        common_batch_clear(batch);
        common_batch_add(batch, t_q[j], n_past++, { 0 }, last);
        if (llama_decode(ctx, batch) != 0) { LOG_ERR("%s: A q decode failed\n", __func__); return 1; }
    }
    ansA = generate(ctx, smpl, batch, vocab, n_past, n_gen);

    // ---- roll back Q + A's generated tokens: delete KV from position m onward ----
    llama_memory_seq_rm(mem, 0, m, -1);
    }

    // ================= drop MIDDLE, compact (skippable via KVDROP_NOPRUNE for a load-only control) =================
    const bool no_prune = getenv("KVDROP_NOPRUNE") != nullptr;
    LOG_INF("%s: BEFORE prune: seq_pos_min=%d seq_pos_max=%d (h=%d m=%d)\n", __func__,
            (int) llama_memory_seq_pos_min(mem, 0), (int) llama_memory_seq_pos_max(mem, 0), h, m);
    int n_after_drop;
    if (no_prune) {
        n_after_drop = (int) (llama_memory_seq_pos_max(mem, 0) + 1); // == m: full KV kept, just re-answer
    } else {
        llama_memory_seq_rm (mem, 0, h, m);          // remove MIDDLE KV
        llama_memory_seq_add(mem, 0, m, -1, -(m - h)); // shift any tail down (tail is empty here; harmless)
        n_after_drop = (int) (llama_memory_seq_pos_max(mem, 0) + 1); // == h if only HEAD remains
    }
    LOG_INF("%s: AFTER prune: seq_pos_min=%d seq_pos_max=%d n_after_drop=%d (expect max=%d)\n", __func__,
            (int) llama_memory_seq_pos_min(mem, 0), (int) llama_memory_seq_pos_max(mem, 0), n_after_drop, h - 1);
    int n_past_b = n_after_drop;

    // ================= B: after KV drop =================
    for (int j = 0; j < (int) t_q.size(); j++) {
        const bool last = (j == (int) t_q.size() - 1);
        common_batch_clear(batch);
        common_batch_add(batch, t_q[j], n_past_b++, { 0 }, last);
        if (llama_decode(ctx, batch) != 0) { LOG_ERR("%s: B q decode failed\n", __func__); return 1; }
    }
    std::string ansB = generate(ctx, smpl, batch, vocab, n_past_b, n_gen);

    auto has_needle = [&](const std::string & s) {
        // case-insensitive contains: the model may echo the needle with different casing
        std::string a = s, b = needle;
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
        std::transform(b.begin(), b.end(), b.begin(), ::tolower);
        return a.find(b) != std::string::npos;
    };

    LOG("\n\n==================== KV-DROP QUALITY RESULT ====================\n");
    LOG("mode            : %s\n", kv_load ? "LOAD (big-model-restart: load KV file, prune, answer)" : "LIVE (single-process A/B)");
    LOG("needle          : %s\n", needle.c_str());
    LOG("dropped MIDDLE  : %d tokens (positions [%d,%d))\n", m - h, h, m);
    LOG("n_past after drop: %d (expected %d = HEAD only)\n", n_after_drop, h);
    LOG("A (full KV)     : \"%s\"  -> needle %s\n", ansA.c_str(), (!kv_load && has_needle(ansA)) ? "FOUND" : (kv_load ? "N/A" : "MISSING"));
    LOG("B (KV dropped)  : \"%s\"  -> needle %s\n", ansB.c_str(), has_needle(ansB) ? "FOUND" : "MISSING");
    if (kv_load) {
        LOG("VERDICT: %s\n", has_needle(ansB) ? "PASS (needle survived save->exit->load->prune, ZERO re-prefill)"
                                              : "FAIL (needle lost through the save/load/prune chain)");
    } else {
        LOG("VERDICT: %s\n", (has_needle(ansA) && has_needle(ansB)) ? "PASS (needle survived KV drop)" :
                              (has_needle(ansA) && !has_needle(ansB)) ? "FAIL (KV drop lost the needle)" :
                              "INCONCLUSIVE (baseline A already missed needle)");
    }
    LOG("================================================================\n\n");

    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
