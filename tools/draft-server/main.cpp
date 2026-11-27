#include "arg.h"
#include "base64.hpp"
#include "common.h"
#include "http.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

static void print_usage(int /*argc*/, char ** argv) {
    LOG("\nusage: %s -m MODEL [options]\n", argv[0]);
    LOG("\n");
    LOG("Serves speculative draft tokens over HTTP for llama-server running with\n");
    LOG("--spec-draft-remote HOST:PORT. The draft model runs on this machine only.\n");
    LOG("\n");
}

struct draft_server {
    llama_context * ctx = nullptr;
    llama_model * model = nullptr;

    const llama_vocab * vocab = nullptr;

    int32_t n_vocab   = 0;
    int32_t n_batch   = 0;
    int32_t n_ctx_seq = 0;
    int32_t n_seq_max = 0;

    // dflash / dspark sidecar
    bool is_dspark = false;

    int32_t n_embd_draft = 0;
    int32_t feat_dim     = 0;
    int32_t block_size   = 0;
    int32_t n_swa        = 0;
    llama_token mask_token_id = 0;
    bool sample_from_anchor = true;
    bool has_conf = false;

    std::vector<int32_t> target_layers;

    // external target weights
    std::map<std::string, size_t> w_received;
    std::map<std::string, size_t> w_total;
    bool weights_ready = false;

    llama_batch batch = {};

    struct seq_state {
        std::vector<llama_token> tokens;

        // dspark maps target positions into a local, contiguous KV range so that
        // features can always be appended to the draft cache
        llama_pos pos_off     = 0;
        bool      pos_off_set = false;
    };

    std::vector<seq_state> states;
    std::vector<common_sampler_ptr> smpls;

    std::mutex mtx;

    draft_server(llama_context * ctx, llama_model * model) : ctx(ctx), model(model) {
        vocab     = llama_model_get_vocab(model);
        n_vocab   = (int32_t) llama_vocab_n_tokens(vocab);
        n_batch   = (int32_t) llama_n_batch(ctx);
        n_ctx_seq = (int32_t) llama_n_ctx_seq(ctx);
        n_seq_max = (int32_t) llama_n_seq_max(ctx);

        batch = llama_batch_init(n_batch, 0, 1);

        states.resize(n_seq_max);

        block_size = (int32_t) llama_model_dflash_block_size(model);
        const uint32_t n_tl = llama_model_target_layer_ids_n(model);
        if (n_tl > 0 && block_size > 0) {
            is_dspark = true;
            n_embd_draft = llama_model_n_embd(model);
            feat_dim     = (int32_t) llama_model_n_embd_inp_enc(model);
            n_swa        = llama_model_n_swa(model);
            mask_token_id = llama_vocab_mask(vocab);

            const int32_t * tl = llama_model_target_layer_ids(model);
            target_layers.assign(tl, tl + n_tl);

            char buf[32] = {};
            if (llama_model_meta_val_str(model, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model, "dflash.has_confidence_head", buf, sizeof(buf)) >= 0) {
                has_conf = std::strcmp(buf, "true") == 0;
            }

            llama_set_embeddings_nextn(ctx, true, true);
            llama_set_causal_attn(ctx, false); // block diffusion attends within the block

            if (llama_model_has_weight(model, "tok_embd") && llama_model_has_weight(model, "output")) {
                weights_ready = true;
            }
        }

        for (int32_t i = 0; i < n_seq_max; ++i) {
            common_params_sampling sparams;
            sparams.no_perf  = true;
            sparams.top_k    = is_dspark ? 1 : 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };

            smpls.emplace_back(common_sampler_init(model, sparams));
        }
    }

    ~draft_server() {
        llama_batch_free(batch);
    }

    static json make_error(const std::string & msg) {
        json j;
        j["ok"]    = false;
        j["error"] = msg;
        return j;
    }

    static json make_need_reset() {
        json j;
        j["ok"]         = false;
        j["need_reset"] = true;
        return j;
    }

    bool decode_tokens(llama_seq_id seq_id, const llama_token * tokens, size_t n, llama_pos pos0, bool logits) {
        common_batch_clear(batch);

        for (size_t i = 0; i < n; ++i) {
            common_batch_add(batch, tokens[i], pos0 + (llama_pos) i, { seq_id }, logits && (i + 1 == n));
        }

        return llama_decode(ctx, batch) == 0;
    }

    bool decode_one(llama_seq_id seq_id, llama_token token, llama_pos pos) {
        return decode_tokens(seq_id, &token, 1, pos, true);
    }

    json handle_props() const {
        const auto info = common_speculative_get_vocab_info(vocab);

        json j;
        j["v"]          = 1;
        j["n_vocab"]    = info.n_vocab;
        j["vocab_type"] = info.vocab_type;
        j["add_bos"]    = info.add_bos;
        j["bos"]        = info.bos;
        j["add_eos"]    = info.add_eos;
        j["eos"]        = info.eos;
        j["token_hash"] = common_speculative_hash_to_hex(info.token_hash);
        j["n_ctx"]      = (int32_t) llama_n_ctx(ctx);
        j["n_ctx_seq"]  = n_ctx_seq;
        j["n_seq_max"]  = n_seq_max;

        if (is_dspark) {
            j["dspark"] = true;
            j["dflash"] = {
                {"block_size",         block_size},
                {"target_layers",      target_layers},
                {"n_embd",             n_embd_draft},
                {"n_embd_inp_enc",     feat_dim},
                {"n_swa",              n_swa},
                {"mask_token_id",      mask_token_id},
                {"sample_from_anchor", sample_from_anchor},
                {"has_confidence",     has_conf},
            };
            j["weights"] = {
                {"tok_embd", llama_model_ext_weight_ready(model, "tok_embd") || llama_model_has_weight(model, "tok_embd")},
                {"output",   llama_model_ext_weight_ready(model, "output")   || llama_model_has_weight(model, "output")},
            };
            j["weights_ready"] = weights_ready;
        }

        return j;
    }

    json handle_weights(const std::string & name, int32_t type, int64_t ne0, int64_t ne1,
            size_t offset, size_t total, const std::string & body) {
        if (!is_dspark) {
            return make_error("this draft model does not take external weights");
        }
        if (name != "tok_embd" && name != "output") {
            return make_error("unknown weight");
        }
        if (body.empty()) {
            return make_error("empty chunk");
        }
        if (ne0 != n_embd_draft || ne1 != n_vocab) {
            return make_error("weight shape does not match the draft model");
        }

        if (offset == 0) {
            if (!llama_model_ext_weight_alloc(model, name.c_str(), type, ne0, ne1)) {
                return make_error("failed to allocate the weight");
            }
            w_received[name] = 0;
            w_total[name]    = total;
        }

        if (!llama_model_ext_weight_set(model, name.c_str(), offset, body.data(), body.size())) {
            return make_error("failed to store the weight chunk");
        }

        w_received[name] = std::max(w_received[name], offset + body.size());
        LOG_INF("%s: %s %.1f/%.1f MiB\n", __func__, name.c_str(),
                w_received[name] / 1024.0 / 1024.0, total / 1024.0 / 1024.0);

        const bool ok_tok = w_total.count("tok_embd") && w_received["tok_embd"] >= w_total["tok_embd"];
        const bool ok_out = w_total.count("output")   && w_received["output"]   >= w_total["output"];
        if (ok_tok && ok_out && !weights_ready) {
            weights_ready = true;
            llama_context_require_reserve(ctx);
            LOG_INF("%s: external target weights are ready\n", __func__);
        }

        json j;
        j["ok"] = true;
        return j;
    }

    json handle_reset(const json & req) {
        const int32_t seq_id = req.value("seq", -1);

        if (seq_id < 0 || seq_id >= n_seq_max) {
            return make_error("invalid seq");
        }
        if (!req.contains("tokens") || !req["tokens"].is_array()) {
            return make_error("missing tokens");
        }

        std::vector<llama_token> tokens;
        tokens.reserve(req["tokens"].size());

        for (const auto & t : req["tokens"]) {
            const int32_t id = t.get<int32_t>();
            if (id < 0 || id >= n_vocab) {
                return make_error("invalid token");
            }
            tokens.push_back(id);
        }

        if (!is_dspark && (int32_t) tokens.size() > n_ctx_seq) {
            return make_error("prompt exceeds the context size");
        }

        auto & st = states[seq_id];

        if (is_dspark) {
            // the draft has no token embeddings: the KV ring is seeded by the
            // target features that arrive with the first /draft request
            if (!llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1)) {
                return make_error("failed to clear the sequence");
            }
            st.tokens      = std::move(tokens);
            st.pos_off     = 0;
            st.pos_off_set = false;

            LOG_DBG("%s: dspark seq %d reset, n_tokens = %zu\n", __func__, seq_id, st.tokens.size());

            json j;
            j["ok"] = true;
            return j;
        }

        size_t n_common = 0;
        while (n_common < tokens.size() && n_common < st.tokens.size() && tokens[n_common] == st.tokens[n_common]) {
            ++n_common;
        }

        if (n_common < st.tokens.size()) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), seq_id, (llama_pos) n_common, -1)) {
                // limited memory modules may reject partial removal - replay the whole prompt
                if (!llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1)) {
                    return make_error("failed to clear the sequence");
                }
                st.tokens.clear();
                n_common = 0;
            }
        }

        for (size_t i = n_common; i < tokens.size(); ) {
            const size_t n = std::min((size_t) n_batch, tokens.size() - i);
            if (!decode_tokens(seq_id, tokens.data() + i, n, (llama_pos) i, false)) {
                // keep the stored state consistent with the KV after a failed decode
                llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
                st.tokens.clear();
                return make_error("failed to decode the prompt");
            }
            i += n;
        }

        st.tokens = std::move(tokens);

        LOG_DBG("%s: seq %d reset, n_tokens = %zu\n", __func__, seq_id, st.tokens.size());

        json j;
        j["ok"] = true;
        return j;
    }

    bool inject_features(llama_seq_id seq_id, const std::vector<llama_pos> & pos, const std::vector<float> & feat) {
        const size_t n_rows = pos.size();
        for (size_t beg = 0; beg < n_rows; beg += (size_t) n_batch) {
            const int32_t n = (int32_t) std::min((size_t) n_batch, n_rows - beg);

            llama_batch b = llama_batch_init(n, feat_dim, 1);
            b.n_tokens = n;

            std::memcpy(b.embd, feat.data() + beg * (size_t) feat_dim, (size_t) n * feat_dim * sizeof(float));

            for (int32_t i = 0; i < n; ++i) {
                b.pos[i]       = pos[beg + i];
                b.n_seq_id[i]  = 1;
                b.seq_id[i][0] = seq_id;
                b.logits[i]    = false;
            }

            const int rc = llama_decode(ctx, b);
            llama_batch_free(b);

            if (rc != 0) {
                LOG_ERR("%s: injection decode failed rc=%d\n", __func__, rc);
                return false;
            }
        }

        return true;
    }

    json handle_draft_dspark(const json & req) {
        const auto t_gen0 = std::chrono::steady_clock::now();
        const int32_t seq_id  = req.value("seq", -1);
        const int32_t n_past  = req.value("n_past", -1);
        const int32_t id_last = req.value("id_last", -1);
        const int32_t n_max   = req.value("n_max", 0);
        const int32_t n_min   = req.value("n_min", 0);
        const float   p_min   = req.value("p_min", 0.0f);

        const std::string prefix_hash = req.value("prefix_hash", std::string());

        if (seq_id < 0 || seq_id >= n_seq_max) {
            return make_error("invalid seq");
        }
        if (n_past < 0) {
            return make_error("invalid n_past");
        }
        if (id_last < 0 || id_last >= n_vocab) {
            return make_error("invalid id_last");
        }
        if (n_max < 0 || n_max > 1024) {
            return make_error("invalid n_max");
        }
        if (!weights_ready) {
            return make_error("target weights are not available yet");
        }
        if (!sample_from_anchor) {
            return make_error("this build only supports anchor-first dspark drafts");
        }

        auto & st = states[seq_id];

        if ((int32_t) st.tokens.size() < n_past) {
            return make_need_reset();
        }

        if ((int32_t) st.tokens.size() > n_past) {
            st.tokens.resize(n_past);
        }

        const uint64_t hash_remote = std::strtoull(prefix_hash.c_str(), nullptr, 16);
        const uint64_t hash_local  = common_speculative_tokens_hash(st.tokens.data(), st.tokens.size(), st.tokens.size());

        if (prefix_hash.empty() || hash_remote != hash_local) {
            return make_need_reset();
        }

        // target features for the committed prefix
        std::vector<llama_pos> feat_pos;
        if (req.contains("feat_pos")) {
            for (const auto & p : req["feat_pos"]) {
                feat_pos.push_back(p.get<llama_pos>());
            }
        }

        std::vector<float> feat;
        if (req.contains("feat") && req["feat"].is_string()) {
            const std::string raw = base64::decode(req["feat"].get<std::string>());
            if (raw.size() % sizeof(float) != 0) {
                return make_error("invalid feature payload");
            }
            feat.resize(raw.size() / sizeof(float));
            std::memcpy(feat.data(), raw.data(), raw.size());

            if (feat.size() != feat_pos.size() * (size_t) feat_dim) {
                return make_error("feature payload does not match feat_pos");
            }
        }

        llama_memory_t mem = llama_get_memory(ctx);

        for (size_t i = 0; i < feat_pos.size(); ++i) {
            if (feat_pos[i] < 0 || feat_pos[i] >= n_past) {
                return make_error("invalid feature position");
            }
            if (i > 0 && feat_pos[i] != feat_pos[0] + (llama_pos) i) {
                return make_error("feature positions are not contiguous");
            }
        }

        if (!st.pos_off_set) {
            st.pos_off     = feat_pos.empty() ? n_past : feat_pos[0];
            st.pos_off_set = true;
            if (!llama_memory_seq_rm(mem, seq_id, -1, -1)) {
                return make_error("failed to clear the sequence");
            }
        }

        llama_pos cache_max = llama_memory_seq_pos_max(mem, seq_id);
        llama_pos block_pos = n_past - st.pos_off;

        if (!feat_pos.empty()) {
            llama_pos rel_first = feat_pos[0] - st.pos_off;

            if (rel_first < 0 || rel_first > cache_max + 1) {
                // a gap or a new local range: start from the first feature
                if (!llama_memory_seq_rm(mem, seq_id, -1, -1)) {
                    return make_error("failed to clear the sequence");
                }
                st.pos_off = feat_pos[0];
                rel_first  = 0;
                cache_max  = -1;
                block_pos  = n_past - st.pos_off;
            } else if (rel_first <= cache_max) {
                // overwrite the draft K of the accepted tokens
                if (!llama_memory_seq_rm(mem, seq_id, rel_first, -1)) {
                    return make_error("failed to remove the draft cache tail");
                }
                cache_max = rel_first - 1;
            }

            std::vector<llama_pos> rel_pos;
            rel_pos.reserve(feat_pos.size());
            for (size_t i = 0; i < feat_pos.size(); ++i) {
                rel_pos.push_back(rel_first + (llama_pos) i);
            }

            if (!inject_features(seq_id, rel_pos, feat)) {
                return make_error("failed to inject target features");
            }

            cache_max = rel_pos.back();
        }

        if (block_pos < 0) {
            return make_error("invalid draft position");
        }
        if (block_pos <= cache_max) {
            // a rejected draft: drop the draft cache rows past the new prefix
            if (!llama_memory_seq_rm(mem, seq_id, block_pos, -1)) {
                return make_error("failed to remove the draft cache tail");
            }
            cache_max = block_pos - 1;
        } else if (block_pos != cache_max + 1) {
            // the features do not reach the new prefix
            if (!llama_memory_seq_rm(mem, seq_id, -1, -1)) {
                return make_error("failed to clear the sequence");
            }
            st.pos_off = n_past;
            block_pos  = 0;
        }

        const int32_t n_draft = std::min(n_max, block_size);
        if (n_draft <= 0) {
            return make_error("invalid draft length");
        }

        // Decode the whole block even when the caller needs fewer columns. The extra drafts are the
        // candidates for the next anchor; the client compares them against its sampled token to decide
        // whether an ahead run is worth it. They cost one wider forward, not another decode.
        const int32_t n_pred  = std::max(n_draft, block_size);
        // block layout: [anchor, mask...]; anchor-first DSpark yields block_size drafts
        const int32_t n_block = n_pred + (sample_from_anchor ? 0 : 1);
        if (block_pos + n_block >= n_ctx_seq) {
            return make_error("draft exceeds the context size");
        }

        common_batch_clear(batch);

        for (int32_t i = 0; i < n_block; ++i) {
            common_batch_add(batch, i == 0 ? id_last : mask_token_id, block_pos + i, { seq_id }, true);
        }

        if (llama_decode(ctx, batch) != 0) {
            return make_error("failed to decode the draft block");
        }

        common_sampler_reset(smpls[seq_id].get());

        const float * conf = has_conf ? llama_get_embeddings_nextn(ctx) : nullptr;

        std::vector<llama_token> out;
        std::vector<llama_token> out_all;
        out.reserve(n_draft);
        out_all.reserve(n_pred);

        for (int32_t i = 0; i < n_pred; ++i) {
            const int32_t row = i + (sample_from_anchor ? 0 : 1);

            common_sampler_sample(smpls[seq_id].get(), ctx, row, true);

            const llama_token_data_array * cur_p = common_sampler_get_candidates(smpls[seq_id].get(), true);
            if (cur_p == nullptr || cur_p->size == 0) {
                break;
            }

            const llama_token id = cur_p->data[0].id;
            out_all.push_back(id);

            if (i >= n_draft) {
                continue; // beyond the requested columns: diagnostic only
            }

            if (p_min > 0.0f) {
                if (conf != nullptr && conf[(size_t) row * n_embd_draft] < p_min) {
                    break;
                }
                if (conf == nullptr && cur_p->data[0].p < p_min) {
                    break;
                }
            }

            common_sampler_accept(smpls[seq_id].get(), id, true);
            out.push_back(id);
        }

        if ((int32_t) out.size() < std::max(0, std::min(n_min, n_max))) {
            out.clear();
        }

        st.tokens.push_back(id_last);
        for (const llama_token id : out) {
            st.tokens.push_back(id);
        }

        json j;
        j["ok"]     = true;
        j["tokens"] = out;
        j["all"]    = out_all;
        j["gen_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_gen0).count();
        return j;
    }

    json handle_draft(const json & req) {
        if (is_dspark) {
            return handle_draft_dspark(req);
        }

        const int32_t seq_id  = req.value("seq", -1);
        const int32_t n_past  = req.value("n_past", -1);
        const int32_t id_last = req.value("id_last", -1);
        const int32_t n_max   = req.value("n_max", 0);
        const int32_t n_min   = req.value("n_min", 0);
        const float   p_min   = req.value("p_min", 0.0f);

        const std::string prefix_hash = req.value("prefix_hash", std::string());

        if (seq_id < 0 || seq_id >= n_seq_max) {
            return make_error("invalid seq");
        }
        if (n_past < 0 || n_past >= n_ctx_seq) {
            return make_error("invalid n_past");
        }
        if (id_last < 0 || id_last >= n_vocab) {
            return make_error("invalid id_last");
        }
        if (n_max < 0 || n_max > 1024) {
            return make_error("invalid n_max");
        }

        auto & st = states[seq_id];

        if ((int32_t) st.tokens.size() < n_past) {
            return make_need_reset();
        }

        if ((int32_t) st.tokens.size() > n_past) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), seq_id, n_past, -1)) {
                return make_need_reset();
            }
            st.tokens.resize(n_past);
        }

        const uint64_t hash_remote = std::strtoull(prefix_hash.c_str(), nullptr, 16);
        const uint64_t hash_local  = common_speculative_tokens_hash(st.tokens.data(), st.tokens.size(), st.tokens.size());

        if (prefix_hash.empty() || hash_remote != hash_local) {
            return make_need_reset();
        }

        if (!decode_one(seq_id, id_last, n_past)) {
            return make_error("failed to decode the last token");
        }
        st.tokens.push_back(id_last);

        common_sampler_reset(smpls[seq_id].get());

        std::vector<llama_token> out;
        out.reserve(n_max);

        for (int32_t i = 0; i < n_max; ++i) {
            common_sampler_sample(smpls[seq_id].get(), ctx, 0, true);

            const llama_token_data_array * cur_p = common_sampler_get_candidates(smpls[seq_id].get(), true);
            if (cur_p == nullptr || cur_p->size == 0) {
                break;
            }

            const llama_token id = cur_p->data[0].id;

            if (cur_p->data[0].p < p_min) {
                break;
            }

            // leave room for the position of the drafted token
            if (n_past + (llama_pos) out.size() + 1 >= n_ctx_seq) {
                break;
            }

            common_sampler_accept(smpls[seq_id].get(), id, true);
            out.push_back(id);

            // logits are only needed when another token will be sampled
            const bool logits = (i + 1 < n_max);
            if (!decode_tokens(seq_id, &id, 1, n_past + (llama_pos) out.size(), logits)) {
                return make_error("failed to decode a drafted token");
            }
            st.tokens.push_back(id);
        }

        if ((int32_t) out.size() < std::max(0, std::min(n_min, n_max))) {
            out.clear();
        }

        json j;
        j["ok"]     = true;
        j["tokens"] = out;
        return j;
    }
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER, print_usage)) {
        return 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("%s: --model is required\n", __func__);
        return 1;
    }

    // dflash drafts get their target weights only after the target connects, so
    // the fitting pass cannot measure them
    {
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * gctx = gguf_init_from_file(params.model.path.c_str(), gp);
        if (gctx) {
            const int kid = gguf_find_key(gctx, "general.architecture");
            if (kid >= 0 && strcmp(gguf_get_val_str(gctx, kid), "dflash") == 0 && params.fit_params) {
                LOG_INF("%s: disabling -fit for a dflash draft\n", __func__);
                params.fit_params = false;
            }
            gguf_free(gctx);
        }
    }

    if (params.n_parallel < 0) {
        LOG_INF("%s: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n", __func__);
        params.n_parallel = 4;
        params.kv_unified = true;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    llama_context * ctx   = llama_init->context();
    llama_model   * model = llama_init->model();

    if (ctx == nullptr || model == nullptr) {
        LOG_ERR("%s: failed to load the draft model\n", __func__);
        return 1;
    }

    const auto rm_type = common_context_can_seq_rm(ctx);
    if (rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO || rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        LOG_ERR("%s: the draft context does not support partial sequence removal\n", __func__);
        return 1;
    }

    draft_server srv(ctx, model);

    httplib::Server svr;

    svr.set_payload_max_length(256 * 1024 * 1024);

    svr.Get("/props", [&](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(srv.mtx);
        res.set_content(srv.handle_props().dump(), "application/json");
    });

    svr.Post("/weights", [&](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(srv.mtx);

        const std::string name   = req.get_param_value("name");
        const int32_t     type   = std::atoi(req.get_param_value("type").c_str());
        const int64_t     ne0    = std::atoll(req.get_param_value("ne0").c_str());
        const int64_t     ne1    = std::atoll(req.get_param_value("ne1").c_str());
        const size_t      offset = (size_t) std::strtoull(req.get_param_value("offset").c_str(), nullptr, 10);
        const size_t      total  = (size_t) std::strtoull(req.get_param_value("total").c_str(),  nullptr, 10);

        try {
            res.set_content(srv.handle_weights(name, type, ne0, ne1, offset, total, req.body).dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(draft_server::make_error(e.what()).dump(), "application/json");
        }
    });

    svr.Post("/reset", [&](const httplib::Request & req, httplib::Response & res) {
        json j;
        try {
            j = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(draft_server::make_error(e.what()).dump(), "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(srv.mtx);
        try {
            res.set_content(srv.handle_reset(j).dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(draft_server::make_error(e.what()).dump(), "application/json");
        }
    });

    svr.Post("/draft", [&](const httplib::Request & req, httplib::Response & res) {
        json j;
        try {
            j = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(draft_server::make_error(e.what()).dump(), "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(srv.mtx);
        try {
            res.set_content(srv.handle_draft(j).dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(draft_server::make_error(e.what()).dump(), "application/json");
        }
    });

    const auto info = common_speculative_get_vocab_info(llama_model_get_vocab(model));

    LOG_INF("%s: listening on %s:%d\n", __func__, params.hostname.c_str(), params.port);
    LOG_INF("%s: n_ctx = %u, n_ctx_seq = %d, n_seq_max = %d\n",
            __func__, llama_n_ctx(ctx), srv.n_ctx_seq, srv.n_seq_max);
    LOG_INF("%s: vocab n_vocab = %d, type = %d, hash = %s\n",
            __func__, info.n_vocab, info.vocab_type, common_speculative_hash_to_hex(info.token_hash).c_str());

    if (srv.is_dspark) {
        LOG_INF("%s: dspark sidecar (block_size = %d, n_swa = %d, target_layers = %zu), waiting for the target weights\n",
                __func__, srv.block_size, srv.n_swa, srv.target_layers.size());
    }

    if (!svr.listen(params.hostname, params.port)) {
        LOG_ERR("%s: failed to bind to %s:%d\n", __func__, params.hostname.c_str(), params.port);
        return 1;
    }

    llama_backend_free();

    return 0;
}
