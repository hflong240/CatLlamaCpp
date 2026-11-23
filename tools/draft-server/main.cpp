#include "arg.h"
#include "common.h"
#include "http.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cstdlib>
#include <cstring>
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

    const llama_vocab * vocab = nullptr;

    int32_t n_vocab   = 0;
    int32_t n_batch   = 0;
    int32_t n_ctx_seq = 0;
    int32_t n_seq_max = 0;

    llama_batch batch = {};

    struct seq_state {
        std::vector<llama_token> tokens;
    };

    std::vector<seq_state> states;
    std::vector<common_sampler_ptr> smpls;

    std::mutex mtx;

    draft_server(llama_context * ctx, const llama_model * model) : ctx(ctx) {
        vocab     = llama_model_get_vocab(model);
        n_vocab   = (int32_t) llama_vocab_n_tokens(vocab);
        n_batch   = (int32_t) llama_n_batch(ctx);
        n_ctx_seq = (int32_t) llama_n_ctx_seq(ctx);
        n_seq_max = (int32_t) llama_n_seq_max(ctx);

        batch = llama_batch_init(n_batch, 0, 1);

        states.resize(n_seq_max);

        for (int32_t i = 0; i < n_seq_max; ++i) {
            common_params_sampling sparams;
            sparams.no_perf  = true;
            sparams.top_k    = 10;
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

        if ((int32_t) tokens.size() > n_ctx_seq) {
            return make_error("prompt exceeds the context size");
        }

        auto & st = states[seq_id];

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

    json handle_draft(const json & req) {
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

    svr.Get("/props", [&](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(srv.mtx);
        res.set_content(srv.handle_props().dump(), "application/json");
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
        res.set_content(srv.handle_reset(j).dump(), "application/json");
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
        res.set_content(srv.handle_draft(j).dump(), "application/json");
    });

    const auto info = common_speculative_get_vocab_info(llama_model_get_vocab(model));

    LOG_INF("%s: listening on %s:%d\n", __func__, params.hostname.c_str(), params.port);
    LOG_INF("%s: n_ctx = %u, n_ctx_seq = %d, n_seq_max = %d\n",
            __func__, llama_n_ctx(ctx), srv.n_ctx_seq, srv.n_seq_max);
    LOG_INF("%s: vocab n_vocab = %d, type = %d, hash = %s\n",
            __func__, info.n_vocab, info.vocab_type, common_speculative_hash_to_hex(info.token_hash).c_str());

    if (!svr.listen(params.hostname, params.port)) {
        LOG_ERR("%s: failed to bind to %s:%d\n", __func__, params.hostname.c_str(), params.port);
        return 1;
    }

    llama_backend_free();

    return 0;
}
