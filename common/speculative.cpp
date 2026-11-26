#include "speculative.h"

#include "base64.hpp"
#include "common.h"
#include "ggml.h"
#include "http.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <map>
#include <cinttypes>
#include <memory>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"draft-remote",  COMMON_SPECULATIVE_TYPE_DRAFT_REMOTE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

static constexpr uint64_t SPEC_HASH_OFFSET = 14695981039346656037ull;
static constexpr uint64_t SPEC_HASH_PRIME  = 1099511628211ull;

common_speculative_vocab_info common_speculative_get_vocab_info(const llama_vocab * vocab) {
    const int32_t n_vocab = (int32_t) llama_vocab_n_tokens(vocab);

    uint64_t hash = SPEC_HASH_OFFSET;
    for (int32_t i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < n_vocab; ++i) {
        const char * text = llama_vocab_get_text(vocab, i);
        const size_t len = text ? std::strlen(text) : 0;

        // mix the length so that token boundaries are unambiguous
        hash ^= (uint64_t) len;
        hash *= SPEC_HASH_PRIME;
        for (size_t j = 0; j < len; ++j) {
            hash ^= (uint8_t) text[j];
            hash *= SPEC_HASH_PRIME;
        }
    }

    return {
        /* .n_vocab    = */ n_vocab,
        /* .vocab_type = */ (int32_t) llama_vocab_type(vocab),
        /* .add_bos    = */ llama_vocab_get_add_bos(vocab),
        /* .bos        = */ llama_vocab_bos(vocab),
        /* .add_eos    = */ llama_vocab_get_add_eos(vocab),
        /* .eos        = */ llama_vocab_eos(vocab),
        /* .token_hash = */ hash,
    };
}

uint64_t common_speculative_tokens_hash(const llama_token * tokens, size_t n, size_t n_window) {
    uint64_t hash = SPEC_HASH_OFFSET;
    if (tokens == nullptr || n == 0 || n_window == 0) {
        return hash;
    }

    const size_t n_hash = std::min(n, n_window);
    const llama_token * beg = tokens + (n - n_hash);
    for (size_t i = 0; i < n_hash; ++i) {
        const uint32_t v = (uint32_t) beg[i];
        for (int b = 0; b < 4; ++b) {
            hash ^= (uint8_t) (v >> (8*b));
            hash *= SPEC_HASH_PRIME;
        }
    }

    return hash;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const = 0;

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_nextn() const { return false; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        LOG_INF("%s: adding speculative implementation 'draft-simple'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    //common_params_speculative_eagle3 params;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
    {
        LOG_INF("%s: adding speculative implementation 'draft-eagle3'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, params.draft.n_max, params.draft.n_min, params.draft.p_min);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & /*dparams*/) override {
        // TODO: implement
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    bool is_mem_shared = false;

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");

        LOG_INF("%s: adding speculative implementation 'draft-mtp'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    LOG_WRN("%s: backend offload failed for seq_id=%d; using CPU sampler\n", __func__, (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        last_n_drafted.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", __func__, (int) rc, (int) batch_in.pos[0]);
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const float * h_row = nullptr;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);

            h_row = pending_h[seq_id].data();
            std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                }
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_nextn() const override {
        return true;
    }
};

std::string common_speculative_hash_to_hex(uint64_t hash) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016" PRIx64, hash);
    return std::string(buf);
}

// draft model running on a remote llama-draft-server
//
// the remote side keeps its own KV cache and advances it from the `n_past` and
// `id_last` values sent with each request; the local side only receives tokens
struct common_speculative_impl_draft_remote : public common_speculative_impl {
    common_params_speculative_draft params;

    std::unique_ptr<httplib::Client> cli;

    bool disabled = false;
    bool warned_prompt_mismatch = false;
    int  n_fail = 0;

    struct seq_state {
        bool need_reset = true;
    };

    std::vector<seq_state> states;

    common_speculative_vocab_info vocab_tgt{};

    int32_t n_vocab = 0;

    // dflash / dspark remote sidecar
    bool dspark = false;
    bool weights_sent = false;
    bool remote_weights_ready = false;

    int32_t block_size = 0;
    int32_t n_swa      = 0;
    int32_t n_embd_tgt = 0;
    int32_t feat_dim   = 0;

    std::vector<int32_t> target_layers;

    struct feat_row {
        llama_pos pos = 0;
        std::vector<float> data;
        bool sent = false;
    };

    std::vector<std::deque<feat_row>> pending;

    // Draft-ahead diagnostic (dspark): the remote decodes the whole block, so its drafts past the
    // requested columns predict the next anchor. Count how often the target's sampled token actually
    // matches that prediction; that ratio is what decides whether an ahead run can skip a sync.
    struct ahead_prev {
        std::vector<llama_token> all;
        int  n_draft = 0;
        bool full    = false;
    };

    std::vector<ahead_prev> ahead;
    uint64_t ahead_n   = 0;
    uint64_t ahead_hit = 0;

    struct draft_result {
        bool ok         = false;
        bool need_reset = false;
        llama_tokens tokens;
    };

    common_speculative_impl_draft_remote(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_REMOTE, n_seq)
        , params(params.draft)
    {
        LOG_INF("%s: adding speculative implementation 'draft-remote'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f, endpoint=%s\n", __func__,
                this->params.n_max, this->params.n_min, this->params.p_min, this->params.remote_endpoint.c_str());

        states.resize(n_seq);
        ahead.resize(n_seq);

        if (this->params.remote_endpoint.empty()) {
            LOG_ERR("%s: no remote endpoint specified, disabling remote drafting\n", __func__);
            disabled = true;
            return;
        }

        if (this->params.ctx_tgt == nullptr) {
            LOG_ERR("%s: target context is not available, disabling remote drafting\n", __func__);
            disabled = true;
            return;
        }

        const llama_model * model_tgt = llama_get_model(this->params.ctx_tgt);
        vocab_tgt = common_speculative_get_vocab_info(llama_model_get_vocab(model_tgt));
        n_vocab = vocab_tgt.n_vocab;
        n_embd_tgt = llama_model_n_embd(model_tgt);

        pending.resize(n_seq);

        // connect early: dflash needs the target layer inputs of the prompt itself
        if (ensure_connected() && dspark) {
            ensure_weights();
        }
    }

    bool ensure_connected() {
        if (cli) {
            return true;
        }
        if (disabled) {
            return false;
        }

        std::string url = params.remote_endpoint;
        if (url.find("://") == std::string::npos) {
            url = "http://" + url;
        }

        try {
            auto pair = common_http_client(url);

            httplib::Client client = std::move(pair.first);
            client.set_connection_timeout(2, 0);
            client.set_read_timeout(10, 0);
            client.set_write_timeout(10, 0);

            auto res = client.Get("/props");
            if (!res) {
                LOG_ERR("%s: failed to reach remote draft server '%s': %s\n", __func__, url.c_str(),
                        httplib::to_string(res.error()).c_str());
                return false;
            }
            if (res->status != 200) {
                LOG_ERR("%s: remote draft server '%s' returned HTTP %d\n", __func__, url.c_str(), res->status);
                return false;
            }

            const nlohmann::json j = nlohmann::json::parse(res->body);

            const int32_t v = j.value("v", 0);
            if (v != 1) {
                LOG_ERR("%s: unsupported remote draft protocol version %d\n", __func__, v);
                return false;
            }

            const int32_t  n_vocab_remote    = j.value("n_vocab", -1);
            const int32_t  vocab_type_remote = j.value("vocab_type", -1);
            const uint64_t hash_remote       = std::strtoull(j.value("token_hash", std::string("0")).c_str(), nullptr, 16);

            const bool compatible =
                n_vocab_remote    == vocab_tgt.n_vocab &&
                vocab_type_remote == vocab_tgt.vocab_type &&
                j.value("add_bos", false) == vocab_tgt.add_bos &&
                j.value("bos", -1)        == (int) vocab_tgt.bos &&
                j.value("add_eos", false) == vocab_tgt.add_eos &&
                j.value("eos", -1)        == (int) vocab_tgt.eos &&
                hash_remote == vocab_tgt.token_hash;

            if (!compatible) {
                LOG_ERR("%s: the remote draft model vocab is incompatible with the target model\n", __func__);
                LOG_ERR("%s: target: n_vocab=%d type=%d hash=%016" PRIx64 "; remote: n_vocab=%d type=%d hash=%016" PRIx64 "\n",
                        __func__,
                        vocab_tgt.n_vocab, vocab_tgt.vocab_type, vocab_tgt.token_hash,
                        n_vocab_remote, vocab_type_remote, hash_remote);
                return false;
            }

            const int32_t n_seq_max = j.value("n_seq_max", 0);
            if (n_seq_max < (int32_t) n_seq) {
                LOG_ERR("%s: remote draft server supports %d sequences, but %u are required\n", __func__, n_seq_max, n_seq);
                return false;
            }

            const int32_t  n_ctx_seq     = j.value("n_ctx_seq", 0);
            const uint32_t n_ctx_seq_tgt = llama_n_ctx_seq(params.ctx_tgt);

            dspark = j.value("dspark", false);
            remote_weights_ready = j.value("weights_ready", false);
            // a restarted draft server has no weights: upload again on reconnect
            weights_sent = remote_weights_ready;
            if (dspark) {
                char arch[64] = {};
                if (llama_model_meta_val_str(llama_get_model(params.ctx_tgt), "general.architecture", arch, sizeof(arch)) < 0 ||
                        strcmp(arch, "deepseek41") != 0) {
                    LOG_ERR("%s: dspark remote drafting requires a deepseek41 target, got '%s'\n", __func__, arch);
                    dspark = false;
                    return false;
                }

                const auto & df = j.at("dflash");
                block_size = df.value("block_size", 0);
                n_swa      = df.value("n_swa", 0);
                feat_dim   = df.value("n_embd_inp_enc", 0);

                target_layers.clear();
                for (const auto & lid : df.at("target_layers")) {
                    target_layers.push_back(lid.get<int32_t>());
                }

                if (block_size <= 0 || target_layers.empty() || feat_dim <= 0) {
                    LOG_ERR("%s: invalid dspark metadata from the remote draft server\n", __func__);
                    return false;
                }
                if ((int32_t) target_layers.size() * n_embd_tgt != feat_dim) {
                    LOG_ERR("%s: target hidden size %d x %zu layers does not match the draft feature size %d\n",
                            __func__, n_embd_tgt, target_layers.size(), feat_dim);
                    return false;
                }

                for (const int32_t lid : target_layers) {
                    if (lid < 0 || lid >= llama_model_n_layer(llama_get_model(params.ctx_tgt))) {
                        LOG_ERR("%s: dspark target layer %d is out of range\n", __func__, lid);
                        return false;
                    }
                    llama_set_embeddings_layer_inp(params.ctx_tgt, (uint32_t) lid, true);
                }

                std::string layers;
                for (size_t i = 0; i < target_layers.size(); ++i) {
                    layers += (i ? ", " : "") + std::to_string(target_layers[i]);
                }
                LOG_INF("%s: remote draft is dspark (block_size=%d, layers=[%s], n_swa=%d)\n",
                        __func__, block_size, layers.c_str(), n_swa);

                // feature payloads for a full window can be several MB
                client.set_read_timeout(60, 0);
            } else if (n_ctx_seq > 0 && n_ctx_seq < (int32_t) n_ctx_seq_tgt) {
                LOG_WRN("%s: remote n_ctx_seq=%d is smaller than target n_ctx_seq=%u - long prompts will not draft\n",
                        __func__, n_ctx_seq, n_ctx_seq_tgt);
            }

            cli = std::make_unique<httplib::Client>(std::move(client));

            LOG_INF("%s: connected to remote draft server '%s' (n_ctx=%d, n_ctx_seq=%d, n_seq_max=%d)\n", __func__,
                    url.c_str(), j.value("n_ctx", 0), n_ctx_seq, n_seq_max);
            return true;
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to connect to remote draft server '%s': %s\n", __func__, url.c_str(), e.what());
            return false;
        }
    }

    bool ensure_weights() {
        if (!dspark || weights_sent || remote_weights_ready) {
            weights_sent = true;
            return true;
        }
        if (!ensure_connected()) {
            return false;
        }

        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);

        cli->set_read_timeout(600, 0);
        cli->set_write_timeout(600, 0);

        const size_t chunk = 32 * 1024 * 1024;

        for (const char * name : { "tok_embd", "output" }) {
            const ggml_tensor * t = llama_model_get_ext_source(model_tgt, name);
            if (t == nullptr) {
                LOG_ERR("%s: the target model has no '%s' tensor to send\n", __func__, name);
                cli->set_read_timeout(60, 0);
                cli->set_write_timeout(10, 0);
                return false;
            }

            const size_t total = ggml_nbytes(t);
            std::vector<char> buf;

            for (size_t off = 0; off < total; off += chunk) {
                const size_t n = std::min(chunk, total - off);
                buf.resize(n);
                ggml_backend_tensor_get(t, buf.data(), off, n);

                const std::string url =
                    std::string("/weights?name=") + name +
                    "&type="   + std::to_string((int) t->type) +
                    "&ne0="    + std::to_string(t->ne[0]) +
                    "&ne1="    + std::to_string(t->ne[1]) +
                    "&offset=" + std::to_string(off) +
                    "&total="  + std::to_string(total);

                if (off > 0 && off % (8 * chunk) == 0) {
                    LOG_INF("%s: %s upload %.1f/%.1f MiB\n", __func__, name,
                            off / 1024.0 / 1024.0, total / 1024.0 / 1024.0);
                }

                auto res = cli->Post(url, std::string(buf.data(), n), "application/octet-stream");
                if (!res || res->status != 200) {
                    LOG_ERR("%s: failed to send %s at offset %zu (HTTP %d)\n", __func__, name, off,
                            res ? res->status : 0);
                    cli->set_read_timeout(60, 0);
                    cli->set_write_timeout(10, 0);
                    return false;
                }

                const nlohmann::json j = nlohmann::json::parse(res->body, nullptr, false);
                if (j.is_discarded() || !j.value("ok", false)) {
                    LOG_ERR("%s: remote draft server rejected %s: %s\n", __func__, name,
                            j.is_discarded() ? "invalid response" : j.value("error", std::string("unknown")).c_str());
                    cli->set_read_timeout(60, 0);
                    cli->set_write_timeout(10, 0);
                    return false;
                }
            }

            LOG_INF("%s: sent target %s (%.1f MiB)\n", __func__, name, total / 1024.0 / 1024.0);
        }

        cli->set_read_timeout(60, 0);
        cli->set_write_timeout(10, 0);

        weights_sent = true;
        return true;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        if (seq_id >= 0 && seq_id < (llama_seq_id) states.size()) {
            states[seq_id].need_reset = true;
        }

        if (dspark && !ensure_weights()) {
            on_failure();
        }
    }

    bool process(const llama_batch & batch) override {
        if (!dspark || disabled) {
            return true;
        }
        if (batch.n_tokens <= 0 || batch.pos == nullptr || batch.seq_id == nullptr) {
            return true;
        }

        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);

        // target layer inputs for this batch, one row per token
        std::vector<const float *> layer_ptr;
        layer_ptr.reserve(target_layers.size());

        for (const int32_t lid : target_layers) {
            const float * p = llama_get_embeddings_layer_inp(params.ctx_tgt, (uint32_t) lid);
            if (p == nullptr) {
                LOG_WRN("%s: target layer %d input is not available\n", __func__, lid);
                return true;
            }
            layer_ptr.push_back(p);
        }

        const int32_t n_embd = llama_model_n_embd(model_tgt);
        if (n_embd != n_embd_tgt) {
            return true;
        }

        const size_t keep = (size_t) std::max(block_size, n_swa) + 16;

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const llama_seq_id seq_id = batch.seq_id[i][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) pending.size()) {
                continue;
            }

            feat_row row;
            row.pos = batch.pos[i];
            row.data.resize((size_t) feat_dim);

            for (size_t k = 0; k < layer_ptr.size(); ++k) {
                std::memcpy(row.data.data() + k * (size_t) n_embd,
                        layer_ptr[k] + (size_t) i * n_embd,
                        (size_t) n_embd * sizeof(float));
            }

            auto & q = pending[seq_id];
            q.push_back(std::move(row));

            while (q.size() > keep) {
                q.pop_front();
            }
        }

        return true;
    }

    bool reset(llama_seq_id seq_id, const llama_tokens & prompt) {
        if (seq_id >= 0 && seq_id < (llama_seq_id) pending.size()) {
            auto & q = pending[seq_id];
            while (!q.empty() && q.back().pos >= (llama_pos) prompt.size()) {
                q.pop_back();
            }
            for (auto & row : q) {
                row.sent = false;
            }
        }

        nlohmann::json req;
        req["seq"]    = seq_id;
        req["tokens"] = prompt;

        cli->set_read_timeout(120, 0);
        auto res = cli->Post("/reset", req.dump(), "application/json");

        if (!res) {
            LOG_ERR("%s: reset failed for seq %d: %s\n", __func__, (int) seq_id, httplib::to_string(res.error()).c_str());
            cli.reset();
            return false;
        }
        cli->set_read_timeout(dspark ? 60 : 10, 0);
        if (res->status != 200) {
            LOG_ERR("%s: reset failed for seq %d: HTTP %d\n", __func__, (int) seq_id, res->status);
            return false;
        }

        const nlohmann::json j = nlohmann::json::parse(res->body, nullptr, false);
        if (j.is_discarded() || !j.value("ok", false)) {
            const std::string err = j.is_discarded() ? "invalid response" : j.value("error", std::string("unknown"));
            LOG_ERR("%s: reset rejected for seq %d: %s\n", __func__, (int) seq_id, err.c_str());
            return false;
        }

        return true;
    }

    draft_result draft_request(llama_seq_id seq_id, const common_speculative_draft_params & dp, int32_t n_max) {
        draft_result result;

        if (dspark && seq_id >= 0 && seq_id < (llama_seq_id) ahead.size()) {
            auto & ap = ahead[seq_id];
            if (ap.full && ap.all.size() > (size_t) ap.n_draft) {
                ++ahead_n;
                if (dp.id_last == ap.all[(size_t) ap.n_draft]) {
                    ++ahead_hit;
                }
                if ((ahead_n % 16) == 0) {
                    LOG_WRN("%s: draft-ahead anchor match %llu/%llu (%.0f%%), n_draft=%d block=%zu\n",
                            __func__, (unsigned long long) ahead_hit, (unsigned long long) ahead_n,
                            100.0 * (double) ahead_hit / (double) ahead_n, ap.n_draft, ap.all.size());
                }
            }
            ap.full = false;
            ap.all.clear();
        }

        if (dspark) {
            if (!ensure_weights()) {
                return result;
            }
            n_max = std::min(n_max, block_size);
            cli->set_read_timeout(60, 0);
        }

        nlohmann::json req;
        req["seq"]         = seq_id;
        req["n_past"]      = dp.n_past;
        req["id_last"]     = dp.id_last;
        req["n_max"]       = n_max;
        req["n_min"]       = params.n_min;
        req["p_min"]       = params.p_min;
        // hash the full prefix: a tail-only window can miss middle-token removals from context shifts
        req["prefix_hash"] = common_speculative_hash_to_hex(
                common_speculative_tokens_hash(dp.prompt->data(), dp.prompt->size(), dp.prompt->size()));

        if (dspark) {
            auto & q = pending[seq_id];

            while (!q.empty() && q.back().pos >= dp.n_past) {
                q.pop_back();
            }

            std::vector<llama_pos> fpos;
            std::string feat;

            for (const auto & row : q) {
                if (row.sent) {
                    continue;
                }
                if (!fpos.empty() && row.pos != fpos.back() + 1) {
                    // keep only the latest contiguous run; the draft cache cannot
                    // represent gaps in the feature positions
                    fpos.clear();
                    feat.clear();
                }
                fpos.push_back(row.pos);
                feat.append(reinterpret_cast<const char *>(row.data.data()), row.data.size() * sizeof(float));
            }

            if (!fpos.empty()) {
                req["feat_pos"] = fpos;
                req["feat"]     = base64::encode(feat.data(), feat.size());
            }
        }

        auto res = cli->Post("/draft", req.dump(), "application/json");
        if (!res) {
            LOG_ERR("%s: draft request failed for seq %d: %s\n", __func__, (int) seq_id,
                    httplib::to_string(res.error()).c_str());
            cli.reset();
            return result;
        }
        if (res->status != 200) {
            LOG_ERR("%s: draft request failed for seq %d: HTTP %d\n", __func__, (int) seq_id, res->status);
            return result;
        }

        const nlohmann::json j = nlohmann::json::parse(res->body, nullptr, false);
        if (j.is_discarded()) {
            LOG_ERR("%s: invalid draft response for seq %d\n", __func__, (int) seq_id);
            return result;
        }

        if (!j.value("ok", false)) {
            result.need_reset = j.value("need_reset", false);
            if (!result.need_reset) {
                const std::string err = j.value("error", std::string("unknown"));
                LOG_ERR("%s: draft rejected for seq %d: %s\n", __func__, (int) seq_id, err.c_str());
            }
            return result;
        }

        result.ok = true;
        if (j.contains("tokens")) {
            for (const auto & id : j["tokens"]) {
                result.tokens.push_back((llama_token) id.get<int32_t>());
            }
        }

        if (dspark) {
            for (auto & row : pending[seq_id]) {
                if (row.pos < dp.n_past) {
                    row.sent = true;
                }
            }
            if (seq_id >= 0 && seq_id < (llama_seq_id) ahead.size() && j.contains("all")) {
                auto & ap = ahead[seq_id];
                ap.all.clear();
                for (const auto & id : j["all"]) {
                    ap.all.push_back((llama_token) id.get<int32_t>());
                }
                ap.n_draft = (int) result.tokens.size();
            }
        }

        return result;
    }

    void on_failure() {
        if (++n_fail >= 3) {
            disabled = true;
            LOG_ERR("%s: too many remote draft failures, disabling remote drafting\n", __func__);
        }
    }

    void on_success() {
        n_fail = 0;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        if (disabled) {
            return;
        }

        if (!ensure_connected()) {
            on_failure();
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            if (dp.prompt == nullptr || dp.prompt->size() != (size_t) dp.n_past) {
                // mtmd prompts drop LLAMA_TOKEN_NULL entries, so the text prompt no longer matches n_past
                if (!warned_prompt_mismatch) {
                    warned_prompt_mismatch = true;
                    LOG_WRN("%s: prompt size does not match n_past (multimodal?), remote drafting is skipped\n", __func__);
                }
                continue;
            }

            int32_t n_max = params.n_max;
            if (dp.n_max > 0) {
                n_max = std::min(n_max, dp.n_max);
            }
            if (n_max <= 0) {
                continue;
            }

            bool ok = false;

            // one reset + retry per call
            for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
                if (states[seq_id].need_reset) {
                    if (!reset(seq_id, *dp.prompt)) {
                        break;
                    }
                    states[seq_id].need_reset = false;
                }

                const draft_result res = draft_request(seq_id, dp, n_max);
                if (res.need_reset) {
                    states[seq_id].need_reset = true;
                    continue;
                }
                if (!res.ok) {
                    break;
                }

                auto & result = *dp.result;
                result.clear();

                for (const llama_token id : res.tokens) {
                    if (id < 0 || id >= n_vocab) {
                        LOG_ERR("%s: remote draft returned invalid token %d (n_vocab = %d)\n", __func__, (int) id, n_vocab);
                        result.clear();
                        break;
                    }
                    result.push_back(id);
                }

                if ((int32_t) result.size() > n_max) {
                    result.resize(n_max);
                }
                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                }

                ok = true;
            }

            if (ok) {
                on_success();
            } else {
                on_failure();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other || seq_id < 0 || seq_id >= (llama_seq_id) ahead.size()) {
            return;
        }
        auto & ap = ahead[seq_id];
        // an ahead continuation is only meaningful when every requested column was accepted: otherwise
        // the next anchor is the target's own sample, not the next draft, and the chain must resync
        ap.full = !ap.all.empty() && (int) n_accepted == ap.n_draft;
        if (!ap.full) {
            ap.all.clear();
        }
    }

    bool need_embd() const override {
        return false;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-simple'\n", __func__);
        LOG_INF("%s: - size_n=%d, size_m=%d, min_hits=%d\n", __func__,
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(this->type).c_str());
        LOG_INF("%s: - size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n", __func__,
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: adding speculative implementation 'ngram-mod'\n", __func__);
        LOG_INF("%s: - n_match=%d, n_max=%d, n_min=%d\n", __func__,
                this->params.n_match, this->params.n_max, this->params.n_min);
        LOG_INF("%s: - mod size=%zu (%.3f MB)\n", __func__,
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) - resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-cache'\n", __func__);
        LOG_INF("%s: - n_draft=%d, cache_static=%s, cache_dynamic=%s\n", __func__,
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        case COMMON_SPECULATIVE_TYPE_DRAFT_REMOTE:  return "draft-remote";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_REMOTE:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3
        bool has_mtp = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) && params.draft.ctx_dft != nullptr;
        bool has_draft_remote = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_REMOTE)) && !params.draft.remote_endpoint.empty();

        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        if ((enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_REMOTE)) && params.draft.remote_endpoint.empty()) {
            LOG_WRN("%s: 'draft-remote' is enabled but no endpoint is set (use --spec-draft-remote HOST:PORT)\n", __func__);
        }

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 10);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
        if (has_draft_remote) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_REMOTE, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_REMOTE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_remote>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_nextn(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_nextn()) {
            return true;
        }
    }

    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
