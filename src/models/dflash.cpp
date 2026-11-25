#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache-iswa.h"

#include <cmath>
#include <stdexcept>
#include <string>

// DFlash / DSpark draft models: a small block-diffusion transformer that reads target
// features through an encoder and injects them into a sliding-window KV cache.

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EMBEDDING_SCALE,       hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE,       hparams.f_attention_scale, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,           hparams.f_logit_scale, false);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    hparams.llm_ffn_op = LLM_FFN_SILU;

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE, hparams.dflash_block_size, false);
    if (hparams.dflash_block_size < 2) {
        throw std::runtime_error("dflash block_size must be >= 2");
    }

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("dflash model requires 'target_layers' in GGUF metadata");
    }
    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: dflash extract_layers = [%s]\n", __func__, layers.c_str());

    // only the DeepSeek-V4.1 (single-pass mHC) backbone is supported for now
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult == 0) {
        throw std::runtime_error("dflash: only the DeepSeek-V4 backbone is supported");
    }

    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,                hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,             hparams.n_swa);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,           hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,                  hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,                 hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,                  hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                   hparams.expert_gating_func);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,              hparams.swiglu_clamp_exp, hparams.n_layer_all);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,       hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINGLE_PASS,         hparams.dsv4_hc_single_pass, false);
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS,            hparams.dsv4_compress_ratios, false);

    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("dflash DSV4 draft expects sqrtsoftplus MoE scoring");
    }
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (hparams.dsv4_compress_ratios[il] != 0) {
            throw std::runtime_error("dflash DSV4 draft expects uncompressed attention on all stages");
        }
    }

    GGML_ASSERT(hparams.n_swa > 0);
    if (hparams.dsv4_hc_single_pass) {
        // the block sees the main-KV ring through the mask row of its first query
        hparams.n_swa += 1;
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,     "weight"), { n_embd }, 0);

    const struct ggml_tensor * markov_meta = ml->get_tensor_meta("markov_w1.weight");
    if (markov_meta) {
        const int64_t rank = markov_meta->ne[0];

        dspark_markov_w1   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { rank, n_vocab }, 0);
        dspark_markov_w2   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { rank, n_vocab }, 0);
        dspark_markov_w2_s = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);

        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + rank, 1 }, TENSOR_NOT_REQUIRED);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);

        LLAMA_LOG_INFO("%s: dflash with DSpark markov head (rank = %lld)\n", __func__, (long long) rank);
    } else {
        throw std::runtime_error("dflash: only DSpark drafts (markov head) are supported");
    }

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_embd_head     = hparams.n_embd_head_k();
    const int64_t o_groups        = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult         = hparams.dsv4_hc_mult;
    const int64_t hc_dim          = hc_mult * n_embd;
    const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), { n_embd }, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), { n_head }, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), { n_embd, q_lora_rank }, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), { q_lora_rank }, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), { q_lora_rank, n_head * n_embd_head }, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), { n_embd, n_embd_head }, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), { n_embd_head }, 0);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), { n_head * n_embd_head / o_groups, o_lora_rank * o_groups }, 0);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), { o_groups * o_lora_rank, n_embd }, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), { hc_dim, hc_mix_dim }, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), { hc_mix_dim }, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), { 3 }, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), { hc_dim, hc_mix_dim }, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), { hc_mix_dim }, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), { 3 }, 0);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), { n_embd, n_expert }, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), { n_expert }, 0);
        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), { n_embd }, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd,   n_ff_exp, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd,   n_expert }, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), { n_embd,   n_ff_exp, n_expert }, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd,                     n_ff_exp * n_expert_shared }, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_exp * n_expert_shared, n_embd                     }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), { n_embd,                     n_ff_exp * n_expert_shared }, 0);
    }
}

// multiply by the cache's Hadamard rotation constants (same as ggml_mul_mat_aux in llama-graph.cpp)
static ggml_tensor * dflash_mul_mat_rot(ggml_context * ctx0, ggml_tensor * cur, ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;
    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d(ctx0, cur, n, ggml_nelements(cur)/n);
    } else {
        res = ggml_reshape_2d(ctx0, cur, n, ggml_nelements(cur)/n);
    }
    res = ggml_mul_mat(ctx0, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);

    return ggml_reshape_4d(ctx0, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}

// split the rope dims off the tail where ggml_rope_set_offset would rotate
static ggml_tensor * dflash_rope(
        ggml_context * ctx0, ggml_tensor * t, ggml_tensor * pos,
        int64_t n_rot, int64_t n_nope, int rope_type, float freq_base) {
    ggml_tensor * nope;
    ggml_tensor * pe;
    if (ggml_n_dims(t) <= 2) {
        nope = ggml_view_2d(ctx0, t, n_nope, t->ne[1], t->nb[1], 0);
        pe   = ggml_view_2d(ctx0, t, n_rot,  t->ne[1], t->nb[1], (size_t) n_nope*ggml_element_size(t));
    } else {
        nope = ggml_view_3d(ctx0, t, n_nope, t->ne[1], t->ne[2], t->nb[1], t->nb[2], 0);
        pe   = ggml_view_3d(ctx0, t, n_rot,  t->ne[1], t->ne[2], t->nb[1], t->nb[2],
                (size_t) n_nope*ggml_element_size(t));
    }

    pe = ggml_rope_ext(ctx0, pe, pos, nullptr, n_rot, rope_type, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    return ggml_concat(ctx0, nope, pe, 0);
}

static ggml_tensor * dflash_derope(
        ggml_context * ctx0, ggml_tensor * t, ggml_tensor * pos,
        int64_t n_rot, int64_t n_nope, int64_t n_head, int64_t n_tokens,
        int rope_type, float freq_base) {
    ggml_tensor * out_nope = ggml_view_3d(ctx0, t, n_nope, n_head, n_tokens, t->nb[1], t->nb[2], 0);
    ggml_tensor * out_pe   = ggml_view_3d(ctx0, t, n_rot,  n_head, n_tokens, t->nb[1], t->nb[2],
            (size_t) n_nope*ggml_element_size(t));

    out_pe = ggml_rope_ext_back(ctx0, out_pe, pos, nullptr, n_rot, rope_type, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    return ggml_concat(ctx0, out_nope, out_pe, 0);
}

// DSpark markov + confidence heads. The base logits are biased per block position with
// w2 @ w1[prev]; the chain is greedy in-graph (the block sampler picks the final tokens).
static void build_dspark_markov_head(
        llm_graph_context & g,
        const llama_model & model,
        ggml_tensor * tokens) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = *g.res;

    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && "DSpark markov weights not loaded");

    const bool has_conf = model.dspark_conf_proj != nullptr;

    ggml_tensor * base = res.t_logits;
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    const int64_t block_size = g.hparams.dflash_block_size;
    GGML_ASSERT(block_size > 0);

    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tok % n_blocks == 0 && "DSpark markov head requires equal-size blocks");

    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return;
    }

    // anchor token (committed last) of every block: token 0 of each block
    const size_t token_stride = (size_t) block_drafts * tokens->nb[0];
    const size_t base_stride  = (size_t) block_drafts * base->nb[1];

    ggml_tensor * prev = ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    // a bonus anchor slot is not a prediction slot
    const int64_t i_draft_beg = 0;

    ggml_tensor * cat      = nullptr;
    ggml_tensor * cat_conf = nullptr;

    const float softcap = g.hparams.f_final_logit_softcapping;

    for (int64_t i = i_draft_beg; i < block_drafts; ++i) {
        ggml_tensor * markov = ggml_get_rows(ctx0, w1, prev); // [rank, n_blocks]

        ggml_tensor * bias = g.build_lora_mm(w2, markov, model.dspark_markov_w2_s);
        if (model.dspark_markov_w2_s == nullptr && w2->ne[1] != n_vocab) {
            GGML_ABORT("dflash: markov_w2 has an unexpected vocabulary size");
        }

        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, i*base->nb[1]);
        ggml_tensor * col    = ggml_add(ctx0, base_i, bias);
        if (softcap > 0.0f) {
            col = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, col, 1.0f/softcap)), softcap);
        }

        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        if (has_conf) {
            ggml_tensor * conf_inp = res.t_embd;
            ggml_tensor * conf_i = ggml_view_2d(ctx0, conf_inp, conf_inp->ne[0], n_blocks,
                    (size_t) block_drafts * conf_inp->nb[1], i*conf_inp->nb[1]);
            ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_i), markov, 0);
            ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
            if (model.dspark_conf_proj_b) {
                conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
            }
            conf = ggml_sigmoid(ctx0, conf);

            cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;
        }

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    // cat is position-major; restore the ubatch block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3));
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    if (has_conf) {
        ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);
        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
        conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);
        conf = ggml_repeat(ctx0, conf, res.t_embd);
        res.t_h_nextn = conf;
        ggml_build_forward_expand(g.gf, conf);
    }

    res.t_logits = out;
    ggml_build_forward_expand(g.gf, out);
}

llama_model_dflash::graph_dsv41::graph_dsv41(const llama_model & model, const llm_graph_params & params)
    : llama_model_deepseek41::graph(params), model(model) {
    const int64_t n_embd_inp       = hparams.n_embd_inp_enc();
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    GGML_ASSERT(hparams.dsv4_hc_single_pass);

    // remote dflash drafts receive the target embeddings after the context is
    // created; build a minimal reserve graph until then
    if (ubatch.token && model.tok_embd == nullptr && model.tok_embd_ext == nullptr) {
        ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(cur);
        ggml_set_output(cur);

        res->t_embd = cur;

        ggml_build_forward_expand(gf, cur);
        return;
    }

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_kv_iswa * inp_attn = build_attn_inp_kv_iswa();

    GGML_ASSERT(ubatch.token || ubatch.embd);

    if (ubatch.embd) {
        // injection: fuse target features and write the shared latent KV into the ring
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_target = inp->embd;
        res->add_input(std::move(inp));

        ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);
        inp_g = build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
        cb(inp_g, "main_x", -1);

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * kv = build_lora_mm(layer.wkv, inp_g);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
            kv = dflash_rope(ctx0, kv, inp_pos, n_embd_head_rope, n_embd_head_nope, rope_type, freq_base);
            cb(kv, "kv_injected", il);

            ggml_tensor * k_cur = kv;
            ggml_tensor * v_cur = kv;
            if (inp_attn->self_k_rot_swa) {
                k_cur = dflash_mul_mat_rot(ctx0, k_cur, inp_attn->self_k_rot_swa);
            }
            if (inp_attn->self_v_rot_swa) {
                v_cur = dflash_mul_mat_rot(ctx0, v_cur, inp_attn->self_v_rot_swa);
            }

            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, k_cur, inp_attn->get_k_idxs_swa(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_v(ctx0, v_cur, inp_attn->get_v_idxs_swa(), il));
        }

        res->t_embd = inp_g;
        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    ggml_tensor * tok_embd = model.tok_embd ? model.tok_embd : model.tok_embd_ext;
    GGML_ASSERT(tok_embd != nullptr);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;
    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    res->add_input(std::move(inp));
    cb(inpL, "inp_noise_embd", -1);

    const int64_t hc            = hparams.dsv4_hc_mult;
    const int64_t n_groups      = hparams.dsv4_o_group_count;
    const int64_t n_heads_group = n_head / n_groups;
    const int64_t o_lora_rank   = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim   = n_heads_group*n_embd_head;
    const float   kq_scale      = 1.0f/sqrtf(float(n_embd_head));

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // the whole block shares the mask row of its first query (the anchor)
    {
        ggml_tensor * m = inp_attn->self_kq_mask_swa_cnv;
        const int64_t n_blocks = ubatch.n_seqs_unq;
        const int64_t bd       = n_blocks > 0 ? n_tokens / n_blocks : 0;
        if (m && bd > 1 && n_tokens % n_blocks == 0 && m->ne[2] == 1 && m->ne[1] % bd == 0 && ggml_is_contiguous(m)) {
            ggml_tensor * m4 = ggml_reshape_4d(ctx0, m, m->ne[0], bd, m->ne[1]/bd, m->ne[3]);
            ggml_tensor * r0 = ggml_view_4d(ctx0, m4, m4->ne[0], 1, m4->ne[2], m4->ne[3], m4->nb[1], m4->nb[2], m4->nb[3], 0);
            m4 = ggml_repeat(ctx0, r0, m4);
            inp_attn->self_kq_mask_swa_cnv = ggml_reshape_4d(ctx0, m4, m->ne[0], m->ne[1], 1, m->ne[3]);
            cb(inp_attn->self_kq_mask_swa_cnv, "dspark_block_mask", -1);
        }
    }

    ggml_tensor * pre_prev = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * attn_pre = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);

        ggml_tensor * cur;
        if (pre_prev == nullptr) {
            cur = ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[2], 0));
        } else {
            cur = build_hc_pre(inpL, pre_prev, il);
        }
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
            qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "qr_norm", il);

            ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
            q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
            q = dflash_rope(ctx0, q, inp_pos, n_embd_head_rope, n_embd_head_nope, rope_type, freq_base);
            cb(q, "q", il);

            ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
            kv = dflash_rope(ctx0, kv, inp_pos, n_embd_head_rope, n_embd_head_nope, rope_type, freq_base);
            cb(kv, "kv", il);

            ggml_tensor * out = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    q, kv, kv,
                    nullptr, layer.attn_sinks, nullptr,
                    kq_scale, il);
            cb(out, "attn_raw", il);

            out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, n_tokens);
            out = dflash_derope(ctx0, out, inp_pos, n_embd_head_rope, n_embd_head_nope, n_head, n_tokens, rope_type, freq_base);
            cb(out, "attn_derope", il);

            out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, n_tokens);
            out = ggml_permute(ctx0, out, 0, 2, 1, 3);
            ggml_tensor * oa = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), out);
            cb(oa, "attn_wo_a", il);
            oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
            oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, n_tokens);

            cur = build_lora_mm(layer.wo_b, oa);
            cb(cur, "attn_out", il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        ggml_tensor * ffn_pre = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);

        ggml_tensor * l_out = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        cb(l_out, "l_out", il);
        ggml_build_forward_expand(gf, l_out);

        pre_prev = ffn_pre;
    }

    ggml_tensor * cur = build_hc_pre(inpL, pre_prev, -1);
    cb(cur, "hc_out", -1);

    // confidence head takes the un-normed collapsed state
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * output = model.output ? model.output : model.output_ext;
    if (output == nullptr) {
        throw std::runtime_error("dflash: the draft has no output projection; provide the target lm head");
    }

    cur = build_lora_mm(output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    build_dspark_markov_head(*this, model, inp_tokens);
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph_dsv41>(*this, params);
        default:
            throw std::runtime_error("dflash: unsupported graph type");
    }
}
