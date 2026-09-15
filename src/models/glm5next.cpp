#include "models.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// GLM-5.3-Flash (arch string "glm5next"): hybrid MoE that mixes Kimi-style KDA
// linear attention (recurrent layers) with weight-absorbed MLA on the full
// attention layers, both wrapped in DeepSeek-V4 sinkhorn hyper-connections (hc).
// The MLA layers carry a block-pooled DSA lightning indexer that restricts each
// query to the top_k cells it scores (build_dsa_top_k -> build_attn_dsa); the
// increment-1 pool/head-reduction are placeholders (mean / uniform sum) pending
// the learned softmax-gated compressor. LLAMA_GLM5NEXT_DSA_DENSE=1 disables the
// indexer and runs dense MLA. The NextN (MTP) head is still loaded but unused
// (see the nextn load-unused block). All ops are the fork's existing ones - no
// new kernels.

// KDA forget-gate floor. GLM5NEXT applies a per-head lower bound to g1 that Kimi
// does not; there is no dedicated gguf key for it, so it lives here as a constant.

void llama_model_glm5next::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // MoE
    ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

    // MLA (NoPE, q-compressed, weight-absorbed)
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);

    // KDA (delta-net) linear attention
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);

    // DSA lightning indexer (loaded for MLA layers, unused in phase-1 graph)
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    // sinkhorn hyper-connection (shared with deepseek4; loader asserts hc == 4)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);

    // NextN / MTP head (block block_count-1, load-unused in phase-1)
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all);

    // KDA forget-gate lower bound (no gguf key; matches the reference implementation)
    hparams.kda_gate_lower_bound = -5.0f;

    // (Swi)GLU clamps: the gguf arrays are block_count long (include the nextn
    // block), so they must be read with n_layer_all - get_key_or_arr throws on an
    // exact-length mismatch. The clamp is applied in build_ffn / build_moe_ffn,
    // arch-gated on LLM_ARCH_GLM5NEXT.
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer_all, false);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    // Layer schedule: n_head_kv[il] == 0 marks a KDA (recurrent) layer, == 1 an
    // MLA (attention) layer. The head_count_kv array was read generically over
    // [0, n_layer_all), so fill is_recr for the full range - the nextn block is
    // an MLA layer (n_head_kv == 1) and its is_recr slot must be valid too.
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    // DSA indexer block pooling (kpool). There is no dedicated gguf key; the reference uses 4.
    // Reuse dsv4_compress_ratios[il] as the per-layer pool size: kpool on the MLA (full-attention)
    // layers that own an indexer, 0 on KDA layers and the nextn block. llama_memory_hybrid_idx sizes
    // the idx cache from indexer_head_size alone, so this only drives the graph, not the memory layout.
    const uint32_t indexer_kpool = 4;
    GGML_ASSERT(hparams.indexer_top_k % indexer_kpool == 0);
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.dsv4_compress_ratios[i] = (i < hparams.n_layer() && !hparams.is_recr(i)) ? indexer_kpool : 0;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_glm5next::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank        = hparams.n_lora_q;
    const int64_t kv_lora_rank       = hparams.n_lora_kv;
    const int64_t n_embd_head_k_mla  = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla  = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();                       // 0 (NoPE)
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t head_dim   = hparams.n_embd_head_kda;
    const int64_t d_inner    = head_dim * n_head;                              // KDA q/k/v width
    const int64_t d_conv     = hparams.ssm_d_conv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t n_indexer_head = hparams.indexer_n_head;
    // indexer_compressor_ape trailing dim (per-head pool factor); loaded but unused in phase-1
    const int64_t indexer_comp_pool = 4;

    const int64_t hc_mult    = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc_mult * n_embd;
    const int64_t hc_mix_dim = (2 + hc_mult) * hc_mult;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer_all; ++i) {
        // The nextn block (i >= n_layer) is preserved but unused: skip-load it.
        const int flags = (i >= n_layer) ? (TENSOR_SKIP | TENSOR_NOT_REQUIRED) : 0;

        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        if (hparams.is_recr(i)) {
            // KDA linear-attention layer (recurrent). Conv weights are stored 3D.
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {d_conv, 1, d_inner}, flags);
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {d_conv, 1, d_inner}, flags);
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {d_conv, 1, d_inner}, flags);

            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, d_inner}, flags);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, d_inner}, flags);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, d_inner}, flags);

            layer.ssm_f_a  = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", i), {n_embd, head_dim}, flags);
            layer.ssm_f_b  = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", i), {head_dim, d_inner}, flags);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, flags);
            layer.ssm_a    = create_tensor(tn(LLM_TENSOR_SSM_A, i), {n_head}, flags);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, flags);
            layer.ssm_g_a  = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", i), {n_embd, head_dim}, flags);
            layer.ssm_g_b  = create_tensor(tn(LLM_TENSOR_SSM_G_B, "weight", i), {head_dim, d_inner}, flags);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim}, flags);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, flags);
        } else {
            // MLA attention layer (weight-absorbed) + DSA indexer (loaded, unused in phase-1).
            layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);

            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);

            // DSA lightning indexer
            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer}, flags);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {n_embd_indexer}, flags);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_indexer_head}, flags);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, n_embd_indexer}, flags);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, n_indexer_head * n_embd_indexer}, flags);
            layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {n_embd_indexer, indexer_comp_pool}, flags);
            layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_indexer}, flags);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);
        }

        if (i < n_layer) {
            // sinkhorn hyper-connection mixers (per attn and per ffn sub-block)
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);
        }

        if (i >= n_layer) {
            // NextN / MTP head - preserved but unused (phase-1 defers speculative decode).
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_glm5next::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

//
// sinkhorn hyper-connection helpers (ported verbatim from deepseek4; both assert hc == 4)
//

static size_t glm5next_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * glm5next_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, glm5next_elem_offset(t, i0));
}

static ggml_tensor * glm5next_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], glm5next_elem_offset(t, i0));
}

static ggml_tensor * glm5next_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_glm5next::graph::build_hc_weighted_sum(
        ggml_tensor * x,
        ggml_tensor * weights) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * acc = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);

        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        acc = acc ? ggml_add(ctx0, acc, cur) : cur;
    }

    return acc;
}

ggml_tensor * llama_model_glm5next::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Same sinkhorn as deepseek4: row softmax
    // over dst, one column normalization, then repeated row/column normalization.
    // LLAMA_DSV4_FUSED_SINKHORN (default ON, =0 to fall back to the op chain) runs
    // the whole thing as one GGML_OP_HC_SINKHORN kernel. CUDA-only, so fall back to
    // the chain when no CUDA device is registered.
    static const bool fused = []() {
        const char * e = getenv("LLAMA_DSV4_FUSED_SINKHORN");
        if (e && atoi(e) == 0) {
            return false;
        }
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const char * name = dev ? ggml_backend_dev_name(dev) : nullptr;
            if (name && strstr(name, "CUDA") != nullptr) {
                return true;
            }
        }
        return false;
    }();
    uint32_t n_iter = hparams.dsv4_hc_sinkhorn_iters;
    if (const char * e = getenv("LLAMA_DSV4_SINKHORN_ITERS")) {
        const int v = atoi(e);
        if (v > 0) { n_iter = (uint32_t) v; }
    }
    if (fused && comb->ne[0] == comb->ne[1] && comb->ne[3] == 1 &&
        (comb->ne[0] == 2 || comb->ne[0] == 4)) {
        ggml_tensor * a = ggml_is_contiguous(comb) ? comb : ggml_cont(ctx0, comb);
        return ggml_hc_sinkhorn(ctx0, a, hparams.dsv4_hc_eps, (int) n_iter);
    }

    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < n_iter; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

ggml_tensor * llama_model_glm5next::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = glm5next_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = glm5next_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = glm5next_view_1d(ctx0, hc_scale, 1, 2);

    ggml_tensor * base_pre  = glm5next_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = glm5next_view_1d(ctx0, hc_base, hc, hc);
    ggml_tensor * base_comb = glm5next_view_1d(ctx0, hc_base, hc*hc, 2*hc);

    ggml_tensor * pre = glm5next_view_2d(ctx0, mixes, hc, nt, 0);
    pre = glm5next_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = glm5next_view_2d(ctx0, mixes, hc, nt, hc);
    *post = glm5next_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    *comb = glm5next_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
    *comb = glm5next_hc_affine(ctx0, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    *comb = build_hc_sinkhorn(*comb, il);
    cb(*comb, "hc_comb", il);

    return build_hc_weighted_sum(x, pre);
}

ggml_tensor * llama_model_glm5next::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_UNUSED(il);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_glm5next::graph::build_hc_mean(ggml_tensor * x) const {
    // Collapse the hc axis of [n_embd, hc, n_tokens] by mean. GLM5NEXT has no
    // learned output hc head; the result feeds an RMS norm, so the mean-vs-sum
    // scale factor is irrelevant, but keep the divide for a faithful residual.
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * acc = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        acc = acc ? ggml_add(ctx0, acc, xh) : ggml_cont(ctx0, xh);
    }

    return ggml_scale(ctx0, acc, 1.0f/(float) hc);
}

//
// KDA linear attention (delta-net); ported from kimi-linear with a forget-gate clamp
//

// Causal Conv1d for Q/K/V. qkv selects the sub-state: 0 = Q, 1 = K, 2 = V.
static ggml_tensor * glm5next_causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w, int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total = 3 * conv_state_size;  // Q + K + V

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all,
                d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

ggml_tensor * llama_model_glm5next::graph::build_kda(llm_graph_input_rs * inp_rs, ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];

    const int64_t n_head   = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t d_inner  = n_head * head_dim;
    const int64_t n_seqs   = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const float   eps_norm = hparams.f_norm_rms_eps;

    const auto * mctx_cur = inp_rs->mctx;
    const auto kv_head = mctx_cur->get_head();

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
    ggml_tensor * Qcur = glm5next_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Kcur = glm5next_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Vcur = glm5next_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

    // g1 = gate_lower_bound * sigmoid(-ssm_a * (f_b(f_a(x)) + dt_bias)), ssm_a = -exp(A_log)
    // i.e. gate_lower_bound * sigmoid(exp(A_log) * (...)), a smooth forget gate bounded in (gate_lower_bound, 0)
    ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
    g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

    // A_log already stored as -exp(A_log) by the converter; broadcast [1, n_head, 1]
    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
    g1 = ggml_mul(ctx0, g1, A);                              // (f_b(f_a(x)) + dt_bias) * (-exp(A_log))
    g1 = ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f));    // sigmoid(exp(A_log) * (...))
    g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound); // * gate_lower_bound (-5.0)
    cb(g1, "kda_g1", il);

    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);

    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

    Qcur = ggml_l2_norm(ctx0, Qcur, eps_norm);
    Kcur = ggml_l2_norm(ctx0, Kcur, eps_norm);

    auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

    ggml_tensor * output = ggml_cont(ctx0, attn_out.first);
    ggml_tensor * new_state = attn_out.second;

    ggml_build_forward_expand(gf,
                             ggml_cpy(ctx0, new_state,
                                      ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                                                   kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    // output gating g2 = g_b(g_a(x)); gate is sigmoid (not swish)
    ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
    ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
    ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);

    ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head, n_seq_tokens * n_seqs);
    ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * gate = ggml_sigmoid(ctx0, g2);
    ggml_tensor * gated = ggml_mul(ctx0, normed, gate);

    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
    cur = ggml_mul_mat(ctx0, layer.wo, gated);
    cb(cur, "kda_out", il);

    return cur;
}

//
// DSA lightning indexer (block-pooled sparse attention); skeleton from qwen4exp build_qsa_top_k.
// glm5next is NoPE, so unlike qwen4exp there is no rope on the pooled key or the query. Each cell
// caches its LayerNorm'd key beside its pool gate (2*indexer_head_size row, see
// llama-memory-hybrid-idx.cpp), so the learned pool sum_m softmax_m(g_m + comp_ape[:,m]) * k_m and
// the indexer_proj per-head weighting can be rebuilt for past members at decode.
//

// The idx cache layout inputs do not depend on the layer, only on its pool size, so the layers
// sharing a pool size share one input set. set_input_qsa is arch-agnostic (positions/cells only),
// so it is reused verbatim; blk_pos is filled but unused here (NoPE -> no rope on the pooled key).
class llama_model_glm5next::llm_graph_input_dsa : public llm_graph_input_i {
public:
    llm_graph_input_dsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias) {}
    virtual ~llm_graph_input_dsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= bias->ne[0]      == (blk_bias ? n_blocks : n_kv);
        res &= bias->ne[1]      == params.ubatch.n_tokens/n_stream;

        return res;
    }

    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // unused (NoPE): stays null, set_input_qsa skips it
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    const bool blk_bias;
};

ggml_tensor * llama_model_glm5next::graph::build_dsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           q_lora,
        ggml_tensor *                           kq_mask,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const auto & layer = model.layers[il];

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    llm_graph_input_dsa * inp = nullptr;

    const auto it = dsa_inps.find((uint32_t) r);
    if (it != dsa_inps.end()) {
        inp = it->second;
    } else {
        auto dsa = std::make_unique<llm_graph_input_dsa>(mctx_hyb, (uint32_t) r, blk_bias);

        dsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        dsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        dsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        // no blk_pos: the indexer is NoPE, so the pooled key is never roped. the rope-position
        // input qwen4exp builds here would be unconsumed, hence left unallocated by ggml-alloc
        dsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        ggml_set_input(dsa->cell_blk);
        ggml_set_input(dsa->blk_cells);
        ggml_set_input(dsa->bias);

        inp = dsa.get();
        res->add_input(std::move(dsa));
        dsa_inps.emplace((uint32_t) r, inp);
    }

    // each cell caches its LayerNorm'd key beside its pool gate, so the softmax-gated pool can be
    // rebuilt for past members at decode. norm is per token before caching, as in the DeepSeek
    // lightning indexer (deepseek32 norms then caches), so the pool mixes normed keys and no norm
    // runs after it. the gate g = comp_wgate @ h stays raw until the softmax.
    ggml_tensor * k_raw = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
    ggml_tensor * k_nrm = build_norm(k_raw, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
    ggml_tensor * g_raw = ggml_mul_mat(ctx0, layer.indexer_comp_wgate, cur);

    ggml_tensor * kg = ggml_concat(ctx0, k_nrm, g_raw, 0);
    kg = ggml_reshape_3d(ctx0, kg, 2*idx_dim, 1, n_tokens);
    cb(kg, "indexer_kg", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, kg, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [2*idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, 2*idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // gathers per stream: blk_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
    members = ggml_reshape_4d(ctx0, members, 2*idx_dim, r, n_blocks, n_stream);
    if (members->type != GGML_TYPE_F32) {
        members = ggml_cast(ctx0, members, GGML_TYPE_F32);   // softmax / mul below want f32
    }

    // split the cached row back into the normed key and the pool gate
    ggml_tensor * k_mem = ggml_cont(ctx0, ggml_view_4d(ctx0, members, idx_dim, r, n_blocks, n_stream,
            members->nb[1], members->nb[2], members->nb[3], 0));
    ggml_tensor * g_mem = ggml_view_4d(ctx0, members, idx_dim, r, n_blocks, n_stream,
            members->nb[1], members->nb[2], members->nb[3], idx_dim*members->nb[0]);

    // learned pool: weight member m by softmax_m(g_m + comp_ape[:,m]) per channel, then mix keys
    ggml_tensor * ape = layer.indexer_comp_ape->type == GGML_TYPE_F32 ? layer.indexer_comp_ape
            : ggml_cast(ctx0, layer.indexer_comp_ape, GGML_TYPE_F32);
    ggml_tensor * gate = ggml_add(ctx0, ggml_cont(ctx0, g_mem),
            ggml_reshape_4d(ctx0, ape, idx_dim, r, 1, 1));
    gate = ggml_cont(ctx0, ggml_permute(ctx0, gate, 1, 0, 2, 3));   // [r, idx_dim, n_blocks, n_stream]
    ggml_tensor * w = ggml_soft_max(ctx0, gate);                    // softmax over the r members

    ggml_tensor * k_perm = ggml_cont(ctx0, ggml_permute(ctx0, k_mem, 1, 0, 2, 3));
    ggml_tensor * pooled = ggml_sum_rows(ctx0, ggml_mul(ctx0, w, k_perm));
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
    cb(pooled, "indexer_k", il);

    // indexer query from the shared MLA q-lora (rank n_lora_q); no q-norm tensor, NoPE
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, q_lora);
    q = ggml_reshape_4d(ctx0, q, idx_dim, n_idx_h, n_tps, n_stream);
    cb(q, "indexer_q", il);

    // weight each head by indexer_proj @ h, pre-scaled by 1/sqrt(idx_dim*n_idx_h) as in deepseek32
    ggml_tensor * hw = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
    hw = ggml_scale(ctx0, hw, 1.0f/sqrtf((float) (idx_dim*n_idx_h)));
    hw = ggml_reshape_4d(ctx0, hw, n_idx_h, 1, n_tps, n_stream);      // broadcast over blocks

    // the reference returns indexer_top_k + kpool - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    // fork: the per-head block score mul_mat is [n_blocks, n_idx_h, n_tps] (n_blocks = n_kv/r) plus its
    // relu+permute copy - kpool r is small so n_blocks ~ n_kv, making this pair ~17 GiB at n_ctx 131072
    // / ub 2048, which dominates the reserved compute buffer and spills into shared host memory. Score
    // the query tokens in chunks and top-k each chunk on its own: top-k is per query token, so the
    // per-chunk winners concatenated along the token axis are exactly the unchunked result while the
    // peak transient is bounded to one chunk. LLAMA_DSA_LID_CHUNK_MB=0 restores the single-pass graph.
    const int64_t per_token = n_blocks * n_idx_h * (int64_t) sizeof(float);
    const int64_t n_chunk   = llm_dsa_chunk_tokens(n_tps, per_token);

    // kq_mask is the per-cell half of the DSA bias (f16 under flash attention); viewed per chunk below
    ggml_tensor * kqm3 = ggml_reshape_3d(ctx0, kq_mask, n_kv, n_tps, n_stream);

    ggml_tensor * top_k = nullptr;
    for (int64_t i0 = 0; i0 < n_tps; i0 += n_chunk) {
        const int64_t nc = std::min(n_chunk, n_tps - i0);

        // this chunk's queries: [idx_dim, n_idx_h, nc, n_stream], strided over the token axis
        ggml_tensor * q_cur = ggml_cont(ctx0, ggml_view_4d(ctx0, q, idx_dim, n_idx_h, nc, n_stream,
                q->nb[1], q->nb[2], q->nb[3], i0*q->nb[2]));

        // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
        ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
                ggml_reshape_3d(ctx0, q_cur, idx_dim, n_idx_h*nc, n_stream));
        score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, nc, n_stream);
        score = ggml_relu(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));   // [n_idx_h, n_blocks, nc, n_stream]

        ggml_tensor * hw_cur = ggml_cont(ctx0, ggml_view_4d(ctx0, hw, n_idx_h, 1, nc, n_stream,
                hw->nb[1], hw->nb[2], hw->nb[3], i0*hw->nb[2]));
        score = ggml_mul(ctx0, score, hw_cur);
        score = ggml_sum_rows(ctx0, score);
        score = ggml_reshape_3d(ctx0, score, n_blocks, nc, n_stream);

        // one value per block, so it is cheaper to bias here than after the cells are expanded
        if (blk_bias) {
            ggml_tensor * bias_cur = ggml_cont(ctx0, ggml_view_3d(ctx0, inp->bias, n_blocks, nc, n_stream,
                    inp->bias->nb[1], inp->bias->nb[2], i0*inp->bias->nb[1]));
            score = ggml_add(ctx0, score, bias_cur);
        }

        // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
        ggml_tensor * expanded = ggml_get_rows(ctx0,
                ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
        expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));   // [n_kv, nc, n_stream]

        if (blk_bias) {
            // flash attention keeps the mask in f16; the scores are f32
            ggml_tensor * mask_cur = ggml_cont(ctx0, ggml_view_3d(ctx0, kqm3, n_kv, nc, n_stream,
                    kqm3->nb[1], kqm3->nb[2], i0*kqm3->nb[1]));
            if (mask_cur->type != GGML_TYPE_F32) {
                mask_cur = ggml_cast(ctx0, mask_cur, GGML_TYPE_F32);
            }
            expanded = ggml_add(ctx0, expanded, mask_cur);
        } else {
            ggml_tensor * bias_cur = ggml_cont(ctx0, ggml_view_3d(ctx0, inp->bias, n_kv, nc, n_stream,
                    inp->bias->nb[1], inp->bias->nb[2], i0*inp->bias->nb[1]));
            expanded = ggml_add(ctx0, expanded, bias_cur);
        }

        // build_attn_dsa reads [n_top_k, n_batch, 1, n_stream]; top-k per chunk, concat on the token axis
        ggml_tensor * tk = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));   // [width, nc, n_stream]
        top_k = top_k ? ggml_concat(ctx0, top_k, tk, 1) : tk;
    }

    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Weight-absorbed MLA restricted to the cells top_k names. Body copied from the K-only
// llm_graph_context::build_attn overload; the mask surgery is the DSA build_attn overload's.
// top_k == nullptr leaves the dense causal mask intact, so it is byte-identical to build_attn.
ggml_tensor * llama_model_glm5next::graph::build_attn_dsa(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) {
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache (K-only latent)
    {
        const auto & k_idxs = inp->get_k_idxs();
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    if (top_k != nullptr) {
        // prepare new kq mask - starts filled with -INFINITY
        ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

        // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
        kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

        // [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
        ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

        // zero source for the unmasked positions: [1, n_top_k, n_batch, n_stream]
        ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);

        ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

        // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
        kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

        // fold in the causal mask the indexer already saw
        kq_mask = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

//
// MLA attention (NoPE, q-compressed, weight-absorbed); ported from kimi-linear + deepseek2 q-compression
//

ggml_tensor * llama_model_glm5next::graph::build_mla(
        llm_graph_input_attn_k *                inp_attn_k,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        int                                     il) {
    const auto & layer = model.layers[il];

    const int64_t n_head            = hparams.n_head();
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const int64_t n_rot             = hparams.n_rot();                  // 0 (NoPE)
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_rot;      // == n_embd_head_k_mla
    const float   kq_scale_mla      = 1.0f / sqrtf((float) n_embd_head_k_mla);

    GGML_ASSERT(n_rot == 0);
    GGML_ASSERT(layer.wk_b && layer.wv_b);

    // LLAMA_GLM5NEXT_DSA_DENSE=1 skips the indexer and runs dense MLA (== phase-1); the default
    // engages the block-pooled sparse indexer. Set it to A/B the indexer against phase-1 numerics.
    static const bool dsa_dense = []() {
        const char * e = getenv("LLAMA_GLM5NEXT_DSA_DENSE");
        return e && atoi(e) != 0;
    }();

    // Q compression: wq_a -> RMS norm -> wq_b. The normed q-lora (q) is also the indexer's query.
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq_b, q);             // [n_head*n_embd_head_k_mla, n_tokens]

    // KV compression (no rope split since n_rot == 0)
    ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);   // [kv_lora_rank, n_tokens]
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

    // absorb q_nope into the latent space via wk_b
    ggml_tensor * q_nope = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);                    // [n_embd_head_qk_nope, n_tokens, n_head]

    ggml_tensor * q_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);  // [kv_lora_rank, n_tokens, n_head]
    q_absorbed = ggml_permute(ctx0, q_absorbed, 0, 2, 1, 3);            // [kv_lora_rank, n_head, n_tokens]
    // no q_pe to concat (NoPE): materialize the permuted view before attention
    Qcur = ggml_cont(ctx0, q_absorbed);
    cb(Qcur, "Qcur", il);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    ggml_tensor * Kcur = kv_cmpr;                                       // [kv_lora_rank, 1, n_tokens]
    ggml_tensor * Vcur = kv_cmpr;                                       // [kv_lora_rank, 1, n_tokens]

    GGML_UNUSED(n_embd_head_qk_nope);

    // DSA indexer: score the cached positions and keep only the top_k this query may attend to
    ggml_tensor * top_k = nullptr;
    const llama_kv_cache_context * mctx_idx = mctx_hyb ? mctx_hyb->get_idx() : nullptr;
    if (!dsa_dense && mctx_idx != nullptr && hparams.dsv4_compress_ratios[il] > 0) {
        top_k = build_dsa_top_k(mctx_hyb, cur, q, inp_attn_k->get_kq_mask(), il);
    }

    cur = build_attn_dsa(inp_attn_k, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, top_k, kq_scale_mla, il);
    cb(cur, "mla_out", il);

    return cur;
}

//
// decoder graph
//

llama_model_glm5next::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    cb(inp, "inp_embd", -1);

    // NoPE (rope.dimension_count == 0): no inp_pos.

    auto * inp_hybrid  = build_inp_mem_hybrid_k();
    auto * inp_rs      = inp_hybrid->get_recr();
    auto * inp_attn_k  = inp_hybrid->get_attn();

    // create_memory builds a 3-cache llama_memory_hybrid_idx for glm5next, so this downcast is
    // safe; get_idx() is the DSA indexer cache (null only if the GGUF carries no indexer tensors)
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp_hybrid->mctx);
    if (const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx()) {
        GGML_ASSERT(mctx_idx->get_n_kv() == mctx_hyb->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // recurrent (KDA) batch-consistency requirements
    GGML_ASSERT(ubatch.n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hc == 4);

    // expand the residual stream to hc parallel copies: [n_embd, 1, nt] -> [n_embd, hc, nt]
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        // === attention sub-block, wrapped in a hyper-connection ===
        cur = build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_kda(inp_rs, cur, il);
        } else {
            cur = build_mla(inp_attn_k, mctx_hyb, cur, il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        // === feed-forward sub-block, wrapped in a hyper-connection ===
        residual = inpL;
        cur = build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    hparams.n_expert, hparams.n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // select the output tokens after the loop (the residual carries the hc axis)
    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    // collapse the hc streams (no learned output head -> plain mean)
    cur = build_hc_mean(inpL);
    cb(cur, "hc_mean", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
