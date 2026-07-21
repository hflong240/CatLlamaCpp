#include "models.h"

#include "llama-moe-stream.h" // fork: cross-layer expert prefetch (llama_moe_layer_cache_lookup/_prefetch)

// Tencent Hy3 (hy_v3): standard GQA attention with q/k norms, a leading dense block,
// then DeepSeek-style MoE layers (sigmoid router + expert-selection bias, ungated shared
// expert). The trailing NextN/MTP block is loaded but not executed here (the base decoder
// only runs the first hparams.n_layer() blocks; MTP is a speculative-decoding draft head).

void llama_model_hy_v3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);

    // HY V3 uses a sigmoid router with expert selection bias by default
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // NextN/MTP: extra decoder block(s) appended beyond the main stack
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_all");

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_hy_v3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    auto load_block = [&](int i, int flags) {
        auto & layer = layers[i];
        const int64_t n_ff_exp_l   = hparams.n_ff_exp;
        const int64_t n_ff_shexp_l = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp_l;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, flags);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // dense FFN (present only in the leading dense block(s))
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);

        // MoE routed experts (sigmoid router + expert-selection bias)
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,           i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {n_ff_exp_l, n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp_l, n_expert, TENSOR_NOT_REQUIRED);

        // shared expert (always active, no gate)
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp_l}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp_l}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp_l, n_embd}, TENSOR_NOT_REQUIRED);
    };

    // trunk decoder blocks (executed)
    for (int i = 0; i < n_layer; ++i) {
        load_block(i, 0);
    }

    // NextN/MTP block(s): registered so the GGUF loads cleanly, but skipped (not executed)
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block(i, TENSOR_SKIP);
        auto & layer = layers[i];
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, TENSOR_SKIP);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, TENSOR_SKIP);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, TENSOR_SKIP);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED | TENSOR_SKIP);
    }
}

std::unique_ptr<llm_graph_context> llama_model_hy_v3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_hy_v3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_norm", il);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_norm", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // MoE routing-lookahead probe (fork, LLAMA_MOE_LOOKAHEAD_PROBE): predict layer il+1's expert
        // routing from THIS layer's just-computed normalized hidden `cur` (NeutronStar-style one-layer
        // lookahead). This is a self-contained pure-GPU subgraph (gate matmul + sigmoid + bias + top-k)
        // that feeds NOTHING downstream, so it cannot perturb real output. Tagged ffn_moe_topk_pred so
        // the debug eval callback dumps it; diff against the actual ffn_moe_topk-(il+1) to measure how
        // well one-layer-ahead prediction works on hy3 (the go/no-go for real lookahead prefetch).
        if (getenv("LLAMA_MOE_LOOKAHEAD_PROBE") && il + 1 < n_layer &&
            model.layers[il + 1].ffn_gate_inp != nullptr) {
            ggml_tensor * pl = build_lora_mm(model.layers[il + 1].ffn_gate_inp, cur); // [n_expert, n_tokens]
            pl = ggml_sigmoid(ctx0, pl);
            if (model.layers[il + 1].ffn_exp_probs_b != nullptr) {
                pl = ggml_add(ctx0, pl, model.layers[il + 1].ffn_exp_probs_b);
            }
            ggml_tensor * pred = ggml_argsort_top_k(ctx0, pl, n_expert_used); // [n_expert_used, n_tokens]
            cb(pred, "ffn_moe_topk_pred", il + 1); // tag with the PREDICTED layer's index
            ggml_build_forward_expand(gf, pred);   // keep it in the graph though nothing consumes it
        }

        // MoE cross-layer prefetch (fork, LLAMA_MOE_PREFETCH): on decode, predict layer il+1's routing
        // from THIS layer's normalized hidden `cur` (projected through il+1's gate_inp) and publish the
        // top-K candidates into il+1's cache so the background loader warms them during this layer's
        // compute - hiding il+1's disk read. Purely speculative: a wrong prediction only wastes a load,
        // never corrupts output (il+1 still runs its real selection + SYNC_BUDGET). Prediction is
        // approximate (one attention block separates cur from il+1's real gate input), so publish a WIDER
        // top-K than n_expert_used (LLAMA_MOE_PREFETCH_K, default 2x n_used) to raise the chance the real
        // top-2 are among the warmed candidates. Decode only (prefill streams everything anyway).
        if (n_tokens <= 1 && getenv("LLAMA_MOE_PREFETCH") && il + 1 < n_layer &&
            model.layers[il + 1].ffn_gate_inp != nullptr && model.layers[il + 1].ffn_gate_exps != nullptr) {
            int k = 2 * n_expert_used;
            if (const char * ke = getenv("LLAMA_MOE_PREFETCH_K")) { const int v = atoi(ke); if (v > 0) { k = v; } }
            if (k > n_expert)      { k = n_expert; }
            if (k < n_expert_used) { k = n_expert_used; }
            // il+1's cache is keyed by its first projection tensor (gate_exps, matching build_moe_ffn's
            // projs[] order gate/up/down). It exists by decode (created during prefill); null => skip.
            llama_moe_layer_cache * next_c = llama_moe_layer_cache_lookup(model.layers[il + 1].ffn_gate_exps);
            if (next_c) {
                ggml_tensor * pl = build_lora_mm(model.layers[il + 1].ffn_gate_inp, cur); // [n_expert, n_tokens]
                pl = ggml_sigmoid(ctx0, pl);
                if (model.layers[il + 1].ffn_exp_probs_b != nullptr) {
                    pl = ggml_add(ctx0, pl, model.layers[il + 1].ffn_exp_probs_b);
                }
                ggml_tensor * pred = ggml_argsort_top_k(ctx0, pl, k); // [k, n_tokens], highest-weight first
                llama_moe_layer_cache_prefetch(next_c, ctx0, gf, pred);
            }
        }

        if (model.layers[il].ffn_gate_inp == nullptr) {
            // dense FFN (leading dense block)
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE routed experts (sigmoid gating + expert-selection bias)
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert (always active, no gate)
            ggml_tensor * sh_out = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(sh_out, "ffn_shared_out", il);

            cur = ggml_add(ctx0, moe_out, sh_out);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
