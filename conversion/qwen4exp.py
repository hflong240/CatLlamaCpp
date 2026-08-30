from __future__ import annotations

import re
from typing import Any, Callable, Iterable, cast

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import ModelBase, TextModel
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel


class _Qwen4ExpMtpMixin:
    """MTP export for Qwen4Exp (Qwen3.8-Flash-Next).

    The checkpoint keeps the draft head under ``mtp.*``:

    * ``mtp.layers.{k}.*``   – a full trunk-style decoder block (attn + MoE + HC)
    * ``mtp.fc_embedding`` / ``mtp.fc_hidden`` – fused side-by-side into ``eh_proj``
    * ``mtp.pre_fc_norm_embedding`` / ``mtp.pre_fc_norm_hidden`` – enorm / hnorm
    * ``mtp.hyper_connection_mixer.*`` – per-block output mixer (nextn.hc_head_*)

    We extend ``block_count`` by the number of MTP layers and remap the HF names so
    the existing tensor_map handles them, following the _Qwen35MtpMixin pattern.
    """

    hparams: dict[str, Any]
    model_arch: gguf.MODEL_ARCH
    gguf_writer: gguf.GGUFWriter
    block_count: int
    tensor_map: gguf.TensorNameMap
    no_mtp: bool
    mtp_only: bool
    _original_block_count: int | None = None
    _mtp_fc_embedding: Tensor | None = None
    _mtp_fc_hidden: Tensor | None = None
    _mtp_fc_bid: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"]
        if not self.no_mtp:
            self.block_count += self.hparams.get("mtp_num_hidden_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        hparams = {**self.hparams, **self.hparams.get("text_config", {})}
        key = next((k for k in ["n_layers", "num_hidden_layers", "n_layer", "num_layers"] if k in hparams), None)
        type(self)._original_block_count = hparams.get(key)
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)  # ty: ignore[unresolved-attribute]

    @classmethod
    def filter_tensors(cls, item):
        assert cls._original_block_count is not None
        if (titem := TextModel.filter_tensors(item)) is None:
            return None
        name, gen = titem

        if name.startswith("model.mtp."):
            name = name.replace("model.", "", 1)

        if name.startswith("mtp."):
            if cls.no_mtp:
                return None
            n_layer = cls._original_block_count
            parts = name.split(".", 3)

            if len(parts) >= 4 and parts[1] == "layers" and parts[2].isdecimal():
                mtp_idx = int(parts[2])
                name = f"model.layers.{n_layer + mtp_idx}.{parts[3]}"
            elif len(parts) == 3:
                if parts[1] == "fc_embedding":
                    name = f"model.layers.{n_layer}.eh_proj_embedding.{parts[2]}"
                elif parts[1] == "fc_hidden":
                    name = f"model.layers.{n_layer}.eh_proj_hidden.{parts[2]}"
                elif parts[1] == "pre_fc_norm_embedding":
                    name = f"model.layers.{n_layer}.enorm.{parts[2]}"
                elif parts[1] == "pre_fc_norm_hidden":
                    name = f"model.layers.{n_layer}.hnorm.{parts[2]}"
                elif parts[1] == "hyper_connection_mixer":
                    name = f"model.layers.{n_layer}.hyper_connection_mixer.{parts[2]}"
        elif cls.mtp_only:
            keep = name in (
                "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
                "embed_tokens.weight", "norm.weight",
            )
            if not keep:
                return None

        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Fuse fc_embedding + fc_hidden → eh_proj (order-independent)
        if name.endswith(".eh_proj_embedding.weight"):
            self._mtp_fc_embedding = data_torch
            self._mtp_fc_bid = bid if bid is not None else 0
            if self._mtp_fc_hidden is not None:
                yield from self._fuse_eh_proj()
            return
        if name.endswith(".eh_proj_hidden.weight"):
            self._mtp_fc_hidden = data_torch
            if self._mtp_fc_embedding is not None:
                yield from self._fuse_eh_proj()
            return

        yield from super().modify_tensors(data_torch, name, bid)  # ty: ignore[unresolved-attribute]

    def _fuse_eh_proj(self) -> Iterable[tuple[str, Tensor]]:
        assert self._mtp_fc_embedding is not None and self._mtp_fc_hidden is not None
        # cat dim=1: the gguf framework transposes weights before writing, so dim=1 here
        # becomes [2*n_embd, n_embd] in the GGUF, matching the C++ create_tensor shape.
        eh_proj = torch.cat([self._mtp_fc_embedding, self._mtp_fc_hidden], dim=1)
        self._mtp_fc_embedding = None
        self._mtp_fc_hidden = None
        gguf_name = self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ, self._mtp_fc_bid, ".weight")
        yield (gguf_name, eh_proj)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()  # ty: ignore[unresolved-attribute]
        if self.no_mtp:
            return
        if (n := self.hparams.get("mtp_num_hidden_layers", 0)) > 0:
            self.gguf_writer.add_nextn_predict_layers(n)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)  # ty: ignore[unresolved-attribute]

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,                  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)    # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen4ExpMtpMixin, _Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    supports_mtp_export = True
    no_mtp = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only the shard names, so the table itself is never held
        self._ple_shards: dict[int, str] = {}
        self._ple_row_dim: int | None = None

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        self.gguf_writer.add_attention_compress_ratios(
            [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
        )

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        img = self.hparams.get("image_token_id")
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        if eos is None:
            raise ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")
        return int(eos)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return []

        if ".ngram_embedding.shard_" in name:
            return self._place_ple_shard(data_torch, name)

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            return [(self.map_tensor_name(name), data_torch.squeeze())]

        return super().modify_tensors(data_torch, name, bid)

    # the shards concatenate into a tensor of well over 100 GB
    # use LazyChunkedTensor here, a single shard resident at a time
    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]

        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1])

        if len(self._ple_shards) < n_parts:
            return []

        # the checkpoint may yield the shards in any order, the row order is by index
        shards = [self._ple_shards[i] for i in sorted(self._ple_shards)]
        rows = 0
        for shard in shards:
            shape = self.model_tensors[shard]().shape
            if int(shape[-1]) != self._ple_row_dim:
                raise ValueError(
                    f"PLE shard {shard} has row dim {int(shape[-1])}, expected {self._ple_row_dim}")
            rows += int(shape[0])

        table = gguf.LazyChunkedTensor(
            [self._load_ple_shard(shard) for shard in shards],
            shape=(rows, self._ple_row_dim),
            dtype=np.float32,
        )
        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        return [(gguf_name + ".weight", cast(Tensor, table))]

    def _load_ple_shard(self, name: str):
        def load() -> np.ndarray:
            from .base import LazyTorchTensor

            # a fresh lazy tensor every call, or to_eager() memoizes every shard
            eager = LazyTorchTensor.to_eager(self.model_tensors[name]())
            return eager.to(torch.float32).contiguous().numpy()
        return load

    def prepare_tensors(self):
        super().prepare_tensors()
        n_parts = self.hparams.get("split_ngram_parts", 0)
        if self._ple_shards and len(self._ple_shards) != n_parts:
            raise ValueError(
                f"got {len(self._ple_shards)} PLE embedding shards, expected {n_parts}"
            )


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
