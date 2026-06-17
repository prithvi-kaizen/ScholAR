"""
embedder.py — Local sentence embedder using cached all-MiniLM-L6-v2
====================================================================
Runs entirely offline using the model already cached in
~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/

No sentence-transformers or transformers package required.
Uses: torch (from anaconda), safetensors (from anaconda), numpy.

Architecture: BERT-base with 6 layers, 12 heads, hidden=384 (MiniLM variant)
Weights are loaded directly from safetensors using the exact HuggingFace key names.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# ── locate cached model ────────────────────────────────────────────────
_HF_CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
_MODEL_DIR_NAME = "models--sentence-transformers--all-MiniLM-L6-v2"
_SNAP_ROOT = _HF_CACHE / _MODEL_DIR_NAME / "snapshots"


def _find_snapshot() -> pathlib.Path:
    snaps = list(_SNAP_ROOT.iterdir())
    if not snaps:
        raise RuntimeError(f"No snapshot found under {_SNAP_ROOT}")
    return snaps[0]


# ── WordPiece tokenizer (pure Python) ────────────────────────────────
class _WordpieceTokenizer:
    def __init__(self, snap: pathlib.Path):
        tok_json = json.loads((snap / "tokenizer.json").read_text())
        raw_vocab = tok_json["model"]["vocab"]
        # vocab is {token_string: int_id}
        if isinstance(next(iter(raw_vocab.values())), int):
            self.vocab = {k: int(v) for k, v in raw_vocab.items()}
        else:
            self.vocab = {v: int(k) for k, v in raw_vocab.items()}

        self.unk_id  = self.vocab.get("[UNK]", 100)
        self.cls_id  = self.vocab.get("[CLS]", 101)
        self.sep_id  = self.vocab.get("[SEP]", 102)
        self.pad_id  = self.vocab.get("[PAD]", 0)
        self.max_len = 128

    def _wordpiece(self, word: str) -> list[int]:
        if word in self.vocab:
            return [self.vocab[word]]
        ids, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            ids.append(found)
            start = end
        return ids

    def tokenize(self, text: str, max_length: Optional[int] = None) -> dict:
        ml = max_length or self.max_len
        words = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9'\s]", text.lower())
        ids = [self.cls_id]
        for w in words:
            ids.extend(self._wordpiece(w))
            if len(ids) >= ml - 1:
                break
        ids.append(self.sep_id)
        ids = ids[:ml]
        pad = ml - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids += [self.pad_id] * pad
        return {"input_ids": ids, "attention_mask": mask}


# ── BERT model with exact HuggingFace parameter names ─────────────────
class _BertSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_attention_heads = cfg["num_attention_heads"]
        self.attention_head_size = cfg["hidden_size"] // cfg["num_attention_heads"]
        self.all_head_size       = cfg["hidden_size"]
        self.query = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.key   = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.value = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.dropout = nn.Dropout(0.0)

    def transpose_for_scores(self, x):
        B, S, _ = x.shape
        x = x.view(B, S, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        q = self.transpose_for_scores(self.query(x))
        k = self.transpose_for_scores(self.key(x))
        v = self.transpose_for_scores(self.value(x))
        scale = math.sqrt(self.attention_head_size)
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale
        if mask is not None:
            scores = scores + mask
        probs  = F.softmax(scores, dim=-1)
        probs  = self.dropout(probs)
        ctx    = torch.matmul(probs, v)
        ctx    = ctx.permute(0, 2, 1, 3).contiguous()
        B, S   = ctx.shape[:2]
        return ctx.view(B, S, self.all_head_size)


class _BertAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg.get("layer_norm_eps", 1e-12)
        self.self   = _BertSelfAttention(cfg)
        self.output = nn.ModuleDict({
            "dense":     nn.Linear(cfg["hidden_size"], cfg["hidden_size"]),
            "LayerNorm": nn.LayerNorm(cfg["hidden_size"], eps=eps),
        })
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, mask=None):
        self_out = self.self(x, mask)
        proj     = self.output["dense"](self_out)
        proj     = self.dropout(proj)
        return self.output["LayerNorm"](x + proj)


class _BertLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg.get("layer_norm_eps", 1e-12)
        self.attention    = _BertAttention(cfg)
        self.intermediate = nn.ModuleDict({
            "dense": nn.Linear(cfg["hidden_size"], cfg["intermediate_size"])
        })
        self.output       = nn.ModuleDict({
            "dense":     nn.Linear(cfg["intermediate_size"], cfg["hidden_size"]),
            "LayerNorm": nn.LayerNorm(cfg["hidden_size"], eps=eps),
        })
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, mask=None):
        x    = self.attention(x, mask)
        inter = F.gelu(self.intermediate["dense"](x))
        out   = self.output["dense"](inter)
        out   = self.dropout(out)
        return self.output["LayerNorm"](x + out)


class _BertEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.ModuleList([_BertLayer(cfg) for _ in range(cfg["num_hidden_layers"])])

    def forward(self, x, mask=None):
        for layer in self.layer:
            x = layer(x, mask)
        return x


class _BertEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg.get("layer_norm_eps", 1e-12)
        self.word_embeddings      = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], padding_idx=0)
        self.position_embeddings  = nn.Embedding(cfg.get("max_position_embeddings", 512), cfg["hidden_size"])
        self.token_type_embeddings= nn.Embedding(cfg.get("type_vocab_size", 2), cfg["hidden_size"])
        self.LayerNorm            = nn.LayerNorm(cfg["hidden_size"], eps=eps)
        self.dropout              = nn.Dropout(0.0)

        self.register_buffer(
            "position_ids",
            torch.arange(cfg.get("max_position_embeddings", 512)).expand((1, -1))
        )

    def forward(self, input_ids):
        seq_len  = input_ids.size(1)
        pos_ids  = self.position_ids[:, :seq_len]
        tok_type = torch.zeros_like(input_ids)
        emb = (self.word_embeddings(input_ids)
               + self.position_embeddings(pos_ids)
               + self.token_type_embeddings(tok_type))
        return self.dropout(self.LayerNorm(emb))


class _BertModel(nn.Module):
    """Pure-PyTorch BERT with exact HuggingFace parameter names for direct weight loading."""

    def __init__(self, cfg):
        super().__init__()
        self.embeddings = _BertEmbeddings(cfg)
        self.encoder    = _BertEncoder(cfg)

    def forward(self, input_ids, attention_mask=None):
        extended_mask = None
        if attention_mask is not None:
            # Convert 0/1 mask → additive large-negative mask (BERT convention)
            extended_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0

        x = self.embeddings(input_ids)
        x = self.encoder(x, extended_mask)

        # Mean pooling over non-padded positions
        if attention_mask is not None:
            mask = attention_mask[:, :, None].float()
            x = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            x = x.mean(1)

        # L2 normalise
        return F.normalize(x, p=2, dim=-1)


def _load_weights(path: pathlib.Path) -> dict[str, torch.Tensor]:
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    return state


class LocalEmbedder:
    """
    Sentence embedder backed by all-MiniLM-L6-v2 loaded from local HF cache.
    Returns L2-normalised 384-dim embeddings.
    No internet connection required.
    """

    def __init__(self):
        snap = _find_snapshot()
        cfg  = json.loads((snap / "config.json").read_text())
        self.tokenizer = _WordpieceTokenizer(snap)

        self.model = _BertModel(cfg)
        state = _load_weights(snap / "model.safetensors")

        # Load with strict=False — positional embeddings use different attribute name
        result = self.model.load_state_dict(state, strict=False)
        self._loaded  = len(state) - len(result.missing_keys)
        self._skipped = len(result.unexpected_keys)
        self._missing = result.missing_keys[:5]  # first 5 for debug

        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Returns (N, 384) float32 L2-normalised embedding matrix."""
        all_embs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks  = [self.tokenizer.tokenize(t) for t in batch]
            ids   = torch.tensor([t["input_ids"]      for t in toks], dtype=torch.long)
            mask  = torch.tensor([t["attention_mask"]  for t in toks], dtype=torch.long)
            with torch.no_grad():
                emb = self.model(ids, mask)
            all_embs.append(emb.numpy())
        return np.vstack(all_embs).astype(np.float32)
