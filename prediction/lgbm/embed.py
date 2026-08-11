"""Transformer session autoencoder encode path for buyer LGBM inference."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
DEFAULT_MAX_LEN = 64
DEFAULT_BATCH_SIZE = 256


class SessionDataset(Dataset):
    def __init__(self, features: Mapping[str, Sequence[str]], stoi: Mapping[str, int], max_len: int):
        self.samples: List[torch.Tensor] = []
        self.session_ids: List[str] = []
        for sid, seq in features.items():
            ids = [stoi[token] for token in list(seq)[:max_len] if token in stoi]
            if ids:
                self.samples.append(torch.tensor(ids, dtype=torch.long))
                self.session_ids.append(str(sid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.session_ids[idx]


def make_collate(pad_idx: int):
    def collate_sessions(batch):
        seqs, sids = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
        x = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
        return x, lengths, list(sids)

    return collate_sessions


class SessionTransformerAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        model_dim: int = 64,
        nhead: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = DEFAULT_MAX_LEN,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.model_dim = model_dim
        self.token_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len + 1, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output = nn.Linear(model_dim, vocab_size)

    def _embed_tokens(self, token_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device) + position_offset
        return (
            self.token_embedding(token_ids) * math.sqrt(self.model_dim)
            + self.position_embedding(positions).unsqueeze(0)
        )

    def encode(self, x: torch.Tensor):
        batch_size = x.size(0)
        token_padding_mask = x.eq(self.pad_idx)
        tokens = self._embed_tokens(x, position_offset=1)
        cls = self.cls_token.expand(batch_size, -1, -1) + self.position_embedding.weight[0].view(
            1, 1, -1
        )
        encoder_input = torch.cat([cls, tokens], dim=1)
        encoder_padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device),
                token_padding_mask,
            ],
            dim=1,
        )
        memory = self.encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
        return memory, encoder_padding_mask

    def forward(self, x: torch.Tensor, bos_idx: int) -> torch.Tensor:
        memory, memory_padding_mask = self.encode(x)
        decoder_input = torch.full_like(x, bos_idx)
        decoder_input[:, 1:] = x[:, :-1]
        decoder_padding_mask = x.eq(self.pad_idx)
        target = self._embed_tokens(decoder_input)
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output(decoded)

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        memory, _ = self.encode(x)
        return memory[:, 0]


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_transformer_bundle(path: Path | str, device: torch.device | None = None) -> dict:
    device = device or resolve_torch_device()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config") or {}
    stoi = payload["stoi"]
    pad_idx = int(config.get("pad_idx", stoi[PAD_TOKEN]))
    model = SessionTransformerAutoencoder(
        vocab_size=int(config.get("vocab_size", len(stoi))),
        pad_idx=pad_idx,
        model_dim=int(config.get("model_dim", 64)),
        nhead=int(config.get("num_heads", 4)),
        encoder_layers=int(config.get("num_encoder_layers", 2)),
        decoder_layers=int(config.get("num_decoder_layers", 2)),
        ffn_dim=int(config.get("ffn_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        max_len=int(config.get("max_len", DEFAULT_MAX_LEN)),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "stoi": stoi,
        "itos": payload.get("itos"),
        "config": config,
        "pad_idx": pad_idx,
        "device": device,
        "max_len": int(config.get("max_len", DEFAULT_MAX_LEN)),
        "model_dim": int(config.get("model_dim", 64)),
    }


@torch.no_grad()
def encode_sessions(
    token_features: Mapping[str, Sequence[str]],
    bundle: dict,
    session_ids: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, np.ndarray]:
    """Encode sessions to embedding vectors. Missing/empty → omitted."""
    model: SessionTransformerAutoencoder = bundle["model"]
    stoi = bundle["stoi"]
    max_len = bundle["max_len"]
    device = bundle["device"]
    model_dim = bundle["model_dim"]
    pad_idx = bundle["pad_idx"]

    if session_ids is None:
        features = dict(token_features)
    else:
        features = {str(sid): token_features.get(str(sid), []) for sid in session_ids}

    dataset = SessionDataset(features, stoi, max_len=max_len)
    result: Dict[str, np.ndarray] = {}
    if len(dataset) == 0:
        return result

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(pad_idx),
    )
    for x, _lengths, sids in loader:
        vectors = model.get_embeddings(x.to(device)).detach().cpu().numpy().astype(np.float32)
        for sid, vector in zip(sids, vectors):
            result[str(sid)] = vector

    # Sessions with empty token sequences stay absent; callers fill zeros if needed.
    _ = model_dim
    return result


def zero_embedding(model_dim: int = 64) -> np.ndarray:
    return np.zeros(model_dim, dtype=np.float32)
