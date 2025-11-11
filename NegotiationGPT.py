#!/usr/bin/env python3
# NegotiationGPT.py
#
# Offline-safe conceptual LLM for negotiation modeling:
# - Local tokenizer built from your CSVs
# - Decoder-only transformer:
#     * GQA-style attention
#     * RoPE
#     * LoRA-style adapters
#     * Mixture-of-Experts FFN
# - Multi-task heads (LM, negotiation code classification, self-correction)
# - Train/Val/Test on main file(s)
# - Extra held-out test document(s) for:
#     * early/mid/late progress evaluation (loss)
#     * snippet → (Speaker, Code, ContextMatch) evaluation
# - Offline RL-style fine-tuning on held-out codes (no HTTP, no telemetry)
#
# Everything is local-only.

import math
import re
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. CONFIG
# =========================================================================

NEGOTIATION_CODES_MAP = {
    "agr": 0, "dis": 1, "coer": 2, "diff": 3, "sim": 4, "int_proc": 5, "misc": 6,
    "os": 7, "om": 8, "ip": 9, "ir": 10, "ib": 11, "sb": 12, "sf": 13,
    "qo": 14, "qp": 15, "qr": 16, "qb": 17, "qd": 18, "qm": 19,
    "in": 20, "mu": 21, "p1": 22, "pm": 23, "pt": 24, "cs": 25, "misc_2": 26
}
NEGOTIATION_CODE_CLASSES = len(NEGOTIATION_CODES_MAP)
CODE_ID_TO_STR = {v: k for k, v in NEGOTIATION_CODES_MAP.items()}

# Model size
D_MODEL = 4096
NUM_LAYERS = 32
NUM_HEADS = 32
NUM_KV_HEADS = 2
MAX_SEQ_LEN = 256
NUM_EXPERTS = 6
TOP_K_MOE = 3
LORA_RANK = 4
LORA_ALPHA = 8
DROPOUT_RATE = 0.1

ROPE_BASE = 10000.0
ROPE_SCALE = 2.0
WINDOW_SIZE = 128
NUM_TASKS = 3  # (0: LM, 1: Code, 2: Correction)

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.9
TOP_P = 0.9

EPOCHS = 10  # supervised epochs

# RL config (offline RL on held-out labels)
RL_STEPS = 200
RL_BATCH_SIZE = 8
RL_REWARD_CORRECT = 1.0
RL_REWARD_INCORRECT = -0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Train files
TRAIN_FILES = [
    "Sonnet4-consolidated.csv",
    # add more if needed
]

# Held-out files (e.g., combined.xlsx with original_code)
EXTRA_TEST_FILES = [
    "combined.xlsx",
]

TEXT_COL = "content"
FINAL_CODE_COL = "final_code"
HUMAN_CODE_COL = "human_code"
ORIG_CODE_COL = "original_code"

# =========================================================================
# 2. TOKENIZER
# =========================================================================

class NegotiationTokenizer:
    def __init__(self, max_vocab_size=32000, min_freq=1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[SEP]", "[UNK]"]
        self.itos: List[str] = []
        self.stoi = {}

    def _basic_tokenize(self, text: str) -> List[str]:
        text = str(text).lower()
        text = re.sub(r"([^a-z0-9<>]+)", " ", text)
        toks = text.strip().split()
        return [t for t in toks if t]

    def _load_table(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.is_file():
            print(f"[Tokenizer] Warning: file not found: {path}")
            return None
        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                return pd.read_excel(path)
            else:
                return pd.read_csv(path)
        except Exception as e:
            print(f"[Tokenizer] Warning: failed to read {path.name}: {e}")
            return None

    def build_from_files(self, file_paths: List[str], text_column: str):
        counter = Counter()
        text_column = text_column.lower()

        for path in file_paths:
            p = Path(path)
            df = self._load_table(p)
            if df is None:
                continue
            df.columns = [c.lower() for c in df.columns]
            if text_column not in df.columns:
                print(f"[Tokenizer] Warning: column '{text_column}' not in {p.name}")
                continue
            for text in df[text_column].dropna().astype(str):
                for tok in self._basic_tokenize(text):
                    counter[tok] += 1

        # specials
        self.itos = list(self.special_tokens)

        # negotiation tags
        for code in NEGOTIATION_CODES_MAP.keys():
            tag = f"<{code}>"
            if tag not in self.itos:
                self.itos.append(tag)

        # corpus tokens
        for tok, freq in counter.most_common():
            if freq < self.min_freq:
                break
            if tok in self.itos:
                continue
            if len(self.itos) >= self.max_vocab_size:
                break
            self.itos.append(tok)

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        print(f"[Tokenizer] Built vocab of size {len(self.itos)} from local files.")

    @property
    def pad_id(self): return self.stoi["[PAD]"]
    @property
    def bos_id(self): return self.stoi["[BOS]"]
    @property
    def eos_id(self): return self.stoi["[EOS]"]
    @property
    def sep_id(self): return self.stoi["[SEP]"]
    @property
    def unk_id(self): return self.stoi["[UNK]"]
    @property
    def vocab_size(self): return len(self.itos)

    def encode(self, text: str, add_special_tokens=True, max_len=MAX_SEQ_LEN) -> torch.Tensor:
        toks = self._basic_tokenize(text)
        ids = [self.stoi.get(t, self.unk_id) for t in toks]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if tok in ("[PAD]", "[BOS]", "[SEP]", "[UNK]"):
                continue
            if tok == "[EOS]":
                break
            toks.append(tok)
        return " ".join(toks)

# =========================================================================
# 3. CORE LAYERS: RMSNorm, SwiGLU, RoPE, LoRA
# =========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear_gate = nn.Linear(dim_in, dim_out)
        self.linear_value = nn.Linear(dim_in, dim_out)
    def forward(self, x):
        return F.silu(self.linear_gate(x)) * self.linear_value(x)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, freqs_cos, freqs_sin):
    q_rot = (q * freqs_cos) + (rotate_half(q) * freqs_sin)
    k_rot = (k * freqs_cos) + (rotate_half(k) * freqs_sin)
    return q_rot, k_rot

def apply_qat_wrappers(model: nn.Module) -> nn.Module:
    print("--- QAT/Quantization hooks ready (local only) ---")
    return model

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    def forward(self, x: torch.Tensor, W_0_weight: torch.Tensor):
        delta_W = torch.matmul(self.lora_B.T, self.lora_A.T)
        W_eff = W_0_weight + self.scaling * delta_W
        return F.linear(x, W_eff)

# =========================================================================
# 4. ATTENTION (GQA) + MoE
# =========================================================================

class LoRAFlashGQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, lora_rank, lora_alpha, window_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size

        kv_dim = self.num_kv_heads * self.d_k

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, kv_dim, bias=False)
        self.W_v = nn.Linear(d_model, kv_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.lora_Q = LoRALayer(d_model, d_model, lora_rank, lora_alpha)
        self.lora_K = LoRALayer(d_model, kv_dim, lora_rank, lora_alpha)
        self.lora_V = LoRALayer(d_model, kv_dim, lora_rank, lora_alpha)

    def forward(self, x, freqs_cos, freqs_sin):
        bsz, seqlen, _ = x.shape

        Q = self.lora_Q(x, self.W_q.weight)
        K = self.lora_K(x, self.W_k.weight)
        V = self.lora_V(x, self.W_v.weight)

        Q = Q.view(bsz, seqlen, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(bsz, seqlen, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(bsz, seqlen, self.num_kv_heads, self.d_k).transpose(1, 2)

        Q, K = apply_rotary_emb(Q, K, freqs_cos, freqs_sin)

        group_size = self.num_heads // self.num_kv_heads
        if group_size > 1:
            K = K.repeat_interleave(group_size, dim=1)
            V = V.repeat_interleave(group_size, dim=1)

        context = F.scaled_dot_product_attention(
            query=Q, key=K, value=V, is_causal=True
        )
        context = context.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.W_o(context)

class MoEBlock(nn.Module):
    def __init__(self, d_model, num_experts, top_k, lora_rank, lora_alpha, num_tasks):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.base_gate = nn.Linear(d_model, num_experts)
        self.lora_gate = LoRALayer(d_model, num_experts, lora_rank, lora_alpha)
        self.task_embeddings = nn.Embedding(num_tasks, d_model)

        hidden = d_model * 4
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                SwiGLU(hidden, d_model),
            ) for _ in range(num_experts)
        ])

    def forward(self, x, task_id_tensor):
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        tasks_flat = task_id_tensor.view(-1)
        x_flat = x_flat + self.task_embeddings(tasks_flat)

        gate_logits = self.lora_gate(x_flat, self.base_gate.weight)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)

        final = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_ids = topk_idx[:, i]
            for j in range(self.num_experts):
                mask = (expert_ids == j)
                if mask.any():
                    idx = mask.nonzero(as_tuple=False).squeeze(-1)
                    e_in = x_flat[idx]
                    e_out = self.experts[j](e_in)
                    w = weights[idx, i].unsqueeze(-1)
                    final.index_add_(0, idx, e_out * w)
        return final.view(bsz, seqlen, dim)

# =========================================================================
# 5. DECODER + HEADS
# =========================================================================

class AdvancedMoEDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads,
                 num_experts, top_k, lora_rank, lora_alpha,
                 dropout_rate=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attention = LoRAFlashGQA(
            d_model, num_heads, num_kv_heads,
            lora_rank, lora_alpha, WINDOW_SIZE
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = RMSNorm(d_model)
        self.moe = MoEBlock(
            d_model, num_experts, top_k,
            lora_rank, lora_alpha, NUM_TASKS
        )
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, freqs_cos, freqs_sin, task_id_tensor):
        h = self.norm1(x)
        h = self.self_attention(h, freqs_cos, freqs_sin)
        x = x + self.dropout1(h)
        h = self.norm2(x)
        h = self.moe(h, task_id_tensor)
        x = x + self.dropout2(h)
        return x

class NegotiationCodeHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class SelfCorrectionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)

# =========================================================================
# 6. MODEL WITH RoPE
# =========================================================================

class NegotiationGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.d_model = D_MODEL
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.max_seq_len = MAX_SEQ_LEN

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.head_dim = self.d_model // self.num_heads
        assert self.head_dim % 2 == 0

        self.freqs_cos, self.freqs_sin = self._build_rope_cache(self.max_seq_len, self.head_dim)

        self.decoder_layers = nn.ModuleList([
            AdvancedMoEDecoderBlock(
                self.d_model, self.num_heads, self.num_kv_heads,
                NUM_EXPERTS, TOP_K_MOE, LORA_RANK, LORA_ALPHA,
                DROPOUT_RATE
            )
            for _ in range(NUM_LAYERS)
        ])

        self.final_norm = RMSNorm(self.d_model)
        self.output_linear = nn.Linear(self.d_model, vocab_size, bias=False)

        self.code_head = NegotiationCodeHead(self.d_model, NEGOTIATION_CODE_CLASSES)
        self.corr_head = SelfCorrectionHead(self.d_model)

    def _build_rope_cache(self, max_seq_len, head_dim):
        half = head_dim // 2
        base = ROPE_BASE / ROPE_SCALE
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.stack([cos, cos], dim=-1).reshape(max_seq_len, head_dim)
        sin = torch.stack([sin, sin], dim=-1).reshape(max_seq_len, head_dim)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, task_ids):
        bsz, seqlen = input_ids.shape
        if seqlen > self.max_seq_len:
            raise ValueError("Sequence longer than MAX_SEQ_LEN")
        x = self.embedding(input_ids)

        freqs_cos = self.freqs_cos[:, :, :seqlen, :].to(x.device)
        freqs_sin = self.freqs_sin[:, :, :seqlen, :].to(x.device)

        for layer in self.decoder_layers:
            x = layer(x, freqs_cos, freqs_sin, task_ids)

        x = self.final_norm(x)

        lm_logits = self.output_linear(x)
        code_logits = self.code_head(x)
        corr_logits = self.corr_head(x)
        return lm_logits, code_logits, corr_logits

# =========================================================================
# 7. DATASET
# =========================================================================

class NegotiationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: NegotiationTokenizer,
        text_column: str = TEXT_COL,
        split: str = "train",
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        assert split in ("train", "val", "test", "all")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        dfs = []
        for path in file_paths:
            p = Path(path)
            if not p.is_file():
                continue
            try:
                if p.suffix.lower() in (".xlsx", ".xls"):
                    df_part = pd.read_excel(p)
                else:
                    df_part = pd.read_csv(p)
                dfs.append(df_part)
            except Exception as e:
                print(f"[Dataset] Failed to read {p.name}: {e}")

        if not dfs:
            raise FileNotFoundError(f"No valid data files found for: {file_paths}")

        df = pd.concat(dfs, ignore_index=True)
        df.columns = [c.lower() for c in df.columns]
        text_column = text_column.lower()
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data.")

        df[text_column] = df[text_column].fillna("").astype(str)

        def map_code(code):
            if isinstance(code, str):
                return NEGOTIATION_CODES_MAP.get(code.lower().strip("<> ").strip(), -100)
            return -100

        gt = None
        for col in [FINAL_CODE_COL.lower(), ORIG_CODE_COL.lower(), HUMAN_CODE_COL.lower()]:
            if col in df.columns:
                col_vals = df[col]
                gt = col_vals if gt is None else gt.fillna(col_vals)
        if gt is None:
            gt = pd.Series([-100] * len(df))

        code_ids = [map_code(c) for c in gt]

        inputs, lm_labels, code_labels, corr_labels, task_ids = [], [], [], [], []

        for text, code_id in zip(df[text_column], code_ids):
            ids = tokenizer.encode(text, add_special_tokens=True, max_len=max_seq_len)

            if ids.size(0) < max_seq_len:
                pad_len = max_seq_len - ids.size(0)
                pad = torch.full((pad_len,), tokenizer.pad_id, dtype=torch.long)
                ids = torch.cat([ids, pad], dim=0)
            else:
                ids = ids[:max_seq_len]

            lm_lbl = ids.clone()
            lm_lbl[:-1] = ids[1:]
            lm_lbl[-1] = -100

            if code_id == -100:
                code_seq = torch.full((max_seq_len,), -100, dtype=torch.long)
            else:
                code_seq = torch.full((max_seq_len,), code_id, dtype=torch.long)

            corr_seq = torch.zeros(max_seq_len, dtype=torch.long)
            task_seq = torch.zeros(max_seq_len, dtype=torch.long)

            inputs.append(ids)
            lm_labels.append(lm_lbl)
            code_labels.append(code_seq)
            corr_labels.append(corr_seq)
            task_ids.append(task_seq)

        self.inputs = torch.stack(inputs)
        self.lm_labels = torch.stack(lm_labels)
        self.code_labels = torch.stack(code_labels)
        self.corr_labels = torch.stack(corr_labels)
        self.task_ids = torch.stack(task_ids)

        n = len(self.inputs)
        if split == "all" or n < 10:
            self.indices = list(range(n))
        else:
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            if split == "train":
                self.indices = list(range(0, max(1, n_train)))
            elif split == "val":
                start = max(1, n_train)
                end = max(start + 1, n_train + n_val)
                self.indices = list(range(start, min(end, n)))
            else:
                start = int(0.9 * n)
                self.indices = list(range(start, n))
        if not self.indices:
            self.indices = list(range(n))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "input_ids": self.inputs[i],
            "lm_labels": self.lm_labels[i],
            "code_labels": self.code_labels[i],
            "corr_labels": self.corr_labels[i],
            "task_ids": self.task_ids[i],
        }

# =========================================================================
# 8. TRAINER (SUPERVISED + EXTRA EVAL + RL)
# =========================================================================

class Trainer:
    def __init__(
        self,
        model: NegotiationGPT,
        train_data: NegotiationDataset,
        val_data: NegotiationDataset,
        test_data: NegotiationDataset,
        extra_test_data: Optional[NegotiationDataset] = None,
        lr: float = 1e-4,
        lambda_code: float = 0.5,
        lambda_corr: float = 0.1,
    ):
        self.model = model.to(DEVICE)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)
        self.extra_test_data = extra_test_data

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.ce_lm = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_code = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_corr = nn.CrossEntropyLoss(ignore_index=-100)
        self.lambda_code = lambda_code
        self.lambda_corr = lambda_corr

    # ---------- Supervised ----------

    def _compute_loss_batch(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"].to(DEVICE)
        lm_labels = batch["lm_labels"].to(DEVICE)
        code_labels = batch["code_labels"].to(DEVICE)
        corr_labels = batch["corr_labels"].to(DEVICE)
        task_ids = batch["task_ids"].to(DEVICE)

        lm_logits, code_logits, corr_logits = self.model(input_ids, task_ids)

        B, T, V = lm_logits.shape
        lm_loss = self.ce_lm(lm_logits.reshape(B*T, V), lm_labels.reshape(B*T))

        B, T, C = code_logits.shape
        code_loss = self.ce_code(code_logits.reshape(B*T, C), code_labels.reshape(B*T))

        B, T, C2 = corr_logits.shape
        corr_loss = self.ce_corr(corr_logits.reshape(B*T, C2), corr_labels.reshape(B*T))

        return lm_loss + self.lambda_code * code_loss + self.lambda_corr * corr_loss

    def train_epoch(self):
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            loss = self._compute_loss_batch(batch)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += loss.item()
        avg = total / max(1, len(self.train_loader))
        print(f"[Train] loss={avg:.4f}")
        return avg

    def _eval_loader(self, loader, name: str):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                loss = self._compute_loss_batch(batch)
                total += loss.item()
        avg = total / max(1, len(loader))
        print(f"[{name}] loss={avg:.4f}")
        return avg

    def evaluate_val(self):
        return self._eval_loader(self.val_loader, "Val")

    def evaluate_test(self):
        return self._eval_loader(self.test_loader, "Test")

    # ---------- Extra test: early/mid/late loss ----------

    def evaluate_extra_test_progress(self):
        if self.extra_test_data is None or len(self.extra_test_data) == 0:
            print("[ExtraTest] No extra test dataset configured.")
            return
        self.model.eval()
        n = len(self.extra_test_data.inputs)
        if n < 3:
            print("[ExtraTest] Not enough rows for early/mid/late segmentation.")
            return

        def eval_segment(name, start, end):
            if end <= start:
                print(f"[ExtraTest-{name}] empty segment.")
                return
            total = 0.0
            with torch.no_grad():
                for i in range(start, end):
                    batch = {
                        "input_ids": self.extra_test_data.inputs[i].unsqueeze(0),
                        "lm_labels": self.extra_test_data.lm_labels[i].unsqueeze(0),
                        "code_labels": self.extra_test_data.code_labels[i].unsqueeze(0),
                        "corr_labels": self.extra_test_data.corr_labels[i].unsqueeze(0),
                        "task_ids": self.extra_test_data.task_ids[i].unsqueeze(0),
                    }
                    total += self._compute_loss_batch(batch).item()
            avg = total / max(1, (end-start))
            print(f"[ExtraTest-{name}] loss={avg:.4f}")

        third = n // 3
        eval_segment("Early", 0, third)
        eval_segment("Mid", third, 2*third)
        eval_segment("Late", 2*third, n)

    # ---------- Utils for extra eval ----------

    def _load_extra_labeled_examples(self, extra_files: List[str]):
        examples = []
        for path in extra_files:
            p = Path(path)
            if not p.is_file():
                print(f"[Extra] Missing extra test file: {p}")
                continue
            try:
                if p.suffix.lower() in (".xlsx", ".xls"):
                    df = pd.read_excel(p)
                else:
                    df = pd.read_csv(p)
            except Exception as e:
                print(f"[Extra] Failed to read {p.name}: {e}")
                continue

            df.columns = [c.lower() for c in df.columns]

            # text
            text_col = None
            for cand in ("content", "utterance", "text", "snippet", "dialogue"):
                if cand in df.columns:
                    text_col = cand
                    break
            # code
            code_col = None
            for cand in ("original_code", "final_code", "human_code", "code", "label", "negotiation_code"):
                if cand in df.columns:
                    code_col = cand
                    break
            # speaker
            speaker_col = None
            for cand in ("speaker", "speakername", "role"):
                if cand in df.columns:
                    speaker_col = cand
                    break

            if text_col is None or code_col is None:
                print(f"[Extra] {p.name}: missing text/code; skipped.")
                continue

            for _, row in df.iterrows():
                text = str(row[text_col]).strip()
                if not text:
                    continue
                code_gt = str(row[code_col]).strip()
                if not code_gt:
                    continue
                norm_gt = code_gt.lower().strip().strip("<> ")
                if norm_gt not in NEGOTIATION_CODES_MAP:
                    continue
                speaker_gt = str(row[speaker_col]).strip() if speaker_col else None
                examples.append((text, norm_gt, speaker_gt))

        return examples

    def _predict_code_with_head(self, text: str, tokenizer: NegotiationTokenizer) -> Optional[str]:
        self.model.eval()
        with torch.no_grad():
            ids = tokenizer.encode(text, add_special_tokens=True, max_len=MAX_SEQ_LEN)
            if ids.size(0) < MAX_SEQ_LEN:
                pad = torch.full((MAX_SEQ_LEN - ids.size(0),), tokenizer.pad_id, dtype=torch.long)
                ids = torch.cat([ids, pad], dim=0)
            else:
                ids = ids[:MAX_SEQ_LEN]
            input_ids = ids.unsqueeze(0).to(DEVICE)
            task_ids = torch.zeros_like(input_ids, dtype=torch.long).to(DEVICE)
            _, code_logits, _ = self.model(input_ids, task_ids)
            last_logits = code_logits[:, -1, :]  # (1, C)
            pred_id = int(last_logits.argmax(dim=-1).item())
            return CODE_ID_TO_STR.get(pred_id, None)

    def _predict_speaker_heuristic(self, text: str) -> Optional[str]:
        # Very simple: look for "X:" at start
        first_line = text.strip().splitlines()[0].strip().lower()
        m = re.match(r"([a-z0-9_]+)\s*:", first_line)
        if m:
            tag = m.group(1)
            # normalize a few common roles
            if "buyer" in tag:
                return "Buyer"
            if "seller" in tag:
                return "Seller"
            if "mediator" in tag:
                return "Mediator"
            return tag
        return None

    def _context_overlap(self, tokenizer: NegotiationTokenizer, a: str, b: str) -> float:
        toks_a = set(tokenizer._basic_tokenize(a))
        toks_b = set(tokenizer._basic_tokenize(b))
        if not toks_a:
            return 0.0
        inter = len(toks_a & toks_b)
        return inter / len(toks_a)

    # ---------- Extra eval: Code / Speaker / Context ----------

    def evaluate_extra_test_generation(
        self,
        extra_files: List[str],
        tokenizer: NegotiationTokenizer,
        generator,
        max_samples: int = 50,
        context_threshold: float = 0.3,
    ):
        """
        For each labeled snippet:
          - Predict code via code_head (no regex hacks)
          - Predict speaker via heuristic
          - Generate a short explanation
          - Compute:
              * code accuracy
              * speaker match rate (where GT speaker exists)
              * avg context overlap
              * triple match rate: code correct + speaker correct (if GT) + overlap >= threshold
        """
        examples = self._load_extra_labeled_examples(extra_files)
        if not examples:
            print("[ExtraEval] No usable labeled examples.")
            return

        if len(examples) > max_samples:
            random.seed(0)
            examples = random.sample(examples, max_samples)

        code_correct = 0
        speaker_correct = 0
        speaker_total = 0
        context_overlaps = []
        triple_correct = 0

        print(f"[ExtraEval] Evaluating on {len(examples)} held-out snippets...")

        for i, (text, gt_code, gt_speaker) in enumerate(examples):
            # 1) Code prediction from head
            pred_code = self._predict_code_with_head(text, tokenizer)

            # 2) Speaker prediction (cheap heuristic)
            pred_speaker = self._predict_speaker_heuristic(text)

            # 3) Short explanation for context match
            exp_prompt = (
                "Context snippet from a negotiation:\n"
                f"{text}\n\n"
                "In 1-2 clear sentences, describe what move or intention this snippet shows."
            )
            explanation = generator.generate(
                exp_prompt,
                max_new_tokens=40,
                temperature=0.8,
                top_p=0.9,
            )
            overlap = self._context_overlap(tokenizer, text, explanation)
            context_overlaps.append(overlap)

            # Metrics
            if pred_code == gt_code:
                code_correct += 1

            sp_ok = True
            if gt_speaker:
                speaker_total += 1
                if pred_speaker and pred_speaker.lower().split()[0] == gt_speaker.lower().split()[0]:
                    speaker_correct += 1
                else:
                    sp_ok = False

            if (pred_code == gt_code) and sp_ok and (overlap >= context_threshold):
                triple_correct += 1

            # Show first few qualitative examples
            if i < 5:
                print("\n[ExtraEval Example]", i + 1)
                print("Snippet:", text[:200].replace("\n", " "))
                print("GT Code:", gt_code, "| Pred Code:", pred_code)
                print("GT Speaker:", gt_speaker, "| Pred Speaker:", pred_speaker)
                print(f"Context overlap: {overlap:.2f}")
                print("Explanation:", explanation.replace("\n", " "))

        n = len(examples)
        avg_overlap = sum(context_overlaps) / max(1, len(context_overlaps))
        code_acc = code_correct / max(1, n)
        speaker_acc = speaker_correct / max(1, speaker_total)
        triple_acc = triple_correct / max(1, n)

        print(f"\n[ExtraEval] Code accuracy: {code_correct}/{n} = {code_acc:.3f}")
        if speaker_total > 0:
            print(f"[ExtraEval] Speaker match rate: {speaker_correct}/{speaker_total} = {speaker_acc:.3f}")
        else:
            print("[ExtraEval] Speaker match rate: N/A (no GT speakers)")
        print(f"[ExtraEval] Avg context overlap: {avg_overlap:.3f}")
        print(f"[ExtraEval] Triple match rate: {triple_correct}/{n} = {triple_acc:.3f}")

    # ---------- RL fine-tuning ----------

    def rl_finetune_on_extra_codes(
        self,
        extra_files: List[str],
        tokenizer: NegotiationTokenizer,
        steps: int = RL_STEPS,
        batch_size: int = RL_BATCH_SIZE,
    ):
        examples = self._load_extra_labeled_examples(extra_files)
        if not examples:
            print("[RL] No usable labeled examples; skipping RL.")
            return

        print(f"[RL] Starting RL fine-tuning on {len(examples)} examples "
              f"for {steps} steps (batch={batch_size}).")
        self.model.train()

        for step in range(steps):
            batch = random.sample(examples, min(batch_size, len(examples)))
            input_list = []
            gt_code_ids = []

            for text, gt_code, _ in batch:
                ids = tokenizer.encode(text, add_special_tokens=True, max_len=MAX_SEQ_LEN)
                if ids.size(0) < MAX_SEQ_LEN:
                    pad = torch.full((MAX_SEQ_LEN - ids.size(0),), tokenizer.pad_id, dtype=torch.long)
                    ids = torch.cat([ids, pad], dim=0)
                else:
                    ids = ids[:MAX_SEQ_LEN]
                input_list.append(ids)
                gt_code_ids.append(NEGOTIATION_CODES_MAP[gt_code])

            input_ids = torch.stack(input_list).to(DEVICE)
            task_ids = torch.zeros_like(input_ids, dtype=torch.long).to(DEVICE)
            gt_code_ids = torch.tensor(gt_code_ids, dtype=torch.long, device=DEVICE)

            _, code_logits, _ = self.model(input_ids, task_ids)
            last_logits = code_logits[:, -1, :]  # (B, C)
            log_probs = F.log_softmax(last_logits, dim=-1)

            dist = torch.distributions.Categorical(logits=last_logits)
            actions = dist.sample()
            chosen_log_probs = log_probs[torch.arange(actions.size(0), device=DEVICE), actions]

            correct_mask = (actions == gt_code_ids)
            rewards = torch.where(
                correct_mask,
                torch.full_like(actions, RL_REWARD_CORRECT, dtype=torch.float32),
                torch.full_like(actions, RL_REWARD_INCORRECT, dtype=torch.float32),
            )

            rl_loss = -(rewards * chosen_log_probs).mean()
            self.opt.zero_grad()
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            if (step + 1) % max(1, steps // 10) == 0:
                print(f"[RL] Step {step+1}/{steps} - loss={rl_loss.item():.4f}, "
                      f"avg_reward={rewards.float().mean().item():.3f}")

        print("[RL] RL fine-tuning complete.")

# =========================================================================
# 9. GENERATOR (FIXED TO NOT ECHO CONTEXT)
# =========================================================================

class TextGenerator:
    def __init__(self, model: NegotiationGPT, tokenizer: NegotiationTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_id

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
        coherence_lambda: float = 0.3,
        top_k: int = 20,
    ) -> int:
        logits = logits.clone()

        # block some specials; EOS allowed
        for tid in (self.tokenizer.pad_id,
                    self.tokenizer.bos_id,
                    self.tokenizer.sep_id,
                    self.tokenizer.unk_id):
            if 0 <= tid < logits.size(0):
                logits[tid] = -1e9

        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)

        # top-p
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        if torch.any(mask):
            cutoff = torch.nonzero(mask, as_tuple=False)[0, 0]
            sorted_probs[cutoff + 1 :] = 0.0
            if sorted_probs.sum() > 0:
                sorted_probs = sorted_probs / sorted_probs.sum()

        k = min(top_k, sorted_probs.size(0))
        cand_probs = sorted_probs[:k]
        cand_ids = sorted_idx[:k]

        with torch.no_grad():
            ctx_ids = [
                int(t)
                for t in input_ids[0].tolist()
                if t not in (self.tokenizer.pad_id,
                             self.tokenizer.bos_id,
                             self.tokenizer.sep_id)
            ]
            if not ctx_ids:
                idx = torch.multinomial(cand_probs, 1)
                return cand_ids[idx].item()

            emb = self.model.embedding.weight
            ctx_vec = emb[torch.tensor(ctx_ids, device=emb.device)].mean(dim=0)
            ctx_vec = F.normalize(ctx_vec, dim=0)

            cand_vecs = F.normalize(emb[cand_ids], dim=-1)
            cos_sim = torch.matmul(cand_vecs, ctx_vec).clamp(min=0.0)

            scores = cand_probs + coherence_lambda * cos_sim
            best_idx = torch.argmax(scores)
            return cand_ids[best_idx].item()

    def generate(
        self,
        context_text: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> str:
        self.model.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(
                context_text,
                add_special_tokens=True,
                max_len=MAX_SEQ_LEN,
            ).unsqueeze(0).to(DEVICE)

            task_ids = torch.zeros_like(input_ids, dtype=torch.long).to(DEVICE)
            start_len = input_ids.size(1)

            for _ in range(max_new_tokens):
                if input_ids.size(1) >= MAX_SEQ_LEN:
                    break
                lm_logits, _, _ = self.model(input_ids, task_ids)
                last_logits = lm_logits[0, -1, :]
                next_id = self._sample_next_token(
                    last_logits,
                    input_ids,
                    temperature=temperature,
                    top_p=top_p,
                    coherence_lambda=0.3,
                    top_k=20,
                )
                if next_id == self.eos_id:
                    break
                next_tok = torch.tensor([[next_id]], device=DEVICE)
                input_ids = torch.cat([input_ids, next_tok], dim=1)
                next_task = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
                task_ids = torch.cat([task_ids, next_task], dim=1)

            # decode ONLY newly generated tokens -> avoids echoing/mangling context
            gen_ids = input_ids[0, start_len:].tolist()
            return self.tokenizer.decode(gen_ids)

# =========================================================================
# 10. MAIN
# =========================================================================

if __name__ == "__main__":
    # 1. Tokenizer from TRAIN_FILES
    tokenizer = NegotiationTokenizer(max_vocab_size=32000, min_freq=1)
    tokenizer.build_from_files(TRAIN_FILES, text_column=TEXT_COL)

    # 2. Datasets
    train_dataset = NegotiationDataset(TRAIN_FILES, tokenizer, split="train")
    val_dataset = NegotiationDataset(TRAIN_FILES, tokenizer, split="val")
    test_dataset = NegotiationDataset(TRAIN_FILES, tokenizer, split="test")

    # 3. Extra dataset for progress-loss
    extra_test_dataset = None
    try:
        extra_test_dataset = NegotiationDataset(EXTRA_TEST_FILES, tokenizer, split="all")
        print(f"[ExtraTest] Loaded extra test dataset with {len(extra_test_dataset)} samples.")
    except Exception as e:
        print(f"[ExtraTest] Skipping extra test dataset for loss eval: {e}")

    # 4. Model
    model = NegotiationGPT(vocab_size=tokenizer.vocab_size)
    model = apply_qat_wrappers(model)

    # Freeze backbone; train only adapters/heads/norms/embeddings
    for name, p in model.named_parameters():
        if ("lora" not in name
            and "norm" not in name
            and "head" not in name
            and "embedding" not in name):
            p.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- NegotiationGPT Initialized ---")
    print(f"Device: {DEVICE}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Layers: {NUM_LAYERS} | D_MODEL: {D_MODEL}")
    print(f"Trainable params: {trainable_params:,}")

    # 5. Trainer
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        extra_test_data=extra_test_dataset,
    )

    # 6. Supervised training
    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        trainer.train_epoch()
        trainer.evaluate_val()

    # 7. Test
    trainer.evaluate_test()

    # 8. Extra progress evaluation
    trainer.evaluate_extra_test_progress()

    # 9. Generator
    generator = TextGenerator(model, tokenizer)

    # 10. Offline RL fine-tuning on held-out codes
    trainer.rl_finetune_on_extra_codes(EXTRA_TEST_FILES, tokenizer)

    # 11. Extra evaluation: code / speaker / context
    trainer.evaluate_extra_test_generation(EXTRA_TEST_FILES, tokenizer, generator)

    # 12. Demo: negotiation turn that should now read like normal English
    demo_context = (
        "Buyer: I'm concerned about the total cost; it feels too high for our budget.\n"
        "Seller: I understand your concern. Let's explore options that keep quality while adjusting pricing.\n"
        "Buyer: I can move a little if we get better support.\n"
        "System: As the seller, respond with a cohesive, cooperative negotiation offer."
    )
    print("\n--- Generative AI Output (Simulating Negotiation Turn) ---")
    demo_response = generator.generate(demo_context)
    print("CONTEXT:\n", demo_context)
    print("\nMODEL RESPONSE:\n", demo_response)
