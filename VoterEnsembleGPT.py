#!/usr/bin/env python3
# VoterEnsembleGPT_refactored.py
# 5-voter ensemble - each voter trained on one vote column (vote_1 to vote_5)
# Splits data by conversation (transcript_name)

import math
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict

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
NEGOTIATION_CODES = [CODE_ID_TO_STR[i] for i in range(NEGOTIATION_CODE_CLASSES)]

# Speaker labels
SPEAKER_MAP = {"buyer": 0, "seller": 1}
SPEAKER_CLASSES = len(SPEAKER_MAP)
SPEAKER_ID_TO_STR = {v: k for k, v in SPEAKER_MAP.items()}

# Model config
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
FFN_DIM = 2048
MAX_SEQ_LEN = 256
DROPOUT_RATE = 0.1

# LoRA config
LORA_RANK = 8
LORA_ALPHA = 16

# Training config
EPOCHS = 30
BATCH_SIZE = 8

# Adaptive Learning Rate config
BASE_LR = 3e-4
EMBEDDING_LR_MULT = 0.1
ATTENTION_LR_MULT = 1.0
FFN_LR_MULT = 1.0
HEAD_LR_MULT = 3.0
LORA_LR_MULT = 2.0
LAYER_DECAY = 0.9

WARMUP_EPOCHS = 2
MIN_LR_RATIO = 0.01

# Loss weights
LAMBDA_LM = 1.0
LAMBDA_CODE = 2.0
LAMBDA_SPEAKER = 1.0

# Ensemble config - 5 voters, one per vote column
NUM_VOTERS = 5
VOTER_LABEL_COLS = ["vote_1", "vote_2", "vote_3", "vote_4", "vote_5"]

# Sliding window config
SLIDING_WINDOW_SIZE = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data config - Only Sonnet4 CSV
DATA_FILE = "Sonnet4-consolidated.csv"
TEXT_COL = "content"

CHECKPOINT_DIR = "checkpoints"

# =========================================================================
# 2. TOKENIZER
# =========================================================================

class NegotiationTokenizer:
    def __init__(self, max_vocab_size=32000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.stoi = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.itos = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    @property
    def vocab_size(self):
        return len(self.stoi)

    def build_from_files(self, paths, text_column="content"):
        import re
        counter = Counter()
        for path in paths:
            p = Path(path)
            if not p.is_file():
                continue
            try:
                if p.suffix.lower() in (".xlsx", ".xls"):
                    df = pd.read_excel(p)
                else:
                    df = pd.read_csv(p)
                df.columns = [c.lower() for c in df.columns]
                text_column_lower = text_column.lower()
                if text_column_lower in df.columns:
                    for text in df[text_column_lower].dropna().astype(str):
                        tokens = re.findall(r"\w+|[^\w\s]", text.lower().strip())
                        counter.update(tokens)
            except Exception as e:
                print(f"[Tokenizer] Error reading {p}: {e}")

        for token, count in counter.most_common(self.max_vocab_size - len(self.stoi)):
            if count >= self.min_freq:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

        print(f"[Tokenizer] Vocab size: {self.vocab_size}")

    def encode(self, text, max_length=None, add_special_tokens=True):
        import re
        tokens = re.findall(r"\w+|[^\w\s]", text.lower().strip())
        ids = [self.stoi.get(t, self.unk_token_id) for t in tokens]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        special = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
        for idx in ids:
            if skip_special_tokens and idx in special:
                continue
            tokens.append(self.itos.get(idx, "<unk>"))
        return " ".join(tokens)

# =========================================================================
# 3. LORA LAYER
# =========================================================================

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        lora_result = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return result + lora_result

# =========================================================================
# 4. TRANSFORMER COMPONENTS
# =========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=MAX_SEQ_LEN, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x, seq_len):
        return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=DROPOUT_RATE):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = LoRALinear(d_model, d_model)
        self.k_proj = LoRALinear(d_model, d_model)
        self.v_proj = LoRALinear(d_model, d_model)
        self.o_proj = LoRALinear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=DROPOUT_RATE):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=DROPOUT_RATE):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# =========================================================================
# 5. CLASSIFICATION HEADS
# =========================================================================

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1, bias=False)
        )
        nn.init.xavier_uniform_(self.attention[0].weight)
        nn.init.zeros_(self.attention[0].bias)
        nn.init.zeros_(self.attention[2].weight)

    def forward(self, x, mask=None):
        attn_weights = self.attention(x)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled


class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=DROPOUT_RATE):
        super().__init__()
        self.attention_pool = AttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        pooled = self.attention_pool(x, mask)
        return self.classifier(pooled)

# =========================================================================
# 6. VOTER MODEL
# =========================================================================

class VoterModel(nn.Module):
    def __init__(self, vocab_size, voter_id=0):
        super().__init__()
        self.voter_id = voter_id
        self.d_model = D_MODEL
        self.num_layers = NUM_LAYERS

        torch.manual_seed(42 + voter_id)

        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(D_MODEL, NUM_HEADS, FFN_DIM, DROPOUT_RATE)
            for _ in range(NUM_LAYERS)
        ])

        self.final_norm = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.code_head = ClassificationHead(D_MODEL, NEGOTIATION_CODE_CLASSES)
        self.speaker_head = ClassificationHead(D_MODEL, SPEAKER_CLASSES)

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask.unsqueeze(0) * padding_mask

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.final_norm(x)

        lm_logits = self.lm_head(x)
        code_logits = self.code_head(x, attention_mask)
        speaker_logits = self.speaker_head(x, attention_mask)

        return lm_logits, code_logits, speaker_logits

# =========================================================================
# 7. DATASET - Split by conversation
# =========================================================================

class VoterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        label_column: str,
        text_column: str = TEXT_COL,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_column = label_column

        df.columns = [c.lower() for c in df.columns]
        text_column = text_column.lower()
        label_column = label_column.lower()

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data.")

        df[text_column] = df[text_column].fillna("").astype(str)

        # Map codes
        def map_code(code):
            if isinstance(code, str):
                return NEGOTIATION_CODES_MAP.get(code.lower().strip("<> ").strip(), -100)
            return -100

        if label_column in df.columns:
            code_ids = [map_code(c) for c in df[label_column]]
            valid_count = sum(1 for c in code_ids if c != -100)
            print(f"[Dataset] {label_column}: {valid_count}/{len(code_ids)} valid labels")
        else:
            print(f"[Dataset] Warning: {label_column} not found, using -100")
            code_ids = [-100] * len(df)

        # Map speakers
        def map_speaker(s):
            if not isinstance(s, str):
                return -100
            t = s.strip().lower()
            if "buyer" in t:
                return SPEAKER_MAP["buyer"]
            if "seller" in t:
                return SPEAKER_MAP["seller"]
            return -100

        speaker_ids = None
        for col in ("speakername", "speaker", "role"):
            if col in df.columns:
                speaker_ids = [map_speaker(v) for v in df[col]]
                break
        if speaker_ids is None:
            speaker_ids = [-100] * len(df)

        texts = df[text_column].tolist()

        # Get transcript names
        transcript_names = [""] * len(df)
        if "transcript_name" in df.columns:
            transcript_names = df["transcript_name"].fillna("").astype(str).tolist()

        self.samples = []
        for i in range(len(df)):
            text = texts[i]
            code_id = code_ids[i]
            speaker_id = speaker_ids[i]
            transcript_name = transcript_names[i]

            ids = tokenizer.encode(text, max_length=max_seq_len)
            pad_len = max_seq_len - len(ids)
            attention_mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [tokenizer.pad_token_id] * pad_len

            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

            lm_labels = input_ids.clone()
            lm_labels[:-1] = input_ids[1:]
            lm_labels[-1] = -100
            lm_labels[attention_mask == 0] = -100

            self.samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'lm_labels': lm_labels,
                'code_label': torch.tensor(code_id, dtype=torch.long),
                'speaker_label': torch.tensor(speaker_id, dtype=torch.long),
                'text': text,
                'transcript_name': transcript_name,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# =========================================================================
# 8. VOTER TRAINER WITH ADAPTIVE LR
# =========================================================================

class VoterTrainer:
    def __init__(self, model: VoterModel, train_data, val_data, voter_name="voter"):
        self.model = model.to(DEVICE)
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=BATCH_SIZE, shuffle=False
        )
        self.voter_name = voter_name

        param_groups = self._build_adaptive_param_groups()
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * EPOCHS
        self.warmup_steps = self.steps_per_epoch * WARMUP_EPOCHS
        self.current_step = 0

        self.ce_lm = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_code = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_speaker = nn.CrossEntropyLoss(ignore_index=-100)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def _build_adaptive_param_groups(self):
        param_groups = []

        embedding_params = [p for n, p in self.model.named_parameters()
                          if 'embedding' in n and p.requires_grad]
        if embedding_params:
            param_groups.append({
                'params': embedding_params,
                'lr': BASE_LR * EMBEDDING_LR_MULT,
                'name': 'embedding'
            })

        for layer_idx in range(self.model.num_layers):
            layer_lr = BASE_LR * (LAYER_DECAY ** (self.model.num_layers - layer_idx - 1))

            attn_params = []
            ffn_params = []
            lora_params = []

            for name, param in self.model.layers[layer_idx].named_parameters():
                if not param.requires_grad:
                    continue
                if 'lora' in name.lower():
                    lora_params.append(param)
                elif 'attention' in name or 'proj' in name:
                    attn_params.append(param)
                else:
                    ffn_params.append(param)

            if attn_params:
                param_groups.append({
                    'params': attn_params,
                    'lr': layer_lr * ATTENTION_LR_MULT,
                    'name': f'layer_{layer_idx}_attn'
                })
            if ffn_params:
                param_groups.append({
                    'params': ffn_params,
                    'lr': layer_lr * FFN_LR_MULT,
                    'name': f'layer_{layer_idx}_ffn'
                })
            if lora_params:
                param_groups.append({
                    'params': lora_params,
                    'lr': layer_lr * LORA_LR_MULT,
                    'name': f'layer_{layer_idx}_lora'
                })

        head_params = [p for n, p in self.model.named_parameters()
                      if 'head' in n and 'lm_head' not in n and p.requires_grad]
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': BASE_LR * HEAD_LR_MULT,
                'name': 'heads'
            })

        added = set()
        for pg in param_groups:
            for p in pg['params']:
                added.add(id(p))

        other_params = [p for p in self.model.parameters()
                       if p.requires_grad and id(p) not in added]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': BASE_LR,
                'name': 'other'
            })

        return param_groups

    def _update_lr(self):
        if self.current_step < self.warmup_steps:
            warmup_progress = self.current_step / max(1, self.warmup_steps)
            lr_factor = 0.1 + 0.9 * warmup_progress
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_factor = MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
            param_group['lr'] = param_group['initial_lr'] * lr_factor

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_code_loss = 0.0
        num_batches = 0

        train_code_correct = 0
        train_code_total = 0

        for batch in self.train_loader:
            self._update_lr()
            self.current_step += 1

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            lm_labels = batch['lm_labels'].to(DEVICE)
            code_labels = batch['code_label'].to(DEVICE)
            speaker_labels = batch['speaker_label'].to(DEVICE)

            lm_logits, code_logits, speaker_logits = self.model(input_ids, attention_mask)

            lm_loss = self.ce_lm(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

            code_mask = code_labels != -100
            speaker_mask = speaker_labels != -100

            code_loss = torch.tensor(0.0, device=DEVICE)
            speaker_loss = torch.tensor(0.0, device=DEVICE)

            if code_mask.any():
                code_loss = self.ce_code(code_logits[code_mask], code_labels[code_mask])
            if speaker_mask.any():
                speaker_loss = self.ce_speaker(speaker_logits[speaker_mask], speaker_labels[speaker_mask])

            loss = LAMBDA_LM * lm_loss + LAMBDA_CODE * code_loss + LAMBDA_SPEAKER * speaker_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_code_loss += code_loss.item()
            num_batches += 1

            if code_mask.any():
                pred_codes = torch.argmax(code_logits, dim=1)
                train_code_correct += (pred_codes[code_mask] == code_labels[code_mask]).sum().item()
                train_code_total += code_mask.sum().item()

        return {
            'loss': total_loss / max(1, num_batches),
            'code_loss': total_code_loss / max(1, num_batches),
            'code_acc': train_code_correct / max(1, train_code_total)
        }

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        code_correct = 0
        code_total = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                lm_labels = batch['lm_labels'].to(DEVICE)
                code_labels = batch['code_label'].to(DEVICE)
                speaker_labels = batch['speaker_label'].to(DEVICE)

                lm_logits, code_logits, speaker_logits = self.model(input_ids, attention_mask)

                lm_loss = self.ce_lm(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

                code_mask = code_labels != -100
                speaker_mask = speaker_labels != -100

                code_loss = torch.tensor(0.0, device=DEVICE)
                speaker_loss = torch.tensor(0.0, device=DEVICE)

                if code_mask.any():
                    code_loss = self.ce_code(code_logits[code_mask], code_labels[code_mask])
                if speaker_mask.any():
                    speaker_loss = self.ce_speaker(speaker_logits[speaker_mask], speaker_labels[speaker_mask])

                loss = LAMBDA_LM * lm_loss + LAMBDA_CODE * code_loss + LAMBDA_SPEAKER * speaker_loss
                total_loss += loss.item()
                num_batches += 1

                if code_mask.any():
                    pred_codes = torch.argmax(code_logits, dim=1)
                    code_correct += (pred_codes[code_mask] == code_labels[code_mask]).sum().item()
                    code_total += code_mask.sum().item()

        return {
            'loss': total_loss / max(1, num_batches),
            'code_acc': code_correct / max(1, code_total)
        }

# =========================================================================
# 9. ENSEMBLE - Majority voting across 5 voters
# =========================================================================

class VoterEnsemble:
    def __init__(self, voters: List[VoterModel]):
        self.voters = voters

    def predict(self, input_ids, attention_mask):
        all_code_logits = []

        with torch.no_grad():
            for voter in self.voters:
                voter.eval()
                _, code_logits, _ = voter(input_ids, attention_mask)
                all_code_logits.append(code_logits)

        ensemble_logits = torch.stack(all_code_logits, dim=0).mean(dim=0)
        predictions = torch.argmax(ensemble_logits, dim=1)

        return predictions, ensemble_logits

    def predict_majority_vote(self, input_ids, attention_mask):
        all_predictions = []

        with torch.no_grad():
            for voter in self.voters:
                voter.eval()
                _, code_logits, _ = voter(input_ids, attention_mask)
                preds = torch.argmax(code_logits, dim=1)
                all_predictions.append(preds)

        all_predictions = torch.stack(all_predictions, dim=0)

        batch_size = all_predictions.shape[1]
        final_predictions = []
        for i in range(batch_size):
            votes = all_predictions[:, i].tolist()
            vote_counts = Counter(votes)
            majority = vote_counts.most_common(1)[0][0]
            final_predictions.append(majority)

        return torch.tensor(final_predictions, device=input_ids.device)

    def evaluate(self, dataloader, use_majority_vote=True):
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            code_labels = batch['code_label'].to(DEVICE)

            if use_majority_vote:
                predictions = self.predict_majority_vote(input_ids, attention_mask)
            else:
                predictions, _ = self.predict(input_ids, attention_mask)

            mask = code_labels != -100
            if mask.any():
                correct += (predictions[mask] == code_labels[mask]).sum().item()
                total += mask.sum().item()

        return correct / max(1, total)

    def evaluate_sliding_window(self, dataset, window_size=SLIDING_WINDOW_SIZE, use_majority_vote=True):
        """Evaluate using sliding window."""
        correct = 0
        total = 0

        for voter in self.voters:
            voter.eval()

        with torch.no_grad():
            for start_idx in range(0, len(dataset) - window_size + 1):
                target_sample = dataset[start_idx + window_size - 1]

                input_ids = target_sample['input_ids'].unsqueeze(0).to(DEVICE)
                attention_mask = target_sample['attention_mask'].unsqueeze(0).to(DEVICE)
                code_label = target_sample['code_label'].to(DEVICE)

                if use_majority_vote:
                    pred = self.predict_majority_vote(input_ids, attention_mask).item()
                else:
                    pred = self.predict(input_ids, attention_mask)[0].item()

                if code_label.item() != -100:
                    if pred == code_label.item():
                        correct += 1
                    total += 1

        return correct / max(1, total)

# =========================================================================
# 10. CHECKPOINT
# =========================================================================

def save_checkpoint(model, name):
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[Checkpoint] Saved {model_path}")

# =========================================================================
# 11. MAIN
# =========================================================================

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.lower() for c in df.columns]

    # Get unique conversations
    conversations = df['transcript_name'].unique().tolist()
    print(f"\n[Data] Found {len(conversations)} conversations:")
    for conv in conversations:
        count = len(df[df['transcript_name'] == conv])
        print(f"  {conv}: {count} samples")

    # Split by conversation: 4 train, 1 val (sliding window), 1 test
    train_convs = conversations[:-2]
    val_conv = conversations[-2]
    test_conv = conversations[-1]

    print(f"\n[Split]")
    print(f"  Train: {train_convs}")
    print(f"  Val (sliding window): {val_conv}")
    print(f"  Test: {test_conv}")

    # Create dataframes for each split
    train_df = df[df['transcript_name'].isin(train_convs)]
    val_df = df[df['transcript_name'] == val_conv]
    test_df = df[df['transcript_name'] == test_conv]

    print(f"\n[Samples] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build tokenizer
    tokenizer = NegotiationTokenizer(max_vocab_size=32000, min_freq=2)
    tokenizer.build_from_files([DATA_FILE], text_column=TEXT_COL)

    print(f"\n--- 5-Voter Ensemble (one per vote column) ---")
    print(f"Device: {DEVICE}")
    print(f"Voter columns: {VOTER_LABEL_COLS}")

    # Create voters and their datasets
    voters = []
    trainers = []

    for i, label_col in enumerate(VOTER_LABEL_COLS):
        print(f"\n[Voter {i}] Creating dataset from '{label_col}'...")

        train_dataset = VoterDataset(train_df.copy(), tokenizer, label_column=label_col)
        val_dataset = VoterDataset(val_df.copy(), tokenizer, label_column=label_col)

        voter = VoterModel(vocab_size=tokenizer.vocab_size, voter_id=i)
        trainer = VoterTrainer(voter, train_dataset, val_dataset, voter_name=f"Voter-{i} ({label_col})")

        voters.append(voter)
        trainers.append(trainer)

    # Train each voter
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        for i, (trainer, label_col) in enumerate(zip(trainers, VOTER_LABEL_COLS)):
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.evaluate()

            print(f"[{label_col}] Train: {train_metrics['loss']:.3f} | Acc: {train_metrics['code_acc']:.3f} | Val: {val_metrics['loss']:.3f} | Acc: {val_metrics['code_acc']:.3f}")

            if val_metrics['loss'] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics['loss']
                trainer.best_val_acc = val_metrics['code_acc']
                save_checkpoint(trainer.model, f"voter_{i}_{label_col}_best")

    # Evaluate ensemble on test set using Final_Code as ground truth
    print(f"\n{'='*60}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*60}")

    # Create test dataset with Final_Code as ground truth
    test_dataset = VoterDataset(test_df.copy(), tokenizer, label_column="final_code")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ensemble = VoterEnsemble(voters)

    # Evaluate with different methods
    avg_acc = ensemble.evaluate(test_loader, use_majority_vote=False)
    majority_acc = ensemble.evaluate(test_loader, use_majority_vote=True)

    # Sliding window evaluation
    val_dataset_final = VoterDataset(val_df.copy(), tokenizer, label_column="final_code")
    sw_acc = ensemble.evaluate_sliding_window(val_dataset_final, SLIDING_WINDOW_SIZE)

    print(f"\nTest Accuracy (vs Final_Code):")
    print(f"  Averaged Logits: {avg_acc:.4f}")
    print(f"  Majority Vote:   {majority_acc:.4f}")
    print(f"\nSliding Window Accuracy (val set, window={SLIDING_WINDOW_SIZE}): {sw_acc:.4f}")

    # Individual voter accuracies
    print("\nIndividual Voter Accuracies (vs Final_Code):")
    for i, (trainer, label_col) in enumerate(zip(trainers, VOTER_LABEL_COLS)):
        trainer.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                code_labels = batch['code_label'].to(DEVICE)

                _, code_logits, _ = trainer.model(input_ids, attention_mask)
                preds = torch.argmax(code_logits, dim=1)

                mask = code_labels != -100
                if mask.any():
                    correct += (preds[mask] == code_labels[mask]).sum().item()
                    total += mask.sum().item()

        acc = correct / max(1, total)
        print(f"  {label_col}: {acc:.4f}")

    # Save final models
    for i, (voter, label_col) in enumerate(zip(voters, VOTER_LABEL_COLS)):
        save_checkpoint(voter, f"voter_{i}_{label_col}_final")

    print(f"\n[Done] 5-voter ensemble training complete!")
