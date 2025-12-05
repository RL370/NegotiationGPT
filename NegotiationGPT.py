#!/usr/bin/env python3
# NegotiationGPT_refactored.py
# Custom transformer with LoRA, proper initialization, and adaptive learning rates
# Splits data by conversation (transcript_name)
#
# KEY FEATURES FOR CONTEXTUAL, CONCISE, PERSONALIZED GENERATION:
# 1. Context-Aware Training: Each training sample includes 3 previous utterances
#    to teach the model to generate contextual responses
# 2. Length Penalty: Small penalty added to LM loss to encourage concise responses
# 3. Multi-Task Optimization: Composite score balances LM quality and negotiation code prediction
#
# ADVANCED NLP TECHNIQUES:
# 5. Label Smoothing (0.1): Prevents overconfidence, improves generalization
# 6. Focal Loss: Focuses training on hard-to-predict tokens (gamma=2.0)
# 7. Repetition Penalty (Training): Penalizes repetitive token patterns during training
# 8. Contrastive Decoding: Penalizes generic responses during generation
# 9. Diversity-Promoting Beam Search: Avoids repetitive beam candidates
# 10. Response Quality Filtering: Filters nonsensical/low-quality outputs

import argparse
import math
import re
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional
import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[Warning] Optuna not available. Run: pip install optuna")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train NegotiationGPT model')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
args = parser.parse_args()

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

# Emotion labels (Optional)
EMOTION_MAP = {
    "neutral": 0, "positive": 1, "negative": 2, "anxious": 3,
    "confident": 4, "frustrated": 5, "empathetic": 6
}
EMOTION_CLASSES = len(EMOTION_MAP)
EMOTION_ID_TO_STR = {v: k for k, v in EMOTION_MAP.items()}
USE_EMOTION_HEAD = False 
USE_OUTCOME_HEAD = False 

# -------------------------------------------------------------------------
# MODEL CONFIG - (REDUCED TO FIGHT OVERFITTING)
# -------------------------------------------------------------------------
D_MODEL = 256       # Reduced from 384
NUM_LAYERS = 2      # Reduced from 3
NUM_HEADS = 4       # Reduced from 6
FFN_DIM = 1024      # Reduced from 1536
MAX_SEQ_LEN = 256
DROPOUT_RATE = 0.3  # Increased from 0.1 to 0.3 to force generalization

# LoRA config (Must be defined even if not used to avoid NameError)
LORA_RANK = 8
LORA_ALPHA = 16

# -------------------------------------------------------------------------
# TRAINING CONFIG
# -------------------------------------------------------------------------
EPOCHS = args.epochs
BATCH_SIZE = 16 

# Adaptive Learning Rate config
BASE_LR = 1e-4  
EMBEDDING_LR_MULT = 0.1
ATTENTION_LR_MULT = 1.0
FFN_LR_MULT = 1.0
HEAD_LR_MULT = 3.0
LORA_LR_MULT = 2.0 
LAYER_DECAY = 0.9

WARMUP_EPOCHS = 2
MIN_LR_RATIO = 0.01

# Loss weights (Adjusted for Overfitting)
LAMBDA_LM = 1.0
LAMBDA_CODE = 0.5   # Reduced from 1.0 to prevent obsession with classification tags
LAMBDA_SPEAKER = 1.0
LAMBDA_CORR = 0.1

# -------------------------------------------------------------------------
# ADVANCED NLP CONFIG
# -------------------------------------------------------------------------
LABEL_SMOOTHING = 0.1           
USE_FOCAL_LOSS = False          
FOCAL_LOSS_GAMMA = 2.0          
REPETITION_PENALTY_TRAIN = 1.0 
USE_CONTRASTIVE_DECODING = True 

# Composite score weights
COMPOSITE_WEIGHT_CODE = 0.5     
COMPOSITE_WEIGHT_LM = 0.5       

# Contextual training config
CONTEXT_WINDOW_SIZE = 0  
USE_CONTEXTUAL_TRAINING = False 

# Generation config
MAX_NEW_TOKENS = 20  
TEMPERATURE = 0.7  
TOP_P = 0.85  
USE_GREEDY = False  
USE_BEAM_SEARCH = True  

# Length penalty
LENGTH_PENALTY_WEIGHT = 0.0 

# Sliding window config
SLIDING_WINDOW_SIZE = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data config 
DATA_FILES = ["Sonnet4-consolidated.csv",
              "negotiation_dataset_twospeaker_5000.csv",
              "negotiation_dataset_5000.csv",
              "negotiation_dataset_multispeaker_10000.csv",
              "negotiation_synthetic_300_conversations_claude.csv",
              "negotiation_2_speakers_claude.csv",
              "negotiation_3_plus_speakers_claude.csv"
              ] 
TEXT_COL = "content"
LABEL_COL = "final_code"

CHECKPOINT_DIR = "checkpoints"

# Optuna config
USE_OPTUNA = False 
OPTUNA_TRIALS = 20 
OPTUNA_TIMEOUT = None 
OPTUNA_STUDY_NAME = "negotiation_gpt_optimization"
OPTUNA_DB_PATH = "optuna_study.db"

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
                        tokens = self._tokenize(text)
                        counter.update(tokens)
            except Exception as e:
                print(f"[Tokenizer] Error reading {p}: {e}")

        for token, count in counter.most_common(self.max_vocab_size - len(self.stoi)):
            if count >= self.min_freq:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

        print(f"[Tokenizer] Vocab size: {self.vocab_size}")

    def _tokenize(self, text):
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def encode(self, text, max_length=None, add_special_tokens=True):
        tokens = self._tokenize(text)
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
        assert self.head_dim * num_heads == d_model

        # Standard linear layers instead of LoRA
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Initialize weights
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

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
# 6. MODEL
# =========================================================================

class NegotiationGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.d_model = D_MODEL
        self.num_layers = NUM_LAYERS

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
        self.corr_head = ClassificationHead(D_MODEL, 2)

        # Optional research-backed heads
        # Emotion classification (Van Kleef et al., 2004)
        if USE_EMOTION_HEAD:
            self.emotion_head = ClassificationHead(D_MODEL, EMOTION_CLASSES)

        # Outcome prediction for RLHF (Thompson & Hastie, 1990)
        if USE_OUTCOME_HEAD:
            self.outcome_head = nn.Linear(D_MODEL, 1)  # Regression: 0-1 quality score

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
        corr_logits = self.corr_head(x, attention_mask)

        # Optional heads
        outputs = [lm_logits, code_logits, corr_logits]

        if USE_EMOTION_HEAD:
            emotion_logits = self.emotion_head(x, attention_mask)
            outputs.append(emotion_logits)

        if USE_OUTCOME_HEAD:
            # Pool sequence for outcome prediction
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
                sum_embeddings = torch.sum(x * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = x.mean(dim=1)

            outcome_score = torch.sigmoid(self.outcome_head(pooled))
            outputs.append(outcome_score)

        return tuple(outputs) if len(outputs) > 3 else (lm_logits, code_logits, corr_logits)

# =========================================================================
# 7. DATASET - Split by conversation
# =========================================================================

class NegotiationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_column: str = TEXT_COL,
        label_column: str = LABEL_COL,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

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
        else:
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

        # Build samples with conversation context for better contextual training
        # KEY INNOVATION: Instead of training on isolated utterances, we train on
        # utterances WITH their conversation history. This teaches the model to:
        # 1. Generate responses that are relevant to what was said before
        # 2. Adapt tone and content based on negotiation flow
        # 3. Produce personalized responses rather than generic ones
        #
        # Format: "Buyer: prev1 | Seller: prev2 | Buyer: prev3 | Seller: [PREDICT THIS]"
        # The model sees context but only predicts the current response
        from collections import defaultdict
        conversations = defaultdict(list)
        for i in range(len(df)):
            conversations[transcript_names[i]].append({
                'text': texts[i],
                'code_id': code_ids[i],
                'speaker_id': speaker_ids[i],
                'index': i
            })

        self.samples = []

        for transcript_name, conv_items in conversations.items():
            for idx, item in enumerate(conv_items):
                # Build context: include previous N utterances
                context_start = max(0, idx - CONTEXT_WINDOW_SIZE)
                context_items = conv_items[context_start:idx]

                # Format context as: "Speaker1: text1 Speaker2: text2 ... [CURRENT]"
                context_parts = []
                for ctx_item in context_items:
                    speaker_label = "Buyer" if ctx_item['speaker_id'] == 0 else "Seller" if ctx_item['speaker_id'] == 1 else "Speaker"
                    context_parts.append(f"{speaker_label}: {ctx_item['text']}")

                # Current utterance (what we're predicting)
                current_text = item['text']
                current_speaker = "Buyer" if item['speaker_id'] == 0 else "Seller" if item['speaker_id'] == 1 else "Speaker"

                # Full input: context + current speaker marker
                if context_parts:
                    full_input = " | ".join(context_parts) + f" | {current_speaker}: {current_text}"
                else:
                    full_input = f"{current_speaker}: {current_text}"

                # Encode full context
                ids = tokenizer.encode(full_input, max_length=max_seq_len)

                # Create labels: only predict the CURRENT utterance, not the context
                # This forces the model to use context to predict the current response
                context_text = " | ".join(context_parts) if context_parts else ""
                if context_text:
                    context_ids = tokenizer.encode(context_text + f" | {current_speaker}:", max_length=max_seq_len)
                    context_len = len(context_ids)
                else:
                    context_len = len(tokenizer.encode(f"{current_speaker}:", max_length=max_seq_len))

                pad_len = max_seq_len - len(ids)
                attention_mask = [1] * len(ids) + [0] * pad_len
                ids = ids + [tokenizer.pad_token_id] * pad_len

                input_ids = torch.tensor(ids, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)

                # LM labels: only predict current utterance tokens, mask context
                lm_labels = input_ids.clone()
                lm_labels[:-1] = input_ids[1:]
                lm_labels[-1] = -100
                lm_labels[:context_len] = -100  # Don't predict context, only current response
                lm_labels[attention_mask == 0] = -100

                self.samples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'lm_labels': lm_labels,
                    'code_label': torch.tensor(item['code_id'], dtype=torch.long),
                    'speaker_label': torch.tensor(item['speaker_id'], dtype=torch.long),
                    'corr_label': torch.tensor(0, dtype=torch.long),
                    'text': current_text,
                    'transcript_name': transcript_name,
                    'has_context': len(context_parts) > 0,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# =========================================================================
# 8. TRAINER WITH ADAPTIVE LEARNING RATES
# =========================================================================

class Trainer:
    def __init__(self, model, train_data, val_data, test_data=None):
        self.model = model.to(DEVICE)
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=BATCH_SIZE, shuffle=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=BATCH_SIZE, shuffle=False
        ) if test_data else None

        param_groups = self._build_adaptive_param_groups()
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1)  # Increased to combat severe overfitting

        self.current_epoch = 0
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * EPOCHS
        self.warmup_steps = self.steps_per_epoch * WARMUP_EPOCHS
        self.current_step = 0

        # Use label smoothing for better generalization
        self.ce_lm = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
        self.ce_code = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
        self.ce_corr = nn.CrossEntropyLoss(ignore_index=-100)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_composite_score = 0.0  # Track best composite score across all tasks

    def focal_loss(self, logits, targets, gamma=FOCAL_LOSS_GAMMA, ignore_index=-100):
        """Focal loss to focus on hard examples.

        FL(p_t) = -(1 - p_t)^gamma * log(p_t)
        where p_t is the probability of the correct class
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
        p_t = torch.exp(-ce_loss)  # Probability of correct class
        focal_weight = (1 - p_t) ** gamma
        focal_loss = focal_weight * ce_loss

        # Mask out ignored indices
        mask = targets != ignore_index
        if mask.sum() > 0:
            return focal_loss[mask].mean()
        return torch.tensor(0.0, device=logits.device)

    def repetition_penalty_loss(self, logits, input_ids, lm_mask, penalty=REPETITION_PENALTY_TRAIN):
        """Add penalty for generating repetitive tokens during training."""
        batch_size, seq_len, vocab_size = logits.shape

        # For each position, penalize tokens that appeared recently
        penalty_loss = torch.tensor(0.0, device=logits.device)
        num_positions = 0

        for b in range(batch_size):
            for t in range(1, seq_len):
                if lm_mask[b, t]:
                    # Look back at previous tokens (window of 5)
                    lookback_start = max(0, t - 5)
                    prev_tokens = input_ids[b, lookback_start:t]

                    # Get probabilities for this position
                    probs = F.softmax(logits[b, t], dim=-1)

                    # Add penalty for tokens that appeared recently
                    for tok in prev_tokens.unique():
                        if tok != -100 and tok < vocab_size:
                            penalty_loss += probs[tok] * penalty
                            num_positions += 1

        if num_positions > 0:
            return penalty_loss / num_positions
        return penalty_loss

    def _build_adaptive_param_groups(self):
        param_groups = []

        embedding_params = []
        for name, param in self.model.named_parameters():
            if 'embedding' in name and param.requires_grad:
                embedding_params.append(param)
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

        head_params = []
        for name, param in self.model.named_parameters():
            if ('head' in name) and param.requires_grad and 'lm_head' not in name:
                head_params.append(param)
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': BASE_LR * HEAD_LR_MULT,
                'name': 'classification_heads'
            })

        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                already_added = False
                for pg in param_groups:
                    if any(p is param for p in pg['params']):
                        already_added = True
                        break
                if not already_added:
                    other_params.append(param)
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': BASE_LR,
                'name': 'other'
            })

        total_params = 0
        print("\n[Adaptive LR] Parameter groups:")
        for pg in param_groups:
            num_params = sum(p.numel() for p in pg['params'])
            total_params += num_params
            print(f"  {pg['name']}: {num_params:,} params, lr={pg['lr']:.2e}")
        print(f"  Total trainable: {total_params:,}")

        return param_groups

    def _update_lr(self):
        if self.current_step < self.warmup_steps:
            warmup_progress = self.current_step / max(1, self.warmup_steps)
            lr_factor = 0.1 + 0.9 * warmup_progress
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_factor = MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            base_lr = param_group.get('initial_lr', param_group['lr'])
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
            param_group['lr'] = base_lr * lr_factor

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_lm_loss = 0.0
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

            lm_logits, code_logits, corr_logits = self.model(input_ids, attention_mask)

            # LM loss with advanced techniques
            lm_mask = lm_labels != -100

            if USE_FOCAL_LOSS:
                # Use focal loss for LM to focus on hard-to-predict tokens
                lm_loss = self.focal_loss(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            else:
                lm_loss = self.ce_lm(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

            # Add repetition penalty during training to discourage repetitive outputs
            if REPETITION_PENALTY_TRAIN > 1.0:
                rep_penalty = self.repetition_penalty_loss(lm_logits, input_ids, lm_mask)
                lm_loss = lm_loss + 0.1 * rep_penalty  # Small weight for repetition penalty

            # Add length penalty to encourage concise responses
            num_predicted_tokens = lm_mask.sum().float()
            length_penalty = LENGTH_PENALTY_WEIGHT * (num_predicted_tokens / lm_mask.size(0))
            lm_loss = lm_loss + length_penalty

            code_mask = code_labels != -100

            code_loss = torch.tensor(0.0, device=DEVICE)

            if code_mask.any():
                code_loss = self.ce_code(code_logits[code_mask], code_labels[code_mask])

            loss = LAMBDA_LM * lm_loss + LAMBDA_CODE * code_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_code_loss += code_loss.item()
            num_batches += 1

            if code_mask.any():
                pred_codes = torch.argmax(code_logits, dim=1)
                train_code_correct += (pred_codes[code_mask] == code_labels[code_mask]).sum().item()
                train_code_total += code_mask.sum().item()

        return {
            'total_loss': total_loss / max(1, num_batches),
            'lm_loss': total_lm_loss / max(1, num_batches),
            'code_loss': total_code_loss / max(1, num_batches),
            'code_acc': train_code_correct / max(1, train_code_total),
        }

    def evaluate(self, loader, show_predictions=False):
        self.model.eval()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_code_loss = 0.0
        code_correct = 0
        code_total = 0
        num_batches = 0

        predictions = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                lm_labels = batch['lm_labels'].to(DEVICE)
                code_labels = batch['code_label'].to(DEVICE)

                lm_logits, code_logits, corr_logits = self.model(input_ids, attention_mask)

                lm_loss = self.ce_lm(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

                code_mask = code_labels != -100

                code_loss = torch.tensor(0.0, device=DEVICE)

                if code_mask.any():
                    code_loss = self.ce_code(code_logits[code_mask], code_labels[code_mask])

                loss = LAMBDA_LM * lm_loss + LAMBDA_CODE * code_loss
                total_loss += loss.item()
                total_lm_loss += lm_loss.item()
                total_code_loss += code_loss.item()
                num_batches += 1

                if code_mask.any():
                    pred_codes = torch.argmax(code_logits, dim=1)
                    code_correct += (pred_codes[code_mask] == code_labels[code_mask]).sum().item()
                    code_total += code_mask.sum().item()

                    if show_predictions:
                        for i in range(len(code_labels)):
                            if code_mask[i]:
                                pred = CODE_ID_TO_STR.get(pred_codes[i].item(), "?")
                                actual = CODE_ID_TO_STR.get(code_labels[i].item(), "?")
                                predictions.append((pred, actual))

        if show_predictions and predictions:
            print("\n  Predictions (pred -> actual):")
            for pred, actual in predictions[:10]:
                match = "Y" if pred == actual else "X"
                print(f"    {match} {pred} -> {actual}")

        avg_lm_loss = total_lm_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_lm_loss, 20))  # Cap at 20 to avoid overflow

        return {
            'loss': total_loss / max(1, num_batches),
            'lm_loss': avg_lm_loss,
            'code_loss': total_code_loss / max(1, num_batches),
            'perplexity': perplexity,
            'code_acc': code_correct / max(1, code_total),
        }

    def evaluate_val(self):
        results = self.evaluate(self.val_loader, show_predictions=True)
        return results

    def evaluate_test(self):
        if self.test_loader is None:
            return {'loss': 0.0, 'lm_loss': 0.0, 'code_loss': 0.0,
                    'perplexity': 0.0, 'code_acc': 0.0}
        results = self.evaluate(self.test_loader, show_predictions=True)
        return results

    def evaluate_sliding_window(self, dataset, window_size=SLIDING_WINDOW_SIZE):
        """Evaluate using sliding window over the validation conversation."""
        self.model.eval()

        results = []
        num_windows = max(1, len(dataset) - window_size + 1)

        correct = 0
        total = 0

        with torch.no_grad():
            for start_idx in range(0, len(dataset) - window_size + 1):
                window_samples = [dataset[i] for i in range(start_idx, start_idx + window_size)]

                # The last sample in the window is the one we predict
                target_sample = window_samples[-1]

                input_ids = target_sample['input_ids'].unsqueeze(0).to(DEVICE)
                attention_mask = target_sample['attention_mask'].unsqueeze(0).to(DEVICE)
                code_label = target_sample['code_label'].to(DEVICE)

                _, code_logits, _ = self.model(input_ids, attention_mask)
                pred = torch.argmax(code_logits, dim=1).item()

                if code_label.item() != -100:
                    if pred == code_label.item():
                        correct += 1
                    total += 1

                    results.append({
                        'window_start': start_idx,
                        'pred': CODE_ID_TO_STR.get(pred, "?"),
                        'actual': CODE_ID_TO_STR.get(code_label.item(), "?"),
                        'correct': pred == code_label.item()
                    })

        accuracy = correct / max(1, total)
        return accuracy, results

# =========================================================================
# 9. CHECKPOINT UTILITIES
# =========================================================================

def save_checkpoint(model, tokenizer, name):
    import os
    import json
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model_path = os.path.join(CHECKPOINT_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), model_path)

    vocab_path = os.path.join(CHECKPOINT_DIR, f"{name}_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"itos": tokenizer.itos, "stoi": tokenizer.stoi}, f)

    print(f"[Checkpoint] Saved to {model_path}")

# =========================================================================
# 10. OPTUNA OBJECTIVE FUNCTION
# =========================================================================

def optuna_objective(trial, train_df, val_df, test_df, tokenizer):
    """Objective function for Optuna hyperparameter optimization.

    Optimizes a composite score that balances two tasks:
    - Language Modeling (50% weight): Lower perplexity = better text generation
    - Code Classification (50% weight): Negotiation tactic prediction accuracy

    Returns:
        float: Composite score in range [0, 1] where higher is better
    """

    # Suggest hyperparameters
    global D_MODEL, NUM_LAYERS, NUM_HEADS, FFN_DIM, DROPOUT_RATE
    global LORA_RANK, LORA_ALPHA, BATCH_SIZE, BASE_LR
    global EMBEDDING_LR_MULT, ATTENTION_LR_MULT, FFN_LR_MULT, HEAD_LR_MULT, LORA_LR_MULT
    global LAYER_DECAY, WARMUP_EPOCHS, LAMBDA_LM, LAMBDA_CODE, LAMBDA_SPEAKER
    global EPOCHS

    # Model architecture
    d_model = trial.suggest_categorical('d_model', [256, 512, 768])
    num_layers = trial.suggest_int('num_layers', 3, 8)
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 12, 16])
    ffn_dim = trial.suggest_categorical('ffn_dim', [1024, 2048, 3072])
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)

    # LoRA
    lora_rank = trial.suggest_int('lora_rank', 4, 16)
    lora_alpha = trial.suggest_int('lora_alpha', 8, 32)

    # Training
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    base_lr = trial.suggest_float('base_lr', 1e-5, 1e-3, log=True)

    # Learning rate multipliers
    embedding_lr_mult = trial.suggest_float('embedding_lr_mult', 0.05, 0.3)
    attention_lr_mult = trial.suggest_float('attention_lr_mult', 0.5, 2.0)
    ffn_lr_mult = trial.suggest_float('ffn_lr_mult', 0.5, 2.0)
    head_lr_mult = trial.suggest_float('head_lr_mult', 1.0, 5.0)
    lora_lr_mult = trial.suggest_float('lora_lr_mult', 1.0, 3.0)
    layer_decay = trial.suggest_float('layer_decay', 0.8, 0.95)

    warmup_epochs = trial.suggest_int('warmup_epochs', 1, 5)

    # Loss weights
    lambda_lm = trial.suggest_float('lambda_lm', 0.5, 2.0)
    lambda_code = trial.suggest_float('lambda_code', 1.0, 3.0)

    # Update global variables
    D_MODEL = d_model
    NUM_LAYERS = num_layers
    NUM_HEADS = num_heads
    FFN_DIM = ffn_dim
    DROPOUT_RATE = dropout_rate
    LORA_RANK = lora_rank
    LORA_ALPHA = lora_alpha
    BATCH_SIZE = batch_size
    BASE_LR = base_lr
    EMBEDDING_LR_MULT = embedding_lr_mult
    ATTENTION_LR_MULT = attention_lr_mult
    FFN_LR_MULT = ffn_lr_mult
    HEAD_LR_MULT = head_lr_mult
    LORA_LR_MULT = lora_lr_mult
    LAYER_DECAY = layer_decay
    WARMUP_EPOCHS = warmup_epochs
    LAMBDA_LM = lambda_lm
    LAMBDA_CODE = lambda_code

    # Use fewer epochs for optimization
    EPOCHS = 20

    print(f"\n[Trial {trial.number}] Testing hyperparameters:")
    print(f"  d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")
    print(f"  batch_size={batch_size}, lr={base_lr:.2e}, lora_rank={lora_rank}")

    try:
        # Create datasets
        train_dataset = NegotiationDataset(train_df, tokenizer)
        val_dataset = NegotiationDataset(val_df, tokenizer)
        test_dataset = NegotiationDataset(test_df, tokenizer)

        # Create model
        model = NegotiationGPT(vocab_size=tokenizer.vocab_size)

        # Create trainer
        trainer = Trainer(model, train_dataset, val_dataset, test_dataset)

        # Training loop with pruning
        best_composite_score = 0.0
        best_val_metrics = None

        for epoch in range(EPOCHS):
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.evaluate_val()

            # Compute composite score that considers both tasks:
            # 1. Code accuracy (maximize)
            # 2. LM quality via inverse perplexity (maximize = lower perplexity is better)

            # Normalize perplexity: good models have PPL 10-50, excellent < 10
            # Convert to 0-1 score where lower perplexity = higher score
            ppl = val_metrics['perplexity']
            ppl_score = max(0.0, 1.0 - (ppl / 100.0))  # Normalize: PPL of 0 = 1.0, PPL of 100 = 0.0

            # Composite score with equal weights for each task
            composite_score = (
                0.5 * val_metrics['code_acc'] +      # Code classification
                0.5 * ppl_score                       # Language modeling quality
            )

            # Report composite score for pruning
            trial.report(composite_score, epoch)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_val_metrics = val_metrics

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}: Composite={composite_score:.4f} | Code={val_metrics['code_acc']:.4f} | PPL={val_metrics['perplexity']:.2f}")

        if best_val_metrics:
            print(f"[Trial {trial.number}] Best Composite Score: {best_composite_score:.4f}")
            print(f"  Code Acc: {best_val_metrics['code_acc']:.4f} | Perplexity: {best_val_metrics['perplexity']:.2f}")
        return best_composite_score

    except Exception as e:
        print(f"[Trial {trial.number}] Failed: {e}")
        raise optuna.TrialPruned()

# =========================================================================
# 11. MAIN
# =========================================================================

if __name__ == "__main__":
    # Load data from multiple files
    dfs = []
    for data_file in DATA_FILES:
        file_path = Path(data_file)
        if not file_path.is_file():
            print(f"[Warning] File not found: {data_file}")
            continue
        try:
            if file_path.suffix.lower() in (".xlsx", ".xls"):
                temp_df = pd.read_excel(data_file)
            else:
                temp_df = pd.read_csv(data_file)
            temp_df.columns = [c.lower() for c in temp_df.columns]
            dfs.append(temp_df)
            print(f"[Data] Loaded {len(temp_df)} samples from {data_file}")
        except Exception as e:
            print(f"[Error] Failed to load {data_file}: {e}")

    if not dfs:
        raise ValueError("No valid data files found!")

    # Normalize column names and add missing transcript_name before concatenating
    normalized_dfs = []
    for i, temp_df in enumerate(dfs):
        # Normalize column names: Content -> content, Original_Code -> final_code
        if 'original_code' in temp_df.columns and 'final_code' not in temp_df.columns:
            temp_df['final_code'] = temp_df['original_code']
        if 'human_code' in temp_df.columns and 'final_code' not in temp_df.columns:
            temp_df['final_code'] = temp_df['human_code']

        # Add transcript_name if missing (use filename + row groups)
        if 'transcript_name' not in temp_df.columns:
            # Create synthetic conversation IDs based on sequential groups of ~50 rows
            rows_per_conv = 50
            num_convs = (len(temp_df) + rows_per_conv - 1) // rows_per_conv
            conv_ids = []
            for conv_idx in range(num_convs):
                start_row = conv_idx * rows_per_conv
                end_row = min((conv_idx + 1) * rows_per_conv, len(temp_df))
                conv_ids.extend([f"{DATA_FILES[i]}_conv_{conv_idx:04d}"] * (end_row - start_row))
            temp_df['transcript_name'] = conv_ids

        normalized_dfs.append(temp_df)

    df = pd.concat(normalized_dfs, ignore_index=True)
    print(f"[Data] Total samples loaded: {len(df)}")

    # Get unique conversations
    conversations = df['transcript_name'].unique().tolist()
    print(f"\n[Data] Found {len(conversations)} conversations:")
    for conv in conversations:
        count = len(df[df['transcript_name'] == conv])
        print(f"  {conv}: {count} samples")

    # IMPROVED: Smart split that ensures at least 1 conversation in val and test
    # Use 80/10/10 split with minimum guarantees
    random.shuffle(conversations)
    total_convs = len(conversations)

    # Ensure at least 1 conversation in val and test
    if total_convs < 3:
        raise ValueError(f"Need at least 3 conversations, got {total_convs}")

    # Use 80/10/10 split with minimum of 1 for val and test
    val_size = max(1, int(0.1 * total_convs))
    test_size = max(1, int(0.1 * total_convs))
    train_size = total_convs - val_size - test_size

    # Ensure train_size is positive
    if train_size < 1:
        # Fall back to simple split: all but 2 for train, 1 for val, 1 for test
        train_size = total_convs - 2
        val_size = 1
        test_size = 1

    train_convs = conversations[:train_size]
    val_convs = conversations[train_size:train_size + val_size]
    test_convs = conversations[train_size + val_size:]

    print(f"\n[Split] Using {train_size}/{val_size}/{test_size} conversation split:")
    print(f"  Train: {len(train_convs)} conversations ({100*len(train_convs)/total_convs:.1f}%)")
    print(f"  Val: {len(val_convs)} conversations ({100*len(val_convs)/total_convs:.1f}%)")
    print(f"  Test: {len(test_convs)} conversations ({100*len(test_convs)/total_convs:.1f}%)")

    # Create dataframes for each split
    train_df = df[df['transcript_name'].isin(train_convs)]
    val_df = df[df['transcript_name'].isin(val_convs)]
    test_df = df[df['transcript_name'].isin(test_convs)]

    print(f"\n[Samples] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build tokenizer from all data files
    tokenizer = NegotiationTokenizer(max_vocab_size=8000, min_freq=2)
    tokenizer.build_from_files(DATA_FILES, text_column=TEXT_COL)

    # =========================================================================
    # OPTUNA HYPERPARAMETER OPTIMIZATION
    # =========================================================================
    if USE_OPTUNA and OPTUNA_AVAILABLE:
        print(f"\n{'='*60}")
        print("STARTING OPTUNA HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Trials: {OPTUNA_TRIALS}")
        print(f"Study: {OPTUNA_STUDY_NAME}")

        # Create or load study
        storage_url = f"sqlite:///{OPTUNA_DB_PATH}"
        study = optuna.create_study(
            study_name=OPTUNA_STUDY_NAME,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            sampler=TPESampler(seed=42)
        )

        # Optimize
        study.optimize(
            lambda trial: optuna_objective(trial, train_df, val_df, test_df, tokenizer),
            n_trials=OPTUNA_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            show_progress_bar=True
        )

        # Print results
        print(f"\n{'='*60}")
        print("OPTUNA OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best composite score: {study.best_value:.4f}")
        print(f"  (Composite = 0.5*Code_Acc + 0.5*LM_Quality)")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save best hyperparameters
        best_params_path = f"{CHECKPOINT_DIR}/best_hyperparameters.json"
        import os
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        print(f"\nBest hyperparameters saved to: {best_params_path}")

        # Update global config with best parameters
        D_MODEL = study.best_params['d_model']
        NUM_LAYERS = study.best_params['num_layers']
        NUM_HEADS = study.best_params['num_heads']
        FFN_DIM = study.best_params['ffn_dim']
        DROPOUT_RATE = study.best_params['dropout_rate']
        LORA_RANK = study.best_params['lora_rank']
        LORA_ALPHA = study.best_params['lora_alpha']
        BATCH_SIZE = study.best_params['batch_size']
        BASE_LR = study.best_params['base_lr']
        EMBEDDING_LR_MULT = study.best_params['embedding_lr_mult']
        ATTENTION_LR_MULT = study.best_params['attention_lr_mult']
        FFN_LR_MULT = study.best_params['ffn_lr_mult']
        HEAD_LR_MULT = study.best_params['head_lr_mult']
        LORA_LR_MULT = study.best_params['lora_lr_mult']
        LAYER_DECAY = study.best_params['layer_decay']
        WARMUP_EPOCHS = study.best_params['warmup_epochs']
        LAMBDA_LM = study.best_params['lambda_lm']
        LAMBDA_CODE = study.best_params['lambda_code']

        # Reset epochs to full training
        EPOCHS = 80

        print(f"\n{'='*60}")
        print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
        print(f"{'='*60}")

    # Create datasets
    train_dataset = NegotiationDataset(train_df, tokenizer)
    val_dataset = NegotiationDataset(val_df, tokenizer)
    test_dataset = NegotiationDataset(test_df, tokenizer)

    # Create model
    model = NegotiationGPT(vocab_size=tokenizer.vocab_size)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n--- NegotiationGPT (LoRA + Adaptive LR) ---")
    print(f"Device: {DEVICE}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset)

    # Track losses for plotting
    train_losses = []
    val_losses = []

    # Early stopping config
    early_stop_patience = 5  # Stop if no improvement for 5 epochs
    early_stop_counter = 0
    best_val_loss_early_stop = float('inf')

    # Training loop
    for epoch in range(EPOCHS):
        trainer.set_epoch(epoch)
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate_val()

        # Track losses for plotting
        train_losses.append(train_metrics['total_loss'])
        val_losses.append(val_metrics['loss'])

        # Sliding window evaluation on validation conversation
        sw_acc, sw_results = trainer.evaluate_sliding_window(val_dataset, SLIDING_WINDOW_SIZE)

        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} (LM:{train_metrics['lm_loss']:.3f} | Code:{train_metrics['code_loss']:.3f})")
        print(f"  Train Acc:  Code={train_metrics['code_acc']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} (LM:{val_metrics['lm_loss']:.3f} | Code:{val_metrics['code_loss']:.3f})")
        print(f"  Val Acc:    Code={val_metrics['code_acc']:.4f}")
        print(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")
        print(f"  Sliding Window Acc: {sw_acc:.4f} (window_size={SLIDING_WINDOW_SIZE})")

        # Compute composite score (same as Optuna)
        ppl_score = max(0.0, 1.0 - (val_metrics['perplexity'] / 100.0))
        composite_score = (
            0.5 * val_metrics['code_acc'] +
            0.5 * ppl_score
        )
        print(f"  Composite Score: {composite_score:.4f}")

        if composite_score > trainer.best_composite_score:
            print(f"  Best Composite: {trainer.best_composite_score:.4f} -> {composite_score:.4f} (saved)")
            trainer.best_composite_score = composite_score
            trainer.best_val_loss = val_metrics['loss']
            trainer.best_val_acc = val_metrics['code_acc']
            save_checkpoint(model, tokenizer, "best")

        # Early stopping based on validation loss
        if val_metrics['loss'] < best_val_loss_early_stop:
            best_val_loss_early_stop = val_metrics['loss']
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  Early stopping counter: {early_stop_counter}/{early_stop_patience}")

        if early_stop_counter >= early_stop_patience:
            print(f"\n[Early Stopping] No improvement for {early_stop_patience} epochs. Stopping training.")
            break

    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    test_metrics = trainer.evaluate_test()
    print(f"Test Loss: {test_metrics['loss']:.4f} (LM:{test_metrics['lm_loss']:.3f} | Code:{test_metrics['code_loss']:.3f})")
    print(f"Test Acc:  Code={test_metrics['code_acc']:.4f}")
    print(f"Test Perplexity: {test_metrics['perplexity']:.2f}")

    # Save final model
    save_checkpoint(model, tokenizer, "final")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    print("\n[Plot] Training and validation loss saved to loss_plot.png")
    plt.close()

    # Plot test accuracy
    plt.figure(figsize=(10, 6))
    test_acc = test_metrics['code_acc']
    plt.bar(['Test Accuracy'], [test_acc], color='green', alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.ylim([0, 1])
    plt.grid(True, axis='y')
    for i, v in enumerate([test_acc]):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    plt.savefig('test_accuracy_plot.png')
    print("[Plot] Test accuracy saved to test_accuracy_plot.png")
    plt.close()
