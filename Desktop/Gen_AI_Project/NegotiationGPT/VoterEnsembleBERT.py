#!/usr/bin/env python3
# VoterEnsembleBERT.py
# BERT/RoBERTa-based ensemble with LoRA/DoRA fine-tuning
# 5-voter ensemble - each voter trained on one vote column (vote_1 to vote_5)
# Splits data by conversation (transcript_name)

import math
import random
from collections import Counter
from typing import List, Dict, Optional
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    RobertaModel, RobertaTokenizer, RobertaConfig,
    BertModel, BertTokenizer, BertConfig
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)
try:
    from peft import DoRAConfig
    DORA_AVAILABLE = True
except ImportError:
    DORA_AVAILABLE = False
    print("[Warning] DoRA not available. Using LoRA only.")

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

# Model config - BERT/RoBERTa
BASE_MODEL_NAME = "roberta-base"  # Options: "bert-base-uncased", "roberta-base", "distilbert-base-uncased"
USE_DORA = False  # Set to True to use DoRA instead of LoRA
MAX_SEQ_LEN = 256
DROPOUT_RATE = 0.3

# LoRA/DoRA config
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["query", "key", "value", "dense"]  # For BERT/RoBERTa

# Training config
EPOCHS = 30
BATCH_SIZE = 8

# Early stopping config
EARLY_STOPPING_PATIENCE = 5

# Learning rate config
BASE_LR = 2e-5  # Lower LR for fine-tuning pre-trained models
WARMUP_EPOCHS = 2
MIN_LR_RATIO = 0.01

# Loss weights
LAMBDA_CODE = 2.0
LAMBDA_SPEAKER = 1.0

# Ensemble config
NUM_VOTERS = 5
VOTER_LABEL_COLS = ["vote_1", "vote_2", "vote_3", "vote_4", "vote_5"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data config
DATA_FILE = "Sonnet4-consolidated.csv"
TEXT_COL = "content"
CHECKPOINT_DIR = "checkpoints"

# =========================================================================
# 2. BERT-BASED VOTER MODEL
# =========================================================================

class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence classification."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1, bias=False)
        )
        nn.init.xavier_uniform_(self.attention[0].weight)
        nn.init.zeros_(self.attention[0].bias)
        nn.init.zeros_(self.attention[2].weight)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, hidden_size]
        attn_weights = self.attention(x)  # [batch_size, seq_len, 1]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (x * attn_weights).sum(dim=1)  # [batch_size, hidden_size]
        return pooled


class ClassificationHead(nn.Module):
    """Classification head for code and speaker prediction."""
    def __init__(self, hidden_size, num_classes, dropout=DROPOUT_RATE):
        super().__init__()
        self.attention_pool = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        pooled = self.attention_pool(x, mask)
        return self.classifier(pooled)


class VoterModelBERT(nn.Module):
    """BERT/RoBERTa-based voter model with LoRA/DoRA."""
    def __init__(self, model_name: str = BASE_MODEL_NAME, voter_id: int = 0,
                 lora_rank: int = LORA_RANK, lora_alpha: int = LORA_ALPHA,
                 lora_dropout: float = LORA_DROPOUT, use_dora: bool = USE_DORA):
        super().__init__()
        self.voter_id = voter_id
        self.model_name = model_name
        
        # Set seed for reproducibility
        torch.manual_seed(42 + voter_id)
        
        # Load pre-trained model
        if "roberta" in model_name.lower():
            self.backbone = RobertaModel.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        elif "bert" in model_name.lower():
            self.backbone = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            # Try AutoModel as fallback
            self.backbone = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hidden_size = self.backbone.config.hidden_size
        
        # Freeze base model parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Add LoRA/DoRA adapters
        if use_dora and DORA_AVAILABLE:
            peft_config = DoRAConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=TARGET_MODULES,
            )
            print(f"[Voter {voter_id}] Using DoRA with rank={lora_rank}, alpha={lora_alpha}")
        else:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=TARGET_MODULES,
            )
            print(f"[Voter {voter_id}] Using LoRA with rank={lora_rank}, alpha={lora_alpha}")
        
        self.backbone = get_peft_model(self.backbone, peft_config)
        
        # Classification heads
        self.code_head = ClassificationHead(hidden_size, NEGOTIATION_CODE_CLASSES, DROPOUT_RATE)
        self.speaker_head = ClassificationHead(hidden_size, SPEAKER_CLASSES, DROPOUT_RATE)
        
        print(f"[Voter {voter_id}] Model: {model_name}, Hidden size: {hidden_size}")
        print(f"[Voter {voter_id}] Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, input_ids, attention_mask=None):
        # Get BERT/RoBERTa embeddings
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Classification
        code_logits = self.code_head(hidden_states, attention_mask)
        speaker_logits = self.speaker_head(hidden_states, attention_mask)
        
        return code_logits, speaker_logits
    
    def get_tokenizer(self):
        return self.tokenizer

# =========================================================================
# 3. DATASET
# =========================================================================

class VoterDatasetBERT(torch.utils.data.Dataset):
    """Dataset for BERT-based models using BERT tokenizer."""
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

            # Tokenize with BERT tokenizer
            encoded = self.tokenizer(
                text,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            self.samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
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
# 4. TRAINER
# =========================================================================

class VoterTrainerBERT:
    """Trainer for BERT-based voter models."""
    def __init__(self, model: VoterModelBERT, train_data, val_data, voter_name="voter",
                 learning_rate: float = BASE_LR, weight_decay: float = 0.01):
        self.model = model.to(DEVICE)
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=BATCH_SIZE, shuffle=False
        )
        self.voter_name = voter_name

        # Optimizer - only train LoRA/DoRA parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * EPOCHS
        self.warmup_steps = self.steps_per_epoch * WARMUP_EPOCHS
        self.current_step = 0

        self.ce_code = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_speaker = nn.CrossEntropyLoss(ignore_index=-100)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def _update_lr(self):
        """Update learning rate with warmup and cosine decay."""
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
            code_labels = batch['code_label'].to(DEVICE)
            speaker_labels = batch['speaker_label'].to(DEVICE)

            code_logits, speaker_logits = self.model(input_ids, attention_mask)

            code_mask = code_labels != -100
            speaker_mask = speaker_labels != -100

            code_loss = torch.tensor(0.0, device=DEVICE)
            speaker_loss = torch.tensor(0.0, device=DEVICE)

            if code_mask.any():
                code_loss = self.ce_code(code_logits[code_mask], code_labels[code_mask])
            if speaker_mask.any():
                speaker_loss = self.ce_speaker(speaker_logits[speaker_mask], speaker_labels[speaker_mask])

            loss = LAMBDA_CODE * code_loss + LAMBDA_SPEAKER * speaker_loss

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
                code_labels = batch['code_label'].to(DEVICE)
                speaker_labels = batch['speaker_label'].to(DEVICE)

                code_logits, speaker_logits = self.model(input_ids, attention_mask)

                code_mask = code_labels != -100
                speaker_mask = speaker_labels != -100

                code_loss = torch.tensor(0.0, device=DEVICE)
                speaker_loss = torch.tensor(0.0, device=DEVICE)

                if code_mask.any():
                    code_loss = self.ce_code(code_logits[code_mask], code_labels[code_mask])
                if speaker_mask.any():
                    speaker_loss = self.ce_speaker(speaker_logits[speaker_mask], speaker_labels[speaker_mask])

                loss = LAMBDA_CODE * code_loss + LAMBDA_SPEAKER * speaker_loss
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
# 5. ENSEMBLE
# =========================================================================

class VoterEnsembleBERT:
    """Ensemble of BERT-based voters."""
    def __init__(self, voters: List[VoterModelBERT]):
        self.voters = voters

    def predict(self, input_ids, attention_mask):
        """Average logits across all voters."""
        all_code_logits = []

        with torch.no_grad():
            for voter in self.voters:
                voter.eval()
                code_logits, _ = voter(input_ids, attention_mask)
                all_code_logits.append(code_logits)

        ensemble_logits = torch.stack(all_code_logits, dim=0).mean(dim=0)
        predictions = torch.argmax(ensemble_logits, dim=1)

        return predictions, ensemble_logits

    def predict_majority_vote(self, input_ids, attention_mask):
        """Majority voting across all voters."""
        all_predictions = []

        with torch.no_grad():
            for voter in self.voters:
                voter.eval()
                code_logits, _ = voter(input_ids, attention_mask)
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
        """Evaluate ensemble on a dataloader."""
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

# =========================================================================
# 6. CHECKPOINT
# =========================================================================

def save_checkpoint(model, name):
    """Save model checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[Checkpoint] Saved {model_path}")

# =========================================================================
# 7. MAIN (for testing)
# =========================================================================

if __name__ == "__main__":
    print("="*60)
    print("BERT-BASED VOTER ENSEMBLE")
    print("="*60)
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Using DoRA: {USE_DORA and DORA_AVAILABLE}")
    print(f"Device: {DEVICE}")
    print("="*60)


