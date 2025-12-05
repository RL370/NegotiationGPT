#!/usr/bin/env python3
# NegotiationGPT_Pretrained.py
# Fine-tuning pretrained GPT-Neo for negotiation text generation and code classification
#
# Uses HuggingFace transformers library with pretrained GPT-Neo-1.3B model
# GPT-Neo is more advanced than GPT-2 with better performance
# Benefits:
# - State-of-the-art text generation from larger pretrained model
# - Better contextual understanding (1.3B parameters vs GPT-2's 117M)
# - Lower perplexity from advanced architecture
# - Faster convergence with transfer learning

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional
import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train NegotiationGPT_Pretrained (GPT-Neo) model')
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

# Model config - Using pretrained GPT-Neo
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # Options: "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"
MAX_SEQ_LEN = 512  # GPT-Neo supports longer sequences
DROPOUT_RATE = 0.2  # Lower dropout since pretrained model already generalizes well

# Training config
EPOCHS = args.epochs
BATCH_SIZE = 4  # Smaller batch size due to larger model
LEARNING_RATE = 5e-5  # Lower LR for fine-tuning pretrained models
WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01

# Loss weights
LAMBDA_LM = 1.0
LAMBDA_CODE = 2.0

# Label smoothing for better generalization
LABEL_SMOOTHING = 0.1

# Contextual training config
CONTEXT_WINDOW_SIZE = 3
USE_CONTEXTUAL_TRAINING = True

# Composite score weights
COMPOSITE_WEIGHT_CODE = 0.5
COMPOSITE_WEIGHT_LM = 0.5

# Sliding window config
SLIDING_WINDOW_SIZE = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data config
DATA_FILES = ["Sonnet4-consolidated.csv",
              "negotiation_dataset_twospeaker_5000.csv",
              "negotiation_synthetic_300_conversations_claude.csv",
              "negotiation_2_speakers_claude.csv",
              "negotiation_3_plus_speakers_claude.csv"
              ]
TEXT_COL = "content"
LABEL_COL = "final_code"

CHECKPOINT_DIR = "checkpoints_pretrained"

# =========================================================================
# 2. MODEL
# =========================================================================

class NegotiationGPT_Pretrained(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()

        # Load pretrained GPT-Neo
        print(f"[Model] Loading GPT-Neo model: {model_name}")
        print("[Model] This may take a few minutes for first-time download...")
        self.gpt_neo = GPTNeoForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use FP16 on GPU for memory efficiency
            low_cpu_mem_usage=True
        )
        self.config = self.gpt_neo.config

        # Freeze lower layers, only fine-tune upper layers
        # This helps prevent overfitting on small datasets
        num_layers = len(self.gpt_neo.transformer.h)
        layers_to_freeze = int(num_layers * 0.7)  # Freeze 70% of layers

        print(f"[Model] Total layers: {num_layers}, Freezing: {layers_to_freeze}")
        for i, layer in enumerate(self.gpt_neo.transformer.h):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        # Get hidden size from config
        hidden_size = self.config.hidden_size

        # Classification heads for negotiation tasks
        self.code_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 2, NEGOTIATION_CODE_CLASSES)
        )

        self.corr_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 2, 2)
        )

        # Initialize classification heads
        for module in [self.code_head, self.corr_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask=None):
        # Get GPT-Neo outputs
        outputs = self.gpt_neo.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # Language modeling head (from pretrained GPT-Neo)
        lm_logits = self.gpt_neo.lm_head(hidden_states)

        # Classification heads - pool sequence representation
        # Use mean pooling over valid tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)

        code_logits = self.code_head(pooled)
        corr_logits = self.corr_head(pooled)

        return lm_logits, code_logits, corr_logits

# =========================================================================
# 3. DATASET
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

        # Build samples with conversation context
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
                # Build context
                context_start = max(0, idx - CONTEXT_WINDOW_SIZE)
                context_items = conv_items[context_start:idx]

                # Format context
                context_parts = []
                for ctx_item in context_items:
                    speaker_label = "Buyer" if ctx_item['speaker_id'] == 0 else "Seller" if ctx_item['speaker_id'] == 1 else "Speaker"
                    context_parts.append(f"{speaker_label}: {ctx_item['text']}")

                # Current utterance
                current_text = item['text']
                current_speaker = "Buyer" if item['speaker_id'] == 0 else "Seller" if item['speaker_id'] == 1 else "Speaker"

                # Full input
                if context_parts:
                    full_input = " | ".join(context_parts) + f" | {current_speaker}: {current_text}"
                else:
                    full_input = f"{current_speaker}: {current_text}"

                # Tokenize with HuggingFace tokenizer
                encoding = tokenizer(
                    full_input,
                    max_length=max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)

                # Create LM labels
                lm_labels = input_ids.clone()
                lm_labels[:-1] = input_ids[1:]
                lm_labels[-1] = -100

                # Mask context tokens in LM labels (only predict current utterance)
                if context_parts:
                    context_text = " | ".join(context_parts) + f" | {current_speaker}:"
                    context_encoding = tokenizer(
                        context_text,
                        max_length=max_seq_len,
                        truncation=True,
                        add_special_tokens=False
                    )
                    context_len = len(context_encoding['input_ids'])
                    lm_labels[:context_len] = -100

                # Mask padding tokens
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
# 4. TRAINER
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

        # Separate parameter groups for different learning rates
        pretrained_params = []
        new_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'gpt_neo' in name:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)

        param_groups = [
            {'params': pretrained_params, 'lr': LEARNING_RATE},
            {'params': new_params, 'lr': LEARNING_RATE * 3}  # Higher LR for new heads
        ]

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

        self.current_epoch = 0
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * EPOCHS
        self.warmup_steps = self.steps_per_epoch * WARMUP_EPOCHS
        self.current_step = 0

        # Loss functions with label smoothing
        self.ce_lm = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
        self.ce_code = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
        self.ce_corr = nn.CrossEntropyLoss(ignore_index=-100)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_composite_score = 0.0

        print(f"\n[Trainer] Initialized")
        print(f"  Pretrained params: {sum(p.numel() for p in pretrained_params):,}")
        print(f"  New params: {sum(p.numel() for p in new_params):,}")

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

            # LM loss
            lm_loss = self.ce_lm(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

            # Code loss
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
        perplexity = math.exp(min(avg_lm_loss, 20))

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
        self.model.eval()

        results = []
        correct = 0
        total = 0

        with torch.no_grad():
            for start_idx in range(0, len(dataset) - window_size + 1):
                window_samples = [dataset[i] for i in range(start_idx, start_idx + window_size)]

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
# 5. CHECKPOINT UTILITIES
# =========================================================================

def save_checkpoint(model, tokenizer, name):
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model_path = os.path.join(CHECKPOINT_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save tokenizer
    tokenizer_path = os.path.join(CHECKPOINT_DIR, f"{name}_tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    print(f"[Checkpoint] Saved to {model_path}")

# =========================================================================
# 6. MAIN
# =========================================================================

if __name__ == "__main__":
    # Load data
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

    df = pd.concat(dfs, ignore_index=True)
    print(f"[Data] Total samples loaded: {len(df)}")

    # Get unique conversations
    conversations = df['transcript_name'].unique().tolist()
    print(f"\n[Data] Found {len(conversations)} conversations")

    # Split by conversation
    train_convs = conversations[:-2]
    val_conv = conversations[-2]
    test_conv = conversations[-1]

    print(f"\n[Split]")
    print(f"  Train: {len(train_convs)} conversations")
    print(f"  Val: {val_conv}")
    print(f"  Test: {test_conv}")

    train_df = df[df['transcript_name'].isin(train_convs)]
    val_df = df[df['transcript_name'] == val_conv]
    test_df = df[df['transcript_name'] == test_conv]

    print(f"\n[Samples] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load pretrained tokenizer
    print(f"\n[Tokenizer] Loading pretrained tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Add special tokens (GPT-Neo uses GPT-2 tokenizer which needs padding token)
    if tokenizer.pad_token is None:
        special_tokens = {'pad_token': '<pad>'}
        tokenizer.add_special_tokens(special_tokens)

    print(f"[Tokenizer] Vocab size: {len(tokenizer)}")

    # Create datasets
    train_dataset = NegotiationDataset(train_df, tokenizer)
    val_dataset = NegotiationDataset(val_df, tokenizer)
    test_dataset = NegotiationDataset(test_df, tokenizer)

    # Create model
    model = NegotiationGPT_Pretrained(MODEL_NAME)

    # Resize token embeddings for new special tokens
    model.gpt_neo.resize_token_embeddings(len(tokenizer))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n--- NegotiationGPT-Neo (Pretrained GPT-Neo-1.3B) ---")
    print(f"Device: {DEVICE}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {total_params - trainable_params:,}")
    print(f"Memory usage: ~{total_params * 2 / 1e9:.2f}GB (FP16)" if torch.cuda.is_available() else "")

    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset)

    # Track losses for plotting
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(EPOCHS):
        trainer.set_epoch(epoch)
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate_val()

        # Track losses for plotting
        train_losses.append(train_metrics['total_loss'])
        val_losses.append(val_metrics['loss'])

        # Sliding window evaluation
        sw_acc, sw_results = trainer.evaluate_sliding_window(val_dataset, SLIDING_WINDOW_SIZE)

        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} (LM:{train_metrics['lm_loss']:.3f} | Code:{train_metrics['code_loss']:.3f})")
        print(f"  Train Acc:  Code={train_metrics['code_acc']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} (LM:{val_metrics['lm_loss']:.3f} | Code:{val_metrics['code_loss']:.3f})")
        print(f"  Val Acc:    Code={val_metrics['code_acc']:.4f}")
        print(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")
        print(f"  Sliding Window Acc: {sw_acc:.4f} (window_size={SLIDING_WINDOW_SIZE})")

        # Compute composite score
        ppl_score = max(0.0, 1.0 - (val_metrics['perplexity'] / 100.0))
        composite_score = (
            COMPOSITE_WEIGHT_CODE * val_metrics['code_acc'] +
            COMPOSITE_WEIGHT_LM * ppl_score
        )
        print(f"  Composite Score: {composite_score:.4f}")

        if composite_score > trainer.best_composite_score:
            print(f"  Best Composite: {trainer.best_composite_score:.4f} -> {composite_score:.4f} (saved)")
            trainer.best_composite_score = composite_score
            trainer.best_val_loss = val_metrics['loss']
            trainer.best_val_acc = val_metrics['code_acc']
            save_checkpoint(model, tokenizer, "best")

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
    plt.title('Training and Validation Loss (Pretrained GPT-Neo)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_pretrained.png')
    print("\n[Plot] Training and validation loss saved to loss_plot_pretrained.png")
    plt.close()

    # Plot test accuracy
    plt.figure(figsize=(10, 6))
    test_acc = test_metrics['code_acc']
    plt.bar(['Test Accuracy'], [test_acc], color='green', alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy (Pretrained GPT-Neo)')
    plt.ylim([0, 1])
    plt.grid(True, axis='y')
    for i, v in enumerate([test_acc]):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    plt.savefig('test_accuracy_plot_pretrained.png')
    print("[Plot] Test accuracy saved to test_accuracy_plot_pretrained.png")
    plt.close()
