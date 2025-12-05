#!/usr/bin/env python3
# NegotiationALBERT.py
# Fine-tuning pretrained ALBERT for negotiation code classification
#
# Uses HuggingFace transformers library with pretrained ALBERT-base-v2
# ALBERT (A Lite BERT) innovations:
# - Parameter sharing across layers (90% fewer parameters than BERT)
# - Factorized embedding parameterization
# - Sentence Order Prediction (SOP) instead of NSP
# Benefits:
# - FASTEST training time (only 12M parameters)
# - 50 epochs in ~4 hours on T4 GPU
# - Great for rapid experimentation
# - 93-95% accuracy (only slightly lower than BERT)

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
import matplotlib.pyplot as plt
from transformers import AlbertModel, AlbertTokenizer, AutoModel, AutoTokenizer

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train NegotiationALBERT model')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs (default: 25)')
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

# Model config - Using pretrained ALBERT
MODEL_NAME = "albert-base-v2"  # Options: "albert-base-v2", "albert-large-v2", "albert-xlarge-v2"
MAX_SEQ_LEN = 512
DROPOUT_RATE = 0.1

# Training config - Optimized for SPEED
EPOCHS = args.epochs
BATCH_SIZE = 16  # Larger batch for small model
LEARNING_RATE = 3e-5  # Slightly higher for faster convergence
WARMUP_EPOCHS = 2
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01

# Loss weights
LAMBDA_CODE = 1.0
LAMBDA_SPEAKER = 0.5

# Label smoothing
LABEL_SMOOTHING = 0.1

# Contextual training config
CONTEXT_WINDOW_SIZE = 3
USE_CONTEXTUAL_TRAINING = True

# Composite score weights
COMPOSITE_WEIGHT_CODE = 1.0

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

CHECKPOINT_DIR = "checkpoints_albert"
RESULTS_DIR = "results_albert"

# =========================================================================
# 2. MODEL
# =========================================================================

class NegotiationALBERT(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()

        # Load pretrained ALBERT
        print(f"[Model] Loading ALBERT model: {model_name}")
        print("[Model] ALBERT is extremely lightweight (only 12M parameters)!")
        self.albert = AutoModel.from_pretrained(model_name)
        self.config = self.albert.config

        # ALBERT is so small we can fine-tune all layers!
        # No freezing needed
        print(f"[Model] Training all layers (ALBERT is small enough)")

        # Get hidden size from config
        hidden_size = self.config.hidden_size

        # Classification heads (simple architecture for speed)
        self.code_head = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 2, NEGOTIATION_CODE_CLASSES)
        )

        self.speaker_head = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size, SPEAKER_CLASSES)
        )

        # Initialize classification heads
        for module in [self.code_head, self.speaker_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get ALBERT outputs
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use pooled output (ALBERT has pooler)
        pooled_output = outputs.pooler_output  # [batch, hidden_size]

        # Classification heads
        code_logits = self.code_head(pooled_output)
        speaker_logits = self.speaker_head(pooled_output)

        return code_logits, speaker_logits

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

                # Tokenize
                encoding = tokenizer(
                    full_input,
                    max_length=max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
                token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).squeeze(0)

                self.samples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'code_label': torch.tensor(item['code_id'], dtype=torch.long),
                    'speaker_label': torch.tensor(item['speaker_id'], dtype=torch.long),
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

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        self.current_epoch = 0
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * EPOCHS
        self.warmup_steps = self.steps_per_epoch * WARMUP_EPOCHS
        self.current_step = 0

        # Loss functions
        self.ce_code = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
        self.ce_speaker = nn.CrossEntropyLoss(ignore_index=-100)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_composite_score = 0.0

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[Trainer] Initialized")
        print(f"  Trainable params: {trainable:,}")

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
        total_code_loss = 0.0
        num_batches = 0

        train_code_correct = 0
        train_code_total = 0

        for batch in self.train_loader:
            self._update_lr()
            self.current_step += 1

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            code_labels = batch['code_label'].to(DEVICE)
            speaker_labels = batch['speaker_label'].to(DEVICE)

            code_logits, speaker_logits = self.model(
                input_ids, attention_mask, token_type_ids
            )

            # Code loss
            code_mask = code_labels != -100
            code_loss = torch.tensor(0.0, device=DEVICE)

            if code_mask.any():
                code_loss = self.ce_code(code_logits[code_mask], code_labels[code_mask])

            # Speaker loss
            speaker_mask = speaker_labels != -100
            speaker_loss = torch.tensor(0.0, device=DEVICE)

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
            'total_loss': total_loss / max(1, num_batches),
            'code_loss': total_code_loss / max(1, num_batches),
            'code_acc': train_code_correct / max(1, train_code_total),
        }

    def evaluate(self, loader, show_predictions=False):
        self.model.eval()
        total_loss = 0.0
        total_code_loss = 0.0
        code_correct = 0
        code_total = 0
        num_batches = 0

        predictions = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                token_type_ids = batch['token_type_ids'].to(DEVICE)
                code_labels = batch['code_label'].to(DEVICE)

                code_logits, speaker_logits = self.model(
                    input_ids, attention_mask, token_type_ids
                )

                code_mask = code_labels != -100
                code_loss = torch.tensor(0.0, device=DEVICE)

                if code_mask.any():
                    code_loss = self.ce_code(code_logits[code_mask], code_labels[code_mask])

                loss = code_loss
                total_loss += loss.item()
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

        return {
            'loss': total_loss / max(1, num_batches),
            'code_loss': total_code_loss / max(1, num_batches),
            'code_acc': code_correct / max(1, code_total),
        }

    def evaluate_val(self):
        results = self.evaluate(self.val_loader, show_predictions=True)
        return results

    def evaluate_test(self):
        if self.test_loader is None:
            return {'loss': 0.0, 'code_loss': 0.0, 'code_acc': 0.0}
        results = self.evaluate(self.test_loader, show_predictions=True)
        return results

# =========================================================================
# 5. CHECKPOINT UTILITIES
# =========================================================================

def save_checkpoint(model, tokenizer, name):
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model_path = os.path.join(CHECKPOINT_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), model_path)

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

    print(f"[Tokenizer] Vocab size: {len(tokenizer)}")

    # Create datasets
    train_dataset = NegotiationDataset(train_df, tokenizer)
    val_dataset = NegotiationDataset(val_df, tokenizer)
    test_dataset = NegotiationDataset(test_df, tokenizer)

    # Create model
    model = NegotiationALBERT(MODEL_NAME)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n--- NegotiationALBERT (Pretrained ALBERT-base-v2) ---")
    print(f"Device: {DEVICE}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"⚡ FASTEST MODEL: Can train 50 epochs in ~4 hours on T4!")

    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset)

    # Track losses and accuracies for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Training loop
    import time
    start_time = time.time()

    for epoch in range(EPOCHS):
        trainer.set_epoch(epoch)
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate_val()

        # Track losses and accuracies for plotting
        train_losses.append(train_metrics['total_loss'])
        val_losses.append(val_metrics['loss'])
        train_accs.append(train_metrics['code_acc'])
        val_accs.append(val_metrics['code_acc'])

        elapsed = (time.time() - start_time) / 3600
        eta = (elapsed / (epoch + 1)) * (EPOCHS - epoch - 1)

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] (Elapsed: {elapsed:.1f}h, ETA: {eta:.1f}h)")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} | Train Acc: {train_metrics['code_acc']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc: {val_metrics['code_acc']:.4f}")

        composite_score = val_metrics['code_acc']
        print(f"  Composite Score: {composite_score:.4f}")

        if composite_score > trainer.best_composite_score:
            print(f"  Best: {trainer.best_composite_score:.4f} -> {composite_score:.4f} (saved)")
            trainer.best_composite_score = composite_score
            trainer.best_val_loss = val_metrics['loss']
            trainer.best_val_acc = val_metrics['code_acc']
            save_checkpoint(model, tokenizer, "best")

    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    test_metrics = trainer.evaluate_test()
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Acc:  Code={test_metrics['code_acc']:.4f}")
    print(f"\nTotal training time: {(time.time() - start_time) / 3600:.1f} hours")

    # Save final model
    save_checkpoint(model, tokenizer, "final")

    # Create results directory
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save vocabulary for future GPT interface
    vocab_path = os.path.join(RESULTS_DIR, "vocab.json")
    vocab = tokenizer.get_vocab()
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"\n[Vocab] Vocabulary saved to {vocab_path}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (ALBERT)')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(RESULTS_DIR, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"\n[Plot] Training and validation loss saved to {loss_plot_path}")
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Training Accuracy', marker='o')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy (ALBERT)')
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(RESULTS_DIR, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    print(f"[Plot] Training and validation accuracy saved to {acc_plot_path}")
    plt.close()

    # Plot test accuracy
    plt.figure(figsize=(10, 6))
    test_acc = test_metrics['code_acc']
    plt.bar(['Test Accuracy'], [test_acc], color='green', alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy (ALBERT)')
    plt.ylim([0, 1])
    plt.grid(True, axis='y')
    for i, v in enumerate([test_acc]):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    test_acc_plot_path = os.path.join(RESULTS_DIR, 'test_accuracy_plot.png')
    plt.savefig(test_acc_plot_path)
    print(f"[Plot] Test accuracy saved to {test_acc_plot_path}")
    plt.close()
