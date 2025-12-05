#!/usr/bin/env python3
# NegotiationT5.py
# Fine-tuning pretrained FLAN-T5 for negotiation text generation and code classification
#
# Uses HuggingFace transformers library with pretrained FLAN-T5 model
# T5 is a text-to-text encoder-decoder model that frames ALL tasks as text generation
# FLAN-T5 is instruction-tuned for better zero-shot and few-shot performance
# Benefits:
# - Unified text-to-text framework (generation + classification)
# - Instruction-following capabilities from FLAN tuning
# - Encoder-decoder architecture for better context understanding
# - Superior performance on reasoning tasks
# - Can handle multiple tasks with natural language prompts

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
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train NegotiationT5 (FLAN-T5) model')
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

# Model config - Using pretrained FLAN-T5
MODEL_NAME = "google/flan-t5-base"  # Options: "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"
MAX_SEQ_LEN = 512  # T5 supports long sequences
MAX_OUTPUT_LEN = 64  # Max length for generated outputs
DROPOUT_RATE = 0.2

# Training config
EPOCHS = args.epochs
BATCH_SIZE = 8
LEARNING_RATE = 3e-4  # Slightly higher for T5 fine-tuning
WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01

# Loss weights (T5 is unified, so we use prompts)
LAMBDA_GEN = 1.0  # Generation loss weight

# Label smoothing
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

CHECKPOINT_DIR = "checkpoints_t5"
RESULTS_DIR = "results_t5"

# =========================================================================
# 2. DATASET
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
                code_str = code.lower().strip("<> ").strip()
                return NEGOTIATION_CODES_MAP.get(code_str, -100), code_str
            return -100, ""

        if label_column in df.columns:
            code_data = [map_code(c) for c in df[label_column]]
            code_ids = [x[0] for x in code_data]
            code_strs = [x[1] for x in code_data]
        else:
            code_ids = [-100] * len(df)
            code_strs = [""] * len(df)

        # Map speakers
        def map_speaker(s):
            if not isinstance(s, str):
                return -100, ""
            t = s.strip().lower()
            if "buyer" in t:
                return SPEAKER_MAP["buyer"], "buyer"
            if "seller" in t:
                return SPEAKER_MAP["seller"], "seller"
            return -100, ""

        speaker_data = None
        for col in ("speakername", "speaker", "role"):
            if col in df.columns:
                speaker_data = [map_speaker(v) for v in df[col]]
                break
        if speaker_data is None:
            speaker_ids = [-100] * len(df)
            speaker_strs = [""] * len(df)
        else:
            speaker_ids = [x[0] for x in speaker_data]
            speaker_strs = [x[1] for x in speaker_data]

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
                'code_str': code_strs[i],
                'speaker_id': speaker_ids[i],
                'speaker_str': speaker_strs[i],
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
                    speaker_label = ctx_item['speaker_str'].capitalize() if ctx_item['speaker_str'] else "Speaker"
                    context_parts.append(f"{speaker_label}: {ctx_item['text']}")

                # Current utterance
                current_text = item['text']
                current_speaker = item['speaker_str'].capitalize() if item['speaker_str'] else "Speaker"

                # T5 uses text-to-text format with prompts
                # Task 1: Code classification - frame as text generation
                if context_parts:
                    context_str = " | ".join(context_parts)
                    input_text_code = f"classify negotiation tactic: {context_str} | {current_speaker}: {current_text}"
                else:
                    input_text_code = f"classify negotiation tactic: {current_speaker}: {current_text}"

                target_text_code = item['code_str'] if item['code_str'] else "unknown"

                # Task 2: Next utterance generation
                if context_parts:
                    input_text_gen = f"generate negotiation response: {context_str} | {current_speaker}:"
                else:
                    input_text_gen = f"generate negotiation response: {current_speaker}:"

                target_text_gen = current_text

                # Tokenize inputs and targets
                # For code classification
                input_encoding_code = tokenizer(
                    input_text_code,
                    max_length=max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                target_encoding_code = tokenizer(
                    target_text_code,
                    max_length=32,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                # For generation
                input_encoding_gen = tokenizer(
                    input_text_gen,
                    max_length=max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                target_encoding_gen = tokenizer(
                    target_text_gen,
                    max_length=MAX_OUTPUT_LEN,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                # T5 uses -100 for padding in labels
                labels_code = target_encoding_code['input_ids'].squeeze(0)
                labels_code[labels_code == tokenizer.pad_token_id] = -100

                labels_gen = target_encoding_gen['input_ids'].squeeze(0)
                labels_gen[labels_gen == tokenizer.pad_token_id] = -100

                self.samples.append({
                    # Code classification task
                    'input_ids_code': input_encoding_code['input_ids'].squeeze(0),
                    'attention_mask_code': input_encoding_code['attention_mask'].squeeze(0),
                    'labels_code': labels_code,
                    # Generation task
                    'input_ids_gen': input_encoding_gen['input_ids'].squeeze(0),
                    'attention_mask_gen': input_encoding_gen['attention_mask'].squeeze(0),
                    'labels_gen': labels_gen,
                    # Metadata
                    'code_label': torch.tensor(item['code_id'], dtype=torch.long),
                    'code_str': item['code_str'],
                    'text': current_text,
                    'transcript_name': transcript_name,
                    'has_context': len(context_parts) > 0,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# =========================================================================
# 3. TRAINER
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

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_composite_score = 0.0

        print(f"\n[Trainer] Initialized")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
        total_gen_loss = 0.0
        num_batches = 0

        train_code_correct = 0
        train_code_total = 0

        for batch in self.train_loader:
            self._update_lr()
            self.current_step += 1

            # Train on both tasks
            # Task 1: Code classification
            input_ids_code = batch['input_ids_code'].to(DEVICE)
            attention_mask_code = batch['attention_mask_code'].to(DEVICE)
            labels_code = batch['labels_code'].to(DEVICE)

            outputs_code = self.model(
                input_ids=input_ids_code,
                attention_mask=attention_mask_code,
                labels=labels_code
            )
            code_loss = outputs_code.loss

            # Task 2: Generation
            input_ids_gen = batch['input_ids_gen'].to(DEVICE)
            attention_mask_gen = batch['attention_mask_gen'].to(DEVICE)
            labels_gen = batch['labels_gen'].to(DEVICE)

            outputs_gen = self.model(
                input_ids=input_ids_gen,
                attention_mask=attention_mask_gen,
                labels=labels_gen
            )
            gen_loss = outputs_gen.loss

            # Combined loss
            loss = 0.5 * code_loss + 0.5 * gen_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_code_loss += code_loss.item()
            total_gen_loss += gen_loss.item()
            num_batches += 1

            # Calculate code accuracy
            with torch.no_grad():
                # Generate predictions
                pred_ids = self.model.generate(
                    input_ids=input_ids_code,
                    attention_mask=attention_mask_code,
                    max_length=16,
                    num_beams=1
                )

                # Decode predictions and compare
                for i in range(len(batch['code_label'])):
                    if batch['code_label'][i].item() != -100:
                        pred_text = tokenizer.decode(pred_ids[i], skip_special_tokens=True).strip().lower()
                        actual_code = batch['code_str'][i].lower()

                        if pred_text == actual_code:
                            train_code_correct += 1
                        train_code_total += 1

        return {
            'total_loss': total_loss / max(1, num_batches),
            'code_loss': total_code_loss / max(1, num_batches),
            'gen_loss': total_gen_loss / max(1, num_batches),
            'code_acc': train_code_correct / max(1, train_code_total),
        }

    def evaluate(self, loader, show_predictions=False):
        self.model.eval()
        total_loss = 0.0
        total_code_loss = 0.0
        total_gen_loss = 0.0
        code_correct = 0
        code_total = 0
        num_batches = 0

        predictions = []

        with torch.no_grad():
            for batch in loader:
                # Code classification
                input_ids_code = batch['input_ids_code'].to(DEVICE)
                attention_mask_code = batch['attention_mask_code'].to(DEVICE)
                labels_code = batch['labels_code'].to(DEVICE)

                outputs_code = self.model(
                    input_ids=input_ids_code,
                    attention_mask=attention_mask_code,
                    labels=labels_code
                )
                code_loss = outputs_code.loss

                # Generation
                input_ids_gen = batch['input_ids_gen'].to(DEVICE)
                attention_mask_gen = batch['attention_mask_gen'].to(DEVICE)
                labels_gen = batch['labels_gen'].to(DEVICE)

                outputs_gen = self.model(
                    input_ids=input_ids_gen,
                    attention_mask=attention_mask_gen,
                    labels=labels_gen
                )
                gen_loss = outputs_gen.loss

                loss = 0.5 * code_loss + 0.5 * gen_loss
                total_loss += loss.item()
                total_code_loss += code_loss.item()
                total_gen_loss += gen_loss.item()
                num_batches += 1

                # Generate predictions for accuracy
                pred_ids = self.model.generate(
                    input_ids=input_ids_code,
                    attention_mask=attention_mask_code,
                    max_length=16,
                    num_beams=1
                )

                for i in range(len(batch['code_label'])):
                    if batch['code_label'][i].item() != -100:
                        pred_text = tokenizer.decode(pred_ids[i], skip_special_tokens=True).strip().lower()
                        actual_code = batch['code_str'][i].lower()

                        if pred_text == actual_code:
                            code_correct += 1
                        code_total += 1

                        if show_predictions and len(predictions) < 10:
                            predictions.append((pred_text, actual_code))

        if show_predictions and predictions:
            print("\n  Predictions (pred -> actual):")
            for pred, actual in predictions[:10]:
                match = "Y" if pred == actual else "X"
                print(f"    {match} {pred} -> {actual}")

        avg_gen_loss = total_gen_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_gen_loss, 20))

        return {
            'loss': total_loss / max(1, num_batches),
            'code_loss': total_code_loss / max(1, num_batches),
            'gen_loss': avg_gen_loss,
            'perplexity': perplexity,
            'code_acc': code_correct / max(1, code_total),
        }

    def evaluate_val(self):
        results = self.evaluate(self.val_loader, show_predictions=True)
        return results

    def evaluate_test(self):
        if self.test_loader is None:
            return {'loss': 0.0, 'code_loss': 0.0, 'gen_loss': 0.0,
                    'perplexity': 0.0, 'code_acc': 0.0}
        results = self.evaluate(self.test_loader, show_predictions=True)
        return results

# =========================================================================
# 4. CHECKPOINT UTILITIES
# =========================================================================

def save_checkpoint(model, tokenizer, name):
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model_path = os.path.join(CHECKPOINT_DIR, f"{name}_model")
    model.save_pretrained(model_path)

    tokenizer_path = os.path.join(CHECKPOINT_DIR, f"{name}_tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    print(f"[Checkpoint] Saved to {model_path}")

# =========================================================================
# 5. MAIN
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
    print(f"\n[Model] Loading pretrained model: {MODEL_NAME}")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n--- NegotiationT5 (Pretrained FLAN-T5-base) ---")
    print(f"Device: {DEVICE}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset)

    # Track losses and accuracies for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Training loop
    for epoch in range(EPOCHS):
        trainer.set_epoch(epoch)
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate_val()

        # Track losses and accuracies for plotting
        train_losses.append(train_metrics['total_loss'])
        val_losses.append(val_metrics['loss'])
        train_accs.append(train_metrics['code_acc'])
        val_accs.append(val_metrics['code_acc'])

        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} (Code:{train_metrics['code_loss']:.3f} | Gen:{train_metrics['gen_loss']:.3f})")
        print(f"  Train Acc:  Code={train_metrics['code_acc']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} (Code:{val_metrics['code_loss']:.3f} | Gen:{val_metrics['gen_loss']:.3f})")
        print(f"  Val Acc:    Code={val_metrics['code_acc']:.4f}")
        print(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")

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
    print(f"Test Loss: {test_metrics['loss']:.4f} (Code:{test_metrics['code_loss']:.3f} | Gen:{test_metrics['gen_loss']:.3f})")
    print(f"Test Acc:  Code={test_metrics['code_acc']:.4f}")
    print(f"Test Perplexity: {test_metrics['perplexity']:.2f}")

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
    plt.title('Training and Validation Loss (T5)')
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
    plt.title('Training and Validation Accuracy (T5)')
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
    plt.title('Test Accuracy (T5)')
    plt.ylim([0, 1])
    plt.grid(True, axis='y')
    for i, v in enumerate([test_acc]):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    test_acc_plot_path = os.path.join(RESULTS_DIR, 'test_accuracy_plot.png')
    plt.savefig(test_acc_plot_path)
    print(f"[Plot] Test accuracy saved to {test_acc_plot_path}")
    plt.close()
