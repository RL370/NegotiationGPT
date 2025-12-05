#!/usr/bin/env python3
# NegotiationMistral.py
# Fine-tuning pretrained Mistral-7B for negotiation text generation and code classification
#
# Uses HuggingFace transformers library with pretrained Mistral-7B model
# Mistral-7B is a state-of-the-art open-source LLM from Mistral AI
# Key innovations:
# - Grouped-Query Attention (GQA) for faster inference
# - Sliding Window Attention (8k context) for long-range dependencies
# - Outperforms LLaMA-2-13B on most benchmarks despite being smaller
# Benefits:
# - State-of-the-art text generation and reasoning
# - Instruction-following capabilities
# - Efficient architecture for production use
# - Strong zero-shot and few-shot performance
# - Better understanding of complex negotiation dynamics

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
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train NegotiationMistral (Mistral-7B) model')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3)')
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

# Model config - Using pretrained Mistral-7B
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Options: "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"
MAX_SEQ_LEN = 1024  # Mistral supports 8k context, but we use 1k for memory efficiency
MAX_OUTPUT_LEN = 64
DROPOUT_RATE = 0.1  # Very low dropout for large models

# Training config - Using LoRA for efficient fine-tuning
EPOCHS = args.epochs
BATCH_SIZE = 2  # Very small batch for 7B model
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8
LEARNING_RATE = 2e-5  # Very low LR for large model
WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01

# LoRA config for efficient fine-tuning
USE_LORA = True
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Loss weights
LAMBDA_LM = 1.0
LAMBDA_CODE = 2.0

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

CHECKPOINT_DIR = "checkpoints_mistral"
RESULTS_DIR = "results_mistral"

# =========================================================================
# 2. MODEL
# =========================================================================

class NegotiationMistral(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()

        # Load pretrained Mistral-7B
        print(f"[Model] Loading Mistral-7B model: {model_name}")
        print("[Model] This will download ~14GB. Please wait...")
        print("[Model] Using 4-bit quantization for memory efficiency...")

        # Load with 4-bit quantization for memory efficiency
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        ) if torch.cuda.is_available() else None

        self.mistral = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        self.config = self.mistral.config

        # Apply LoRA for efficient fine-tuning
        if USE_LORA and torch.cuda.is_available():
            print("[Model] Applying LoRA for efficient fine-tuning...")
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            self.mistral = get_peft_model(self.mistral, lora_config)
            self.mistral.print_trainable_parameters()

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
        # Get Mistral outputs
        outputs = self.mistral(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        lm_logits = outputs.logits  # Language modeling logits

        # Classification heads - pool sequence representation
        # Use mean pooling over valid tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
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

                # Tokenize with Mistral tokenizer
                encoding = tokenizer(
                    full_input,
                    max_length=max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)

                # Create LM labels (shift for autoregressive training)
                lm_labels = input_ids.clone()
                lm_labels[:-1] = input_ids[1:]
                lm_labels[-1] = -100

                # Mask context tokens (only predict current utterance)
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
        self.model = model.to(DEVICE) if not torch.cuda.is_available() else model  # Already on device if quantized
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=BATCH_SIZE, shuffle=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=BATCH_SIZE, shuffle=False
        ) if test_data else None

        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        self.current_epoch = 0
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * EPOCHS
        self.warmup_steps = self.steps_per_epoch * WARMUP_EPOCHS
        self.current_step = 0
        self.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS

        # Loss functions
        self.ce_lm = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
        self.ce_code = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_composite_score = 0.0

        print(f"\n[Trainer] Initialized")
        print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
        print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

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

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
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

            loss = (LAMBDA_LM * lm_loss + LAMBDA_CODE * code_loss) / self.gradient_accumulation_steps

            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self._update_lr()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.current_step += 1

            total_loss += (loss.item() * self.gradient_accumulation_steps)
            total_lm_loss += lm_loss.item()
            total_code_loss += code_loss.item()
            num_batches += 1

            # Track accuracy
            if code_mask.any():
                with torch.no_grad():
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

# =========================================================================
# 5. CHECKPOINT UTILITIES
# =========================================================================

def save_checkpoint(model, tokenizer, name):
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save model (LoRA adapters if using PEFT)
    model_path = os.path.join(CHECKPOINT_DIR, f"{name}_model")
    if USE_LORA and hasattr(model.mistral, 'save_pretrained'):
        model.mistral.save_pretrained(model_path)
    else:
        torch.save(model.state_dict(), f"{model_path}.pt")

    # Save tokenizer
    tokenizer_path = os.path.join(CHECKPOINT_DIR, f"{name}_tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    # Save classification heads
    heads_path = os.path.join(CHECKPOINT_DIR, f"{name}_heads.pt")
    torch.save({
        'code_head': model.code_head.state_dict(),
        'corr_head': model.corr_head.state_dict(),
    }, heads_path)

    print(f"[Checkpoint] Saved to {model_path}")

# =========================================================================
# 6. MAIN
# =========================================================================

if __name__ == "__main__":
    # Check dependencies
    try:
        import peft
        import bitsandbytes
        print("[Setup] PEFT and BitsAndBytes available for efficient fine-tuning")
    except ImportError:
        print("[Warning] Install PEFT and BitsAndBytes for best performance:")
        print("  pip install peft bitsandbytes accelerate")
        USE_LORA = False

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

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Tokenizer] Vocab size: {len(tokenizer)}")

    # Create datasets
    train_dataset = NegotiationDataset(train_df, tokenizer)
    val_dataset = NegotiationDataset(val_df, tokenizer)
    test_dataset = NegotiationDataset(test_df, tokenizer)

    # Create model
    model = NegotiationMistral(MODEL_NAME)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n--- NegotiationMistral (Pretrained Mistral-7B) ---")
    print(f"Device: {DEVICE}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {total_params - trainable_params:,}")
    if USE_LORA:
        print(f"LoRA enabled: Training only {100 * trainable_params / total_params:.2f}% of parameters")

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
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} (LM:{train_metrics['lm_loss']:.3f} | Code:{train_metrics['code_loss']:.3f})")
        print(f"  Train Acc:  Code={train_metrics['code_acc']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} (LM:{val_metrics['lm_loss']:.3f} | Code:{val_metrics['code_loss']:.3f})")
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
    print(f"Test Loss: {test_metrics['loss']:.4f} (LM:{test_metrics['lm_loss']:.3f} | Code:{test_metrics['code_loss']:.3f})")
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
    plt.title('Training and Validation Loss (Mistral)')
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
    plt.title('Training and Validation Accuracy (Mistral)')
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
    plt.title('Test Accuracy (Mistral)')
    plt.ylim([0, 1])
    plt.grid(True, axis='y')
    for i, v in enumerate([test_acc]):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    test_acc_plot_path = os.path.join(RESULTS_DIR, 'test_accuracy_plot.png')
    plt.savefig(test_acc_plot_path)
    print(f"[Plot] Test accuracy saved to {test_acc_plot_path}")
    plt.close()
