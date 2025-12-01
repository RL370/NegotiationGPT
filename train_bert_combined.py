#!/usr/bin/env python3
"""
BERT training with combined datasets (5 train / 1 val / 1 test split).
Handles different column formats across datasets.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import json
import os

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training datasets (5)
TRAIN_FILES = [
    "Sonnet4-consolidated.csv",
    "negotiation_dataset_multispeaker_10000.csv",
    "negotiation_synthetic_300_conversations_claude.csv",
    "negotiation_2_speakers_claude.csv",
    "negotiation_3_plus_speakers_claude.csv",
]

# Validation dataset (1)
VAL_FILE = "negotiation_dataset_5000.csv"

# Test dataset (1)
TEST_FILE = "negotiation_dataset_twospeaker_5000.csv"

# Best hyperparameters
MODEL_NAME = "roberta-base"
LORA_RANK = 6
LORA_ALPHA = 24
LORA_DROPOUT = 0.1
LEARNING_RATE = 4.89e-5
WEIGHT_DECAY = 0.26
EPOCHS = 15
BATCH_SIZE = 16
MAX_SEQ_LEN = 128

print(f"Device: {DEVICE}")


def load_and_standardize(filepath):
    """Load CSV and standardize columns to (Content, Label)."""
    df = pd.read_csv(filepath)
    
    # Find content column
    content_col = 'Content' if 'Content' in df.columns else None
    if content_col is None:
        raise ValueError(f"No Content column in {filepath}")
    
    # Find label column (try different names)
    label_col = None
    for col in ['Original_Code', 'Final_Code', 'vote_1', 'human_code']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"No label column found in {filepath}")
    
    # Standardize
    result = pd.DataFrame({
        'Content': df[content_col].fillna("").astype(str),
        'Label': df[label_col].astype(str).str.lower().str.strip()
    })
    
    print(f"  {filepath}: {len(result)} rows, label_col={label_col}")
    return result


def build_code_map(df):
    """Build mapping from code strings to integers."""
    codes = df['Label'].unique()
    codes = [c for c in codes if pd.notna(c) and c != 'nan' and c != '']
    code_to_id = {code: i for i, code in enumerate(sorted(codes))}
    id_to_code = {i: code for code, i in code_to_id.items()}
    return code_to_id, id_to_code


class NegotiationDataset(Dataset):
    def __init__(self, df, tokenizer, code_to_id, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.texts = df['Content'].tolist()
        self.labels = [code_to_id.get(c, -100) for c in df['Label']]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class NegotiationBERT(nn.Module):
    def __init__(self, num_classes, model_name=MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )
        self.bert = get_peft_model(self.bert, lora_config)
        self.bert.print_trainable_parameters()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        valid_mask = labels != -100
        if not valid_mask.any():
            continue
            
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        loss = criterion(logits[valid_mask], labels[valid_mask])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
        total += valid_mask.sum().item()
        
        if batch_idx % 100 == 0:
            print(f"    Batch {batch_idx}/{len(loader)}", end="\r")
    
    return total_loss / len(loader), correct / total if total > 0 else 0


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            valid_mask = labels != -100
            if not valid_mask.any():
                continue
                
            logits = model(input_ids, attention_mask)
            loss = criterion(logits[valid_mask], labels[valid_mask])
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()
    
    return total_loss / len(loader) if len(loader) > 0 else 0, correct / total if total > 0 else 0


def main():
    print("\n" + "="*60)
    print("LOADING AND COMBINING DATASETS")
    print("="*60)
    
    # Load training data (5 files)
    print("\nTraining datasets:")
    train_dfs = []
    for f in TRAIN_FILES:
        df = load_and_standardize(f)
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"\nTotal training samples: {len(train_df)}")
    
    # Load validation data
    print("\nValidation dataset:")
    val_df = load_and_standardize(VAL_FILE)
    
    # Load test data
    print("\nTest dataset:")
    test_df = load_and_standardize(TEST_FILE)
    
    # Build code mapping from ALL data
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    code_to_id, id_to_code = build_code_map(all_df)
    num_classes = len(code_to_id)
    
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Train: {len(train_df):,} samples")
    print(f"Val:   {len(val_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df):,} samples")
    print(f"Classes: {num_classes}")
    
    # Save code mapping
    with open(os.path.join(CHECKPOINT_DIR, "code_mapping_combined.json"), 'w') as f:
        json.dump({"code_to_id": code_to_id, "id_to_code": {str(k): v for k, v in id_to_code.items()}}, f, indent=2)
    
    # Tokenizer
    print(f"\n{'='*60}")
    print("INITIALIZING MODEL")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Datasets
    train_dataset = NegotiationDataset(train_df, tokenizer, code_to_id)
    val_dataset = NegotiationDataset(val_df, tokenizer, code_to_id)
    test_dataset = NegotiationDataset(test_df, tokenizer, code_to_id)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Model
    model = NegotiationBERT(num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} ({train_acc*100:.1f}%)")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f} ({val_acc*100:.1f}%)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model_combined.pt"))
            print(f"  ★ New best! Saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
    
    # Test
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model_combined.pt")))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    print(f"\n{'*'*60}")
    print(f"  TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'*'*60}")
    
    # Save results
    results = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "num_classes": num_classes,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_files": TRAIN_FILES,
        "val_file": VAL_FILE,
        "test_file": TEST_FILE,
    }
    
    with open(os.path.join(CHECKPOINT_DIR, "combined_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {CHECKPOINT_DIR}/combined_results.json")
    print("DONE!")


if __name__ == "__main__":
    main()

