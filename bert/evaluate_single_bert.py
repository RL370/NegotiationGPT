#!/usr/bin/env python3
"""
Evaluate Single BERT Model on Test Set
Uses best hyperparameters from Optuna to train and evaluate a single model
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from VoterEnsembleBERT
from VoterEnsembleBERT import (
    VoterModelBERT, VoterDatasetBERT, VoterTrainerBERT,
    DEVICE, DATA_FILE, TEXT_COL, VOTER_LABEL_COLS
)

CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 8
EPOCHS = 30  # Match ensemble training (will early stop)

def main():
    print("="*60)
    print("SINGLE BERT MODEL - TEST SET EVALUATION")
    print("="*60)
    
    # Load best hyperparameters
    best_params_file = os.path.join(CHECKPOINT_DIR, "best_hyperparameters_bert.json")
    if not os.path.exists(best_params_file):
        print(f"ERROR: {best_params_file} not found!")
        return
    
    with open(best_params_file, 'r') as f:
        data = json.load(f)
        best_params = data['best_params']
    
    print(f"\n[Config] Best hyperparameters from Optuna:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Load data
    print(f"\n[Data] Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.lower() for c in df.columns]
    print(f"  Total samples: {len(df)}")
    
    # Split by conversation
    conversations = df['transcript_name'].unique().tolist()
    print(f"  Found {len(conversations)} conversations")
    
    train_convs = conversations[:-2]
    val_conv = conversations[-2]
    test_conv = conversations[-1]
    
    train_df = df[df['transcript_name'].isin(train_convs)]
    val_df = df[df['transcript_name'] == val_conv]
    test_df = df[df['transcript_name'] == test_conv]
    
    print(f"  Train: {len(train_df)} samples ({len(train_convs)} conversations)")
    print(f"  Val: {len(val_df)} samples (1 conversation)")
    print(f"  Test: {len(test_df)} samples (1 conversation)")
    
    # Use vote_1 as label (same as Optuna tuning)
    label_col = VOTER_LABEL_COLS[0]  # vote_1
    print(f"\n[Training] Using label column: {label_col}")
    
    # Get model name from best params
    model_name = best_params.get('model_name', 'roberta-base')
    
    # Create tokenizer
    print(f"\n[Tokenizer] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = VoterDatasetBERT(train_df.copy(), tokenizer=tokenizer, label_column=label_col)
    val_dataset = VoterDatasetBERT(val_df.copy(), tokenizer=tokenizer, label_column=label_col)
    test_dataset = VoterDatasetBERT(test_df.copy(), tokenizer=tokenizer, label_column=label_col)
    
    # Create model with best params
    print(f"\n[Model] Creating VoterModelBERT with best hyperparameters...")
    model = VoterModelBERT(
        model_name=best_params.get('model_name', 'roberta-base'),
        voter_id=0,
        lora_rank=best_params.get('lora_rank', 6),
        lora_alpha=best_params.get('lora_alpha', 24),
        lora_dropout=best_params.get('lora_dropout', 0.1),
        use_dora=False
    )
    
    # Create trainer
    trainer = VoterTrainerBERT(
        model=model,
        train_data=train_dataset,
        val_data=val_dataset,
        voter_name="Single-BERT",
        learning_rate=best_params.get('learning_rate', 4.89e-5),
        weight_decay=best_params.get('weight_decay', 0.26)
    )
    
    # Training with early stopping
    print(f"\n[Training] Starting training for {EPOCHS} epochs...")
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_results = trainer.train_epoch()
        val_results = trainer.evaluate()
        
        train_loss = train_results['loss']
        train_acc = train_results['code_acc']
        val_loss = val_results['loss']
        val_acc = val_results['code_acc']
        
        print(f"  Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "single_bert_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n[Result] Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # Load best model and evaluate on test set
    print(f"\n[Test] Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "single_bert_best.pt")))
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            code_labels = batch['code_label'].to(DEVICE)
            
            code_logits, _ = model(input_ids, attention_mask)
            predictions = torch.argmax(code_logits, dim=-1)
            
            # Only count valid labels (not -100)
            valid_mask = code_labels != -100
            if valid_mask.any():
                correct += (predictions[valid_mask] == code_labels[valid_mask]).sum().item()
                total += valid_mask.sum().item()
    
    test_acc = correct / total if total > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"FINAL RESULTS - SINGLE BERT MODEL")
    print(f"="*60)
    print(f"  Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Test Accuracy:       {test_acc*100:.2f}%")
    print(f"="*60)
    
    # Save results
    results = {
        "model": "Single BERT",
        "validation_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "hyperparameters": best_params
    }
    
    with open(os.path.join(CHECKPOINT_DIR, "single_bert_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {CHECKPOINT_DIR}/single_bert_results.json")
    
    return test_acc

if __name__ == "__main__":
    main()

