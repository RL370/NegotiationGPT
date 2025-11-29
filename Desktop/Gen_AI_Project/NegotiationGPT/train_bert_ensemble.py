#!/usr/bin/env python3
# train_bert_ensemble.py
# Train BERT-based ensemble with best hyperparameters from Optuna

import sys
import os
import json
import pandas as pd
import torch
from VoterEnsembleBERT import (
    VoterModelBERT, VoterDatasetBERT, VoterTrainerBERT, VoterEnsembleBERT,
    NEGOTIATION_CODES_MAP, CODE_ID_TO_STR, SPEAKER_MAP,
    DATA_FILE, TEXT_COL, VOTER_LABEL_COLS, CHECKPOINT_DIR,
    DEVICE, EPOCHS, EARLY_STOPPING_PATIENCE, save_checkpoint
)

class Tee:
    """Write to both file and stdout."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout
    
    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)
        self.file.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()

def main():
    log_file = "bert_training_output.log"
    tee = Tee(log_file)
    sys.stdout = tee
    sys.stderr = tee
    
    try:
        print("="*60)
        print("BERT-BASED ENSEMBLE TRAINING")
        print("="*60)
        print(f"Device: {DEVICE}")
        print(f"Epochs: {EPOCHS}")
        print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        print("="*60)
        
        # Load best hyperparameters (per-voter if available, otherwise single file)
        use_per_voter_params = False
        best_params_per_voter = {}
        
        # Try to load per-voter hyperparameters first
        for voter_id in range(len(VOTER_LABEL_COLS)):
            voter_params_file = os.path.join(CHECKPOINT_DIR, f"best_hyperparameters_bert_voter_{voter_id}.json")
            if os.path.exists(voter_params_file):
                with open(voter_params_file, 'r') as f:
                    best_params_per_voter[voter_id] = json.load(f)['best_params']
                use_per_voter_params = True
        
        if use_per_voter_params:
            print(f"\n[Config] Loaded per-voter hyperparameters:")
            for voter_id, params in best_params_per_voter.items():
                print(f"  Voter {voter_id+1}:")
                for key, value in params.items():
                    print(f"    {key}: {value}")
        else:
            # Fall back to single hyperparameter file
            best_params_file = os.path.join(CHECKPOINT_DIR, "best_hyperparameters_bert.json")
            if os.path.exists(best_params_file):
                with open(best_params_file, 'r') as f:
                    single_best_params = json.load(f)['best_params']
                # Use same params for all voters
                for voter_id in range(len(VOTER_LABEL_COLS)):
                    best_params_per_voter[voter_id] = single_best_params
                print(f"\n[Config] Loaded single hyperparameter file from {best_params_file}")
                print(f"  Using same params for all voters:")
                for key, value in single_best_params.items():
                    print(f"    {key}: {value}")
            else:
                # Use defaults if no Optuna results
                default_params = {
                    'model_name': 'roberta-base',
                    'lora_rank': 8,
                    'lora_alpha': 16,
                    'lora_dropout': 0.1,
                    'learning_rate': 2e-5,
                    'weight_decay': 0.01,
                    'use_dora': False
                }
                for voter_id in range(len(VOTER_LABEL_COLS)):
                    best_params_per_voter[voter_id] = default_params
                print(f"\n[Config] Using default hyperparameters (no Optuna results found)")
        
        # Load data
        df = pd.read_csv(DATA_FILE)
        df.columns = [c.lower() for c in df.columns]
        
        conversations = df['transcript_name'].unique().tolist()
        print(f"\n[Data] Found {len(conversations)} conversations")
        
        # Split by conversation
        train_convs = conversations[:-2]
        val_conv = conversations[-2]
        test_conv = conversations[-1]
        
        train_df = df[df['transcript_name'].isin(train_convs)]
        val_df = df[df['transcript_name'] == val_conv]
        test_df = df[df['transcript_name'] == test_conv]
        
        print(f"[Split] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create voters and trainers
        voters = []
        trainers = []
        
        # Get tokenizer from first voter (all use same tokenizer)
        voter_0_params = best_params_per_voter[0]
        first_voter = VoterModelBERT(
            voter_id=0,
            model_name=voter_0_params.get('model_name', 'roberta-base'),
            lora_rank=voter_0_params.get('lora_rank', 8),
            lora_alpha=voter_0_params.get('lora_alpha', 16),
            lora_dropout=voter_0_params.get('lora_dropout', 0.1),
            use_dora=voter_0_params.get('use_dora', False)
        )
        tokenizer = first_voter.get_tokenizer()
        voters.append(first_voter)
        
        # Create remaining voters with their specific hyperparameters
        for i in range(1, len(VOTER_LABEL_COLS)):
            voter_params = best_params_per_voter[i]
            voter = VoterModelBERT(
                voter_id=i,
                model_name=voter_params.get('model_name', 'roberta-base'),
                lora_rank=voter_params.get('lora_rank', 8),
                lora_alpha=voter_params.get('lora_alpha', 16),
                lora_dropout=voter_params.get('lora_dropout', 0.1),
                use_dora=voter_params.get('use_dora', False)
            )
            voters.append(voter)
        
        # Create datasets and trainers for each voter with voter-specific hyperparameters
        for i, (voter, label_col) in enumerate(zip(voters, VOTER_LABEL_COLS)):
            train_dataset = VoterDatasetBERT(train_df.copy(), tokenizer, label_column=label_col)
            val_dataset = VoterDatasetBERT(val_df.copy(), tokenizer, label_column=label_col)
            
            voter_params = best_params_per_voter[i]
            trainer = VoterTrainerBERT(
                voter,
                train_dataset,
                val_dataset,
                voter_name=f"voter_{i+1}",
                learning_rate=voter_params.get('learning_rate', 2e-5),
                weight_decay=voter_params.get('weight_decay', 0.01)
            )
            trainers.append(trainer)
        
        # Training loop with early stopping
        print(f"\n{'='*60}")
        print("TRAINING STARTED")
        print(f"{'='*60}\n")
        
        best_overall_val_acc = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(EPOCHS):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{EPOCHS}")
            print(f"{'='*60}")
            
            # Train all voters
            epoch_avg_train_acc = 0.0
            epoch_avg_val_acc = 0.0
            
            for i, trainer in enumerate(trainers):
                train_metrics = trainer.train_epoch()
                val_metrics = trainer.evaluate()
                
                epoch_avg_train_acc += train_metrics['code_acc']
                epoch_avg_val_acc += val_metrics['code_acc']
                
                print(f"\n[{trainer.voter_name}]")
                print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['code_acc']:.4f}")
                print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['code_acc']:.4f}")
                
                # Check for overfitting
                if train_metrics['code_acc'] > 0.95 and val_metrics['code_acc'] < 0.5:
                    print(f"  ⚠️  Overfitting detected!")
            
            epoch_avg_train_acc /= len(trainers)
            epoch_avg_val_acc /= len(trainers)
            
            print(f"\n[Ensemble Average]")
            print(f"  Train Acc: {epoch_avg_train_acc:.4f}")
            print(f"  Val Acc:   {epoch_avg_val_acc:.4f}")
            
            # Early stopping check
            if epoch_avg_val_acc > best_overall_val_acc:
                best_overall_val_acc = epoch_avg_val_acc
                epochs_without_improvement = 0
                print(f"  ✅ New best validation accuracy: {best_overall_val_acc:.4f}")
                
                # Save best checkpoints
                for i, voter in enumerate(voters):
                    save_checkpoint(voter, f"best_voter_{i+1}")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement ({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})")
            
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\n{'='*60}")
                print(f"Early stopping: No improvement for {EARLY_STOPPING_PATIENCE} epochs")
                print(f"Best validation accuracy: {best_overall_val_acc:.4f}")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                for i, voter in enumerate(voters):
                    save_checkpoint(voter, f"epoch_{epoch+1}_voter_{i+1}")
        
        # Final evaluation on test set
        print(f"\n{'='*60}")
        print("FINAL TEST EVALUATION")
        print(f"{'='*60}\n")
        
        ensemble = VoterEnsembleBERT(voters)
        
        # Test evaluation
        test_datasets = []
        for label_col in VOTER_LABEL_COLS:
            test_dataset = VoterDatasetBERT(test_df.copy(), tokenizer, label_column=label_col)
            test_datasets.append(test_dataset)
        
        # Evaluate each voter individually
        print("Individual Voter Performance:")
        for i, (voter, test_dataset) in enumerate(zip(voters, test_datasets)):
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
            voter.eval()
            
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    code_labels = batch['code_label'].to(DEVICE)
                    
                    code_logits, _ = voter(input_ids, attention_mask)
                    pred_codes = torch.argmax(code_logits, dim=1)
                    
                    mask = code_labels != -100
                    if mask.any():
                        correct += (pred_codes[mask] == code_labels[mask]).sum().item()
                        total += mask.sum().item()
            
            acc = correct / max(1, total)
            print(f"  Voter {i+1}: {acc:.4f} ({correct}/{total})")
        
        # Ensemble evaluation
        print("\nEnsemble Performance:")
        test_dataset = VoterDatasetBERT(test_df.copy(), tokenizer, label_column=VOTER_LABEL_COLS[0])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        ensemble_acc = ensemble.evaluate(test_loader, use_majority_vote=True)
        print(f"  Ensemble (majority vote): {ensemble_acc:.4f}")
        
        ensemble_acc_avg = ensemble.evaluate(test_loader, use_majority_vote=False)
        print(f"  Ensemble (average logits): {ensemble_acc_avg:.4f}")
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {best_overall_val_acc:.4f}")
        print(f"Final test accuracy: {ensemble_acc:.4f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = tee.stdout
        sys.stderr = sys.__stderr__
        tee.close()

if __name__ == "__main__":
    main()

