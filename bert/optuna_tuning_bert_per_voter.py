#!/usr/bin/env python3
# optuna_tuning_bert_per_voter.py
# Optuna Hyperparameter Tuning for BERT-based VoterEnsembleGPT
# Tunes each voter separately to find optimal hyperparameters per voter

import optuna
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, Any
import json
import os
import sys

# Import from VoterEnsembleBERT
from VoterEnsembleBERT import (
    VoterModelBERT, VoterDatasetBERT, VoterTrainerBERT,
    NEGOTIATION_CODES_MAP, CODE_ID_TO_STR,
    DEVICE, DATA_FILE, TEXT_COL, VOTER_LABEL_COLS, CHECKPOINT_DIR,
    BASE_MODEL_NAME, USE_DORA
)

# Reduced epochs for faster hyperparameter search
TUNING_EPOCHS = 5  # Use fewer epochs during tuning
FULL_EPOCHS = 30   # Full training epochs after finding best params

# Optuna study config
N_TRIALS = 15  # Fewer trials per voter (since we're doing 5 voters)
TIMEOUT = None

# Early stopping config
EARLY_STOPPING_PATIENCE = 5

# Storage for Optuna study (SQLite database - persists results)
STUDY_DB_TEMPLATE = os.path.join(CHECKPOINT_DIR, "optuna_study_bert_voter_{}.db")
STUDY_NAME_TEMPLATE = "voter_ensemble_bert_hyperopt_voter_{}"


def create_model_with_params(vocab_size: int, params: Dict[str, Any], voter_id: int = 0):
    """Create a VoterModelBERT with given hyperparameters."""
    model = VoterModelBERT(
        model_name=params.get('model_name', BASE_MODEL_NAME),
        voter_id=voter_id,
        lora_rank=params['lora_rank'],
        lora_alpha=params['lora_alpha'],
        lora_dropout=params.get('lora_dropout', 0.1),
        use_dora=params.get('use_dora', USE_DORA)
    )
    return model


def create_trainer_with_params(model, train_dataset, val_dataset, params, voter_name):
    """Create a VoterTrainerBERT with given hyperparameters."""
    trainer = VoterTrainerBERT(
        model,
        train_dataset,
        val_dataset,
        voter_name=voter_name,
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    return trainer


def save_trial_checkpoint(trial: optuna.Trial, params: Dict[str, Any], val_acc: float, voter_id: int):
    """Save trial hyperparameters and results to checkpoint file."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"optuna_trials_bert_voter_{voter_id}.json")
    
    # Load existing trials if file exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            trials_data = json.load(f)
    else:
        trials_data = {"trials": [], "best_trial": None, "best_value": 0.0, "voter_id": voter_id}
    
    # Add current trial
    trial_data = {
        "trial_number": trial.number,
        "params": params,
        "validation_accuracy": val_acc,
        "state": "COMPLETE"
    }
    trials_data["trials"].append(trial_data)
    
    # Update best trial if this is better
    if val_acc > trials_data.get("best_value", 0.0):
        trials_data["best_trial"] = trial.number
        trials_data["best_value"] = val_acc
        trials_data["best_params"] = params
    
    # Save updated trials
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(trials_data, f, indent=2)
    
    # Also save best hyperparameters separately
    if val_acc >= trials_data.get("best_value", 0.0):
        best_params_file = os.path.join(CHECKPOINT_DIR, f"best_hyperparameters_bert_voter_{voter_id}.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_value': trials_data["best_value"],
                'best_params': trials_data.get("best_params", params),
                'best_trial': trials_data["best_trial"],
                'voter_id': voter_id,
                'n_trials_completed': len(trials_data["trials"]),
                'n_trials_total': N_TRIALS
            }, f, indent=2)


def objective(trial: optuna.Trial, voter_id: int, label_col: str) -> float:
    """Optuna objective function - trains model and returns validation accuracy."""
    
    # Suggest hyperparameters
    params = {
        # Model selection
        'model_name': trial.suggest_categorical('model_name', [
            'roberta-base',
            'bert-base-uncased',
            'distilbert-base-uncased'
        ]),
        
        # LoRA/DoRA parameters
        'lora_rank': trial.suggest_int('lora_rank', 4, 16, step=2),
        'lora_alpha': trial.suggest_int('lora_alpha', 8, 32, step=4),
        'lora_dropout': trial.suggest_float('lora_dropout', 0.05, 0.2, step=0.05),
        'use_dora': trial.suggest_categorical('use_dora', [False, True]) if USE_DORA else False,
        
        # Training parameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.3, step=0.05),
    }
    
    print(f"\n{'='*60}")
    print(f"Voter {voter_id+1} - Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*60}")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Log hyperparameters to file
    log_file = os.path.join(CHECKPOINT_DIR, f"optuna_trials_log_voter_{voter_id}.txt")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Voter {voter_id+1} - Trial {trial.number} - Started\n")
        f.write(f"{'='*60}\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
        f.flush()
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.lower() for c in df.columns]
    
    conversations = df['transcript_name'].unique().tolist()
    train_convs = conversations[:-2]
    val_conv = conversations[-2]
    
    train_df = df[df['transcript_name'].isin(train_convs)]
    val_df = df[df['transcript_name'] == val_conv]
    
    # Create model and get tokenizer
    model = create_model_with_params(0, params, voter_id=voter_id)
    tokenizer = model.get_tokenizer()
    
    # Train this specific voter
    train_dataset = VoterDatasetBERT(train_df.copy(), tokenizer, label_column=label_col)
    val_dataset = VoterDatasetBERT(val_df.copy(), tokenizer, label_column=label_col)
    
    trainer = create_trainer_with_params(model, train_dataset, val_dataset, params, f"Voter-{voter_id+1}-Trial-{trial.number}")
    
    # Train for reduced epochs with early stopping
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(TUNING_EPOCHS):
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate()
        
        val_acc = val_metrics['code_acc']
        train_acc = train_metrics['code_acc']
        
        # Check for overfitting
        if train_acc > 0.95 and val_acc < 0.5:
            print(f"  ⚠️  Overfitting detected (train: {train_acc:.3f}, val: {val_acc:.3f})")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping if no improvement
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break
        
        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        
        # Handle pruning
        if trial.should_prune():
            print(f"  Trial pruned at epoch {epoch+1}")
            raise optuna.TrialPruned()
    
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    # Save checkpoint after each trial
    save_trial_checkpoint(trial, params, best_val_acc, voter_id)
    
    # Log result to file
    log_file = os.path.join(CHECKPOINT_DIR, f"optuna_trials_log_voter_{voter_id}.txt")
    with open(log_file, 'a') as f:
        f.write(f"  Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Voter {voter_id+1} - Trial {trial.number} - Completed\n")
        f.flush()
    
    return best_val_acc


def run_optuna_study_for_voter(voter_id: int):
    """Run Optuna study for a specific voter."""
    
    label_col = VOTER_LABEL_COLS[voter_id]
    
    print("\n" + "="*60)
    print(f"OPTUNA HYPERPARAMETER TUNING - VOTER {voter_id+1}")
    print(f"Label column: {label_col}")
    print("="*60)
    print(f"Trials: {N_TRIALS}")
    print(f"Epochs per trial: {TUNING_EPOCHS}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Create study with SQLite storage (persists results)
    study_db = STUDY_DB_TEMPLATE.format(voter_id)
    study_name = STUDY_NAME_TEMPLATE.format(voter_id)
    storage_url = f"sqlite:///{study_db}"
    
    # Create objective function for this voter
    def voter_objective(trial):
        return objective(trial, voter_id, label_col)
    
    try:
        # Try to load existing study
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url
        )
        print(f"📂 Loaded existing study from {study_db}")
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"   Completed trials: {completed_trials}/{len(study.trials)}")
        if len(study.trials) > 0 and study.best_trial is not None:
            print(f"   Current best: {study.best_value:.4f} (Trial {study.best_trial.number})")
    except:
        # Create new study if it doesn't exist
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='maximize',
            load_if_exists=False
        )
        print(f"📝 Created new study: {study_name}")
    
    # Run optimization
    print(f"\n🚀 Starting optimization for Voter {voter_id+1}...")
    study.optimize(voter_objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE - VOTER {voter_id+1}")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best hyperparameters
    best_params_file = os.path.join(CHECKPOINT_DIR, f"best_hyperparameters_bert_voter_{voter_id}.json")
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
            'voter_id': voter_id,
            'n_trials': len(study.trials)
        }, f, indent=2)
    
    print(f"\nBest hyperparameters saved to: {best_params_file}")
    
    return study.best_params, study.best_value


def main():
    """Run Optuna tuning for all voters."""
    
    print("\n" + "="*60)
    print("OPTUNA TUNING - EACH VOTER SEPARATELY")
    print("="*60)
    print(f"Total voters: {len(VOTER_LABEL_COLS)}")
    print(f"Trials per voter: {N_TRIALS}")
    print(f"Total trials: {len(VOTER_LABEL_COLS) * N_TRIALS}")
    print("="*60)
    
    all_best_params = {}
    all_best_values = {}
    
    # Tune each voter separately
    for voter_id in range(len(VOTER_LABEL_COLS)):
        print(f"\n{'='*60}")
        print(f"TUNING VOTER {voter_id+1}/{len(VOTER_LABEL_COLS)}")
        print(f"{'='*60}")
        
        best_params, best_value = run_optuna_study_for_voter(voter_id)
        all_best_params[voter_id] = best_params
        all_best_values[voter_id] = best_value
        
        print(f"\n✅ Voter {voter_id+1} complete: {best_value:.4f}")
    
    # Save summary
    summary_file = os.path.join(CHECKPOINT_DIR, "optuna_summary_all_voters.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'best_params_per_voter': all_best_params,
            'best_values_per_voter': all_best_values,
            'average_best_value': sum(all_best_values.values()) / len(all_best_values)
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ALL VOTERS TUNED!")
    print(f"{'='*60}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nAverage best validation accuracy: {sum(all_best_values.values()) / len(all_best_values):.4f}")
    print("\nBest hyperparameters per voter:")
    for voter_id, params in all_best_params.items():
        print(f"  Voter {voter_id+1}: {all_best_values[voter_id]:.4f}")


if __name__ == "__main__":
    main()

