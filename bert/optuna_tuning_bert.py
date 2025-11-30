#!/usr/bin/env python3
# optuna_tuning_bert.py
# Optuna Hyperparameter Tuning for BERT-based VoterEnsembleGPT
# Optimizes hyperparameters to maximize validation accuracy

import optuna
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, Any
import json
import os

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
N_TRIALS = 25  # Number of hyperparameter trials (reduced from 30 to avoid getting stuck)
TIMEOUT = None

# Early stopping config
EARLY_STOPPING_PATIENCE = 5

# Storage for Optuna study (SQLite database - persists results)
# Note: CHECKPOINT_DIR is imported from VoterEnsembleBERT above
STUDY_DB = os.path.join(CHECKPOINT_DIR, "optuna_study_bert.db")
STUDY_NAME = "voter_ensemble_bert_hyperopt"


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


def save_trial_checkpoint(trial: optuna.Trial, params: Dict[str, Any], val_acc: float):
    """Save trial hyperparameters and results to checkpoint file."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "optuna_trials_bert.json")
    
    # Load existing trials if file exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            trials_data = json.load(f)
    else:
        trials_data = {"trials": [], "best_trial": None, "best_value": 0.0}
    
    # Add current trial
    trial_data = {
        "trial_number": trial.number,
        "params": params,
        "validation_accuracy": val_acc,
        "state": "COMPLETE"  # Trial is complete when this function is called
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
    
    # Also save best hyperparameters separately (always update if this is the best)
    if val_acc >= trials_data.get("best_value", 0.0):
        best_params_file = os.path.join(CHECKPOINT_DIR, "best_hyperparameters_bert.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_value': trials_data["best_value"],
                'best_params': trials_data.get("best_params", params),
                'best_trial': trials_data["best_trial"],
                'n_trials_completed': len(trials_data["trials"]),
                'n_trials_total': N_TRIALS
            }, f, indent=2)


def objective(trial: optuna.Trial) -> float:
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
    print(f"Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*60}")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Log hyperparameters to file
    log_file = os.path.join(CHECKPOINT_DIR, "optuna_trials_log.txt")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Trial {trial.number} - Started\n")
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
    model = create_model_with_params(0, params, voter_id=0)
    tokenizer = model.get_tokenizer()
    
    # Train a single voter (vote_1) for speed during tuning
    label_col = VOTER_LABEL_COLS[0]
    
    train_dataset = VoterDatasetBERT(train_df.copy(), tokenizer, label_column=label_col)
    val_dataset = VoterDatasetBERT(val_df.copy(), tokenizer, label_column=label_col)
    
    trainer = create_trainer_with_params(model, train_dataset, val_dataset, params, f"Trial-{trial.number}")
    
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
    save_trial_checkpoint(trial, params, best_val_acc)
    
    # Log result to file
    log_file = os.path.join(CHECKPOINT_DIR, "optuna_trials_log.txt")
    with open(log_file, 'a') as f:
        f.write(f"  Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Trial {trial.number} - Completed\n")
        f.flush()
    
    return best_val_acc


def run_optuna_study():
    """Run Optuna study to find best hyperparameters."""
    
    print("\n" + "="*60)
    print("OPTUNA HYPERPARAMETER TUNING - BERT/RoBERTa")
    print("="*60)
    print(f"Trials: {N_TRIALS}")
    print(f"Epochs per trial: {TUNING_EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"Base model options: roberta-base, bert-base-uncased, distilbert-base-uncased")
    print("="*60)
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Create study with SQLite storage (persists results)
    storage_url = f"sqlite:///{STUDY_DB}"
    
    try:
        # Try to load existing study
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=storage_url
        )
        print(f"📂 Loaded existing study from {STUDY_DB}")
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"   Completed trials: {completed_trials}/{len(study.trials)}")
        if len(study.trials) > 0 and study.best_trial is not None:
            print(f"   Current best: {study.best_value:.4f} (Trial {study.best_trial.number})")
    except optuna.exceptions.DuplicatedStudyError:
        # Study exists but load failed - try with load_if_exists
        study = optuna.create_study(
            direction='maximize',
            study_name=STUDY_NAME,
            storage=storage_url,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
            load_if_exists=True
        )
        print(f"📂 Loaded existing study (using load_if_exists)")
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"   Completed trials: {completed_trials}/{len(study.trials)}")
    except Exception as e:
        # Create new study if it doesn't exist
        study = optuna.create_study(
            direction='maximize',  # Maximize validation accuracy
            study_name=STUDY_NAME,
            storage=storage_url,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
            load_if_exists=True
        )
        print(f"📝 Created new study, saving to {STUDY_DB}")
    
    # Initialize log file
    log_file = os.path.join(CHECKPOINT_DIR, "optuna_trials_log.txt")
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Optuna Study Started/Resumed\n")
        f.write(f"Study: {STUDY_NAME}\n")
        f.write(f"Database: {STUDY_DB}\n")
        f.write(f"Total trials: {N_TRIALS}\n")
        f.write(f"Completed so far: {len(study.trials)}\n")
        f.write(f"{'='*60}\n")
    
    # Calculate remaining trials
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = max(0, N_TRIALS - completed_trials)
    
    if remaining_trials > 0:
        print(f"\n📊 Running {remaining_trials} more trials (total target: {N_TRIALS})")
        
        # Run optimization
        study.optimize(objective, n_trials=remaining_trials, timeout=TIMEOUT, show_progress_bar=True)
    else:
        print(f"\n✅ All {N_TRIALS} trials already completed!")
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    best_params_file = os.path.join(CHECKPOINT_DIR, "best_hyperparameters_bert.json")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': N_TRIALS
        }, f, indent=2)
    
    print(f"\nBest hyperparameters saved to: {best_params_file}")
    
    # Log final results
    log_file = os.path.join(CHECKPOINT_DIR, "optuna_trials_log.txt")
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"OPTIMIZATION COMPLETE\n")
        f.write(f"{'='*60}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best validation accuracy: {study.best_value:.4f}\n")
        f.write(f"\nBest hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nTotal trials completed: {len(study.trials)}\n")
        f.write(f"{'='*60}\n")
    
    # Create visualization plots
    try:
        import optuna.visualization as vis
        
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html(os.path.join(CHECKPOINT_DIR, "bert_optimization_history.html"))
        
        fig2 = vis.plot_param_importances(study)
        fig2.write_html(os.path.join(CHECKPOINT_DIR, "bert_param_importances.html"))
        
        print("Visualization plots saved to checkpoints/")
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    return study


if __name__ == "__main__":
    study = run_optuna_study()
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Review the best hyperparameters above")
    print("2. Run: python train_bert_ensemble.py")
    print("   This will train the full ensemble with best hyperparameters")
    print("="*60)


