#!/usr/bin/env python3
# Optuna Hyperparameter Tuning for VoterEnsembleGPT
# Optimizes hyperparameters to maximize validation accuracy

import optuna
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, Any
import json
import os

# Import from VoterEnsembleGPT
from VoterEnsembleGPT import (
    NegotiationTokenizer, VoterModel, VoterDataset, VoterTrainer,
    VoterEnsemble, NEGOTIATION_CODES_MAP, CODE_ID_TO_STR,
    DEVICE, DATA_FILE, TEXT_COL, VOTER_LABEL_COLS, CHECKPOINT_DIR
)

# Reduced epochs for faster hyperparameter search
TUNING_EPOCHS = 5  # Use fewer epochs during tuning, then train full with best params
FULL_EPOCHS = 30   # Full training epochs after finding best params

# Optuna study config
N_TRIALS = 20  # Number of hyperparameter trials to run
TIMEOUT = None  # Set to hours*3600 if you want time limit (e.g., 8*3600 for 8 hours)


def create_model_with_params(vocab_size: int, params: Dict[str, Any], voter_id: int = 0):
    """Create a VoterModel with given hyperparameters."""
    # Temporarily modify global config (not ideal but works for this use case)
    import VoterEnsembleGPT as vgpt
    
    # Save original values
    original_d_model = vgpt.D_MODEL
    original_num_layers = vgpt.NUM_LAYERS
    original_num_heads = vgpt.NUM_HEADS
    original_dropout = vgpt.DROPOUT_RATE
    original_lora_rank = vgpt.LORA_RANK
    original_lora_alpha = vgpt.LORA_ALPHA
    
    # Set new values
    vgpt.D_MODEL = params['d_model']
    vgpt.NUM_LAYERS = params['num_layers']
    vgpt.NUM_HEADS = params['num_heads']
    vgpt.DROPOUT_RATE = params['dropout_rate']
    vgpt.LORA_RANK = params['lora_rank']
    vgpt.LORA_ALPHA = params['lora_alpha']
    
    # Create model
    model = VoterModel(vocab_size=vocab_size, voter_id=voter_id)
    
    # Restore original values
    vgpt.D_MODEL = original_d_model
    vgpt.NUM_LAYERS = original_num_layers
    vgpt.NUM_HEADS = original_num_heads
    vgpt.DROPOUT_RATE = original_dropout
    vgpt.LORA_RANK = original_lora_rank
    vgpt.LORA_ALPHA = original_lora_alpha
    
    return model


def create_trainer_with_params(model, train_data, val_data, params: Dict[str, Any], voter_name: str):
    """Create a VoterTrainer with given hyperparameters."""
    import VoterEnsembleGPT as vgpt
    
    # Save original values
    original_base_lr = vgpt.BASE_LR
    original_lambda_code = vgpt.LAMBDA_CODE
    original_lambda_speaker = vgpt.LAMBDA_SPEAKER
    original_batch_size = vgpt.BATCH_SIZE
    
    # Set new values
    vgpt.BASE_LR = params['learning_rate']
    vgpt.LAMBDA_CODE = params['lambda_code']
    vgpt.LAMBDA_SPEAKER = params['lambda_speaker']
    vgpt.BATCH_SIZE = params['batch_size']
    
    # Create trainer
    trainer = VoterTrainer(model, train_data, val_data, voter_name=voter_name)
    
    # Recreate DataLoaders with new batch size
    trainer.train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True
    )
    trainer.val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=params['batch_size'], shuffle=False
    )
    
    # Recalculate steps
    trainer.steps_per_epoch = len(trainer.train_loader)
    trainer.total_steps = trainer.steps_per_epoch * TUNING_EPOCHS
    trainer.warmup_steps = trainer.steps_per_epoch * 2  # WARMUP_EPOCHS
    
    # Modify weight decay in optimizer (recreate optimizer with new weight_decay)
    param_groups = trainer._build_adaptive_param_groups()
    trainer.optimizer = torch.optim.AdamW(param_groups, weight_decay=params['weight_decay'])
    
    # Restore original values
    vgpt.BASE_LR = original_base_lr
    vgpt.LAMBDA_CODE = original_lambda_code
    vgpt.LAMBDA_SPEAKER = original_lambda_speaker
    vgpt.BATCH_SIZE = original_batch_size
    
    return trainer


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function - trains model and returns validation accuracy."""
    
    # Suggest hyperparameters
    params = {
        # Model architecture (keep reasonable ranges)
        'd_model': trial.suggest_int('d_model', 256, 768, step=128),
        'num_layers': trial.suggest_int('num_layers', 4, 8, step=1),
        'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.3, step=0.05),
        
        # LoRA parameters
        'lora_rank': trial.suggest_int('lora_rank', 4, 16, step=2),
        'lora_alpha': trial.suggest_int('lora_alpha', 8, 32, step=4),
        
        # Training parameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
        'weight_decay': trial.suggest_float('weight_decay', 0.001, 0.1, log=True),
        
        # Loss weights
        'lambda_code': trial.suggest_float('lambda_code', 1.0, 4.0, step=0.5),
        'lambda_speaker': trial.suggest_float('lambda_speaker', 0.5, 2.0, step=0.5),
    }
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*60}")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.lower() for c in df.columns]
    
    conversations = df['transcript_name'].unique().tolist()
    train_convs = conversations[:-2]
    val_conv = conversations[-2]
    
    train_df = df[df['transcript_name'].isin(train_convs)]
    val_df = df[df['transcript_name'] == val_conv]
    
    # Build tokenizer
    tokenizer = NegotiationTokenizer(max_vocab_size=32000, min_freq=2)
    tokenizer.build_from_files([DATA_FILE], text_column=TEXT_COL)
    
    # Train a single voter (vote_1) for speed during tuning
    # You can modify to train all 5 if needed, but it will be slower
    label_col = VOTER_LABEL_COLS[0]  # Use vote_1 for tuning
    
    train_dataset = VoterDataset(train_df.copy(), tokenizer, label_column=label_col)
    val_dataset = VoterDataset(val_df.copy(), tokenizer, label_column=label_col)
    
    # Create model and trainer with suggested hyperparameters
    model = create_model_with_params(tokenizer.vocab_size, params, voter_id=0)
    trainer = create_trainer_with_params(model, train_dataset, val_dataset, params, f"Trial-{trial.number}")
    
    # Train for reduced epochs
    best_val_acc = 0.0
    for epoch in range(TUNING_EPOCHS):
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate()
        
        val_acc = val_metrics['code_acc']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        
        # Handle pruning
        if trial.should_prune():
            print(f"  Trial pruned at epoch {epoch+1}")
            raise optuna.TrialPruned()
    
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    return best_val_acc


def run_optuna_study():
    """Run Optuna study to find best hyperparameters."""
    
    print("\n" + "="*60)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("="*60)
    print(f"Trials: {N_TRIALS}")
    print(f"Epochs per trial: {TUNING_EPOCHS}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize validation accuracy
        study_name='voter_ensemble_hyperopt',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=True)
    
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
    best_params_file = os.path.join(CHECKPOINT_DIR, "best_hyperparameters.json")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
        }, f, indent=2)
    
    print(f"\nBest hyperparameters saved to: {best_params_file}")
    
    # Create visualization (optional, requires plotly)
    try:
        import optuna.visualization as vis
        
        # Save plots
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(CHECKPOINT_DIR, "optimization_history.html"))
        
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(CHECKPOINT_DIR, "param_importances.html"))
        
        print("Visualization plots saved to checkpoints/")
    except ImportError:
        print("Install plotly for visualization: pip install plotly")
    
    return study.best_params


def train_with_best_params(best_params: Dict[str, Any]):
    """Train full model with best hyperparameters found by Optuna."""
    print("\n" + "="*60)
    print("TRAINING FULL MODEL WITH BEST HYPERPARAMETERS")
    print("="*60)
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.lower() for c in df.columns]
    
    conversations = df['transcript_name'].unique().tolist()
    train_convs = conversations[:-2]
    val_conv = conversations[-2]
    test_conv = conversations[-1]
    
    train_df = df[df['transcript_name'].isin(train_convs)]
    val_df = df[df['transcript_name'] == val_conv]
    test_df = df[df['transcript_name'] == test_conv]
    
    # Build tokenizer
    tokenizer = NegotiationTokenizer(max_vocab_size=32000, min_freq=2)
    tokenizer.build_from_files([DATA_FILE], text_column=TEXT_COL)
    
    # Import and modify global config
    import VoterEnsembleGPT as vgpt
    
    # Set best hyperparameters
    vgpt.D_MODEL = best_params['d_model']
    vgpt.NUM_LAYERS = best_params['num_layers']
    vgpt.NUM_HEADS = best_params['num_heads']
    vgpt.DROPOUT_RATE = best_params['dropout_rate']
    vgpt.LORA_RANK = best_params['lora_rank']
    vgpt.LORA_ALPHA = best_params['lora_alpha']
    vgpt.BASE_LR = best_params['learning_rate']
    vgpt.BATCH_SIZE = best_params['batch_size']
    vgpt.LAMBDA_CODE = best_params['lambda_code']
    vgpt.LAMBDA_SPEAKER = best_params['lambda_speaker']
    vgpt.EPOCHS = FULL_EPOCHS
    
    # Train all 5 voters
    voters = []
    trainers = []
    
    for i, label_col in enumerate(VOTER_LABEL_COLS):
        print(f"\n[Voter {i}] Training with best hyperparameters on '{label_col}'...")
        
        train_dataset = VoterDataset(train_df.copy(), tokenizer, label_column=label_col)
        val_dataset = VoterDataset(val_df.copy(), tokenizer, label_column=label_col)
        
        voter = VoterModel(vocab_size=tokenizer.vocab_size, voter_id=i)
        trainer = VoterTrainer(voter, train_dataset, val_dataset, voter_name=f"Voter-{i} ({label_col})")
        
        # Set weight decay
        for param_group in trainer.optimizer.param_groups:
            param_group['weight_decay'] = best_params['weight_decay']
        
        voters.append(voter)
        trainers.append(trainer)
    
    # Import save_checkpoint function
    from VoterEnsembleGPT import save_checkpoint
    
    # Train for full epochs
    for epoch in range(FULL_EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{FULL_EPOCHS}")
        print(f"{'='*60}")
        
        for i, (trainer, label_col) in enumerate(zip(trainers, VOTER_LABEL_COLS)):
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.evaluate()
            
            print(f"[{label_col}] Train: {train_metrics['loss']:.3f} | Acc: {train_metrics['code_acc']:.3f} | Val: {val_metrics['loss']:.3f} | Acc: {val_metrics['code_acc']:.3f}")
            
            # Save checkpoint if validation improved
            if val_metrics['loss'] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics['loss']
                trainer.best_val_acc = val_metrics['code_acc']
                save_checkpoint(trainer.model, f"voter_{i}_{label_col}_best")
                print(f"  ✓ Saved best checkpoint for {label_col} (val_loss: {val_metrics['loss']:.4f})")
        
        # Save periodic checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            for i, (trainer, label_col) in enumerate(zip(trainers, VOTER_LABEL_COLS)):
                save_checkpoint(trainer.model, f"voter_{i}_{label_col}_epoch_{epoch+1}")
                print(f"  ✓ Saved periodic checkpoint for {label_col} at epoch {epoch+1}")
    
    # Evaluate ensemble
    test_dataset = VoterDataset(test_df.copy(), tokenizer, label_column="final_code")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    ensemble = VoterEnsemble(voters)
    avg_acc = ensemble.evaluate(test_loader, use_majority_vote=False)
    majority_acc = ensemble.evaluate(test_loader, use_majority_vote=True)
    
    print(f"\n{'='*60}")
    print("FINAL ENSEMBLE RESULTS (with optimized hyperparameters)")
    print(f"{'='*60}")
    print(f"Test Accuracy (Averaged Logits): {avg_acc:.4f}")
    print(f"Test Accuracy (Majority Vote): {majority_acc:.4f}")
    
    # Save final checkpoints for all voters
    print(f"\nSaving final checkpoints...")
    for i, (voter, label_col) in enumerate(zip(voters, VOTER_LABEL_COLS)):
        save_checkpoint(voter, f"voter_{i}_{label_col}_final")
        print(f"  ✓ Saved final checkpoint for {label_col}")
    
    return ensemble, voters


if __name__ == "__main__":
    # Step 1: Run hyperparameter optimization
    best_params = run_optuna_study()
    
    # Step 2: Train full model with best parameters
    print("\n\nProceed with full training using best hyperparameters? (This will take longer)")
    # Uncomment the line below to automatically train with best params
    # ensemble, voters = train_with_best_params(best_params)

