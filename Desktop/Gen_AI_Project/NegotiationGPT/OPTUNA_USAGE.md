# Optuna Hyperparameter Tuning Guide

## Installation

First, install Optuna:
```bash
pip install optuna
```

For visualization (optional but recommended):
```bash
pip install plotly
```

## Usage

### Step 1: Run Hyperparameter Optimization

This will test different hyperparameter combinations to find the best ones:

```bash
cd /Users/alarakaymak/Desktop/Gen_AI_Project/NegotiationGPT
python optuna_tuning.py
```

**What it does:**
- Runs `N_TRIALS` (default: 20) trials
- Each trial trains for `TUNING_EPOCHS` (default: 5) epochs (faster than full training)
- Tests different combinations of:
  - Model architecture (d_model, num_layers, num_heads)
  - Dropout rate
  - LoRA parameters (rank, alpha)
  - Learning rate
  - Batch size
  - Weight decay
  - Loss weights (lambda_code, lambda_speaker)
- Saves best hyperparameters to `checkpoints/best_hyperparameters.json`

**Time estimate:** 
- 20 trials × 5 epochs × ~5 voters = ~500 training runs
- On CPU: ~4-8 hours
- On GPU: ~1-2 hours

### Step 2: Train Full Model with Best Hyperparameters

After finding best hyperparameters, uncomment the last line in `optuna_tuning.py`:

```python
# Uncomment this line:
ensemble, voters = train_with_best_params(best_params)
```

Or manually load the best params and train:

```python
import json
from optuna_tuning import train_with_best_params

# Load best hyperparameters
with open('checkpoints/best_hyperparameters.json', 'r') as f:
    data = json.load(f)
    best_params = data['best_params']

# Train full model
ensemble, voters = train_with_best_params(best_params)
```

## Configuration

Edit these variables in `optuna_tuning.py`:

- `N_TRIALS = 20` - Number of hyperparameter combinations to try
- `TUNING_EPOCHS = 5` - Epochs per trial (fewer = faster, but less accurate)
- `FULL_EPOCHS = 30` - Full training epochs after finding best params
- `TIMEOUT = None` - Time limit in seconds (e.g., `8*3600` for 8 hours)

## Hyperparameter Search Space

The script optimizes:

1. **Model Architecture:**
   - `d_model`: 256-768 (step 128)
   - `num_layers`: 4-8
   - `num_heads`: 4, 8, or 16
   - `dropout_rate`: 0.05-0.3 (step 0.05)

2. **LoRA:**
   - `lora_rank`: 4-16 (step 2)
   - `lora_alpha`: 8-32 (step 4)

3. **Training:**
   - `learning_rate`: 1e-5 to 1e-3 (log scale)
   - `batch_size`: 4, 8, or 16
   - `weight_decay`: 0.001-0.1 (log scale)

4. **Loss Weights:**
   - `lambda_code`: 1.0-4.0 (step 0.5)
   - `lambda_speaker`: 0.5-2.0 (step 0.5)

## Output Files

- `checkpoints/best_hyperparameters.json` - Best hyperparameters found
- `checkpoints/optimization_history.html` - Plot of optimization progress (if plotly installed)
- `checkpoints/param_importances.html` - Which hyperparameters matter most (if plotly installed)

## Tips

1. **Start with fewer trials** (N_TRIALS=10) to test quickly
2. **Use GPU if available** - much faster
3. **Monitor validation accuracy** - watch for overfitting
4. **Adjust search space** - if best params are at boundaries, expand the range
5. **Use pruning** - Optuna will automatically stop bad trials early

## Expected Results

After optimization, you should see:
- Higher validation accuracy than default hyperparameters
- Better generalization (smaller gap between train/val accuracy)
- Optimal model size for your data

The best hyperparameters will be saved and can be used for full training.




