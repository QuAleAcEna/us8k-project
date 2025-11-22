# UrbanSound8K Audio Classification

CNN and GRU baselines for UrbanSound8K. The repo includes single-fold training, 10-fold CV, Optuna hyperparameter search, and quick plotting helpers. All runs write to `runs/`.

## Project Layout
- `src/train_iter1.py` — mel-spectrogram CNN baseline (fold1 test, fold10 val).
- `src/train_rnn_iter1.py` — mel-spectrogram GRU baseline (same folds as above).
- `src/run_cv.py` — 10-fold cross-validation for either model type.
- `src/optuna_search.py` — Optuna search over batch, lr, dropout for CNN/GRU.
- `src/plot_results.py`, `src/plot_rnn_results.py` — quick plots (acc/loss/CM) for a run.
- `runs/` — outputs per run (config, histories, checkpoints, metrics, plots).
- `report.ipynb` — analysis notebook (optional, not required to train).

## Requirements
- Python 3.10+ recommended.
- GPU (CUDA/MPS) optional; scripts default to CPU if not found.
- Install deps (pick the right torch build for your hardware):
  ```bash
  pip install torch torchaudio librosa pandas numpy scikit-learn tqdm matplotlib seaborn optuna python-dotenv
  ```

## Dataset Setup
1) Download UrbanSound8K and unzip so you have `audio/` and `metadata/UrbanSound8K.csv`.  
2) Set `US8K_ROOT` to that folder (via `.env` or shell). Example `.env`:
   ```bash
   US8K_ROOT=/path/to/UrbanSound8K
   # optional overrides
   CNN_BATCH=32
   CNN_EPOCHS=50
   CNN_LR=0.001
   CNN_DROPOUT=0.3
   CNN_PATIENCE=7
   CNN_MIN_DELTA=0.001
   RNN_BATCH=32
   RNN_EPOCHS=15
   RNN_LR=0.001
   RNN_DROPOUT=0.2
   RNN_PATIENCE=7
   RNN_MIN_DELTA=0.001
   ```

## Quickstart Training
Run from the project root:
- CNN baseline (fold1 test, fold10 val):
  ```bash
  python -m src.train_iter1
  ```
- GRU baseline:
  ```bash
  python -m src.train_rnn_iter1
  ```
Each run creates `runs/<model>_iter1_fold1_<timestamp>/` with `config.json`, `history.csv`, `results.json`, `confusion_matrix.csv`, and the best checkpoint `model_best.pt`.

### Overriding hyperparameters (env vars)
You can override the defaults inline when running the single-fold scripts:
```bash
CNN_BATCH=48 CNN_EPOCHS=60 CNN_LR=0.0008 CNN_DROPOUT=0.35 CNN_PATIENCE=8 CNN_MIN_DELTA=0.0005 python -m src.train_iter1
```
Use the `RNN_*` equivalents for the GRU version.

## 10-Fold Cross-Validation
Runs every fold (val fold is the next one cyclically) and aggregates confusion matrices:
```bash
python -m src.run_cv --model cnn   # or rnn
# overrides if needed:
python -m src.run_cv --model cnn --epochs 12 --batch 48 --lr 0.0008 --dropout 0.35
```
Outputs are stored under `runs/cv_<model>_<timestamp>/` with per-fold JSONs, `summary.csv/json`, and a cumulative CM plot.

## Hyperparameter Search (Optuna)
Bayesian search over batch size, learning rate, dropout; epochs are fixed via `--epochs` (default 15):
```bash
python -m src.optuna_search --model cnn --trials 30 \
  --study-name us8k_cnn --storage sqlite:///optuna.db \
  --csv runs/optuna_results.csv
```
Use `--model rnn` for GRU. Results append to the CSV and (if `--storage` is set) persist the study.

## Plotting a Run
- CNN (pick a run dir):
  ```bash
  python -c "from src.plot_results import plot_results; plot_results('runs/cnn_iter1_fold1_YYYYMMDD-HHMMSS')"
  ```
- GRU (uses the latest `rnn_iter1_fold1*` by default):
  ```bash
  python -m src.plot_rnn_results
  ```
Plots are saved to `runs/.../plots/` (accuracy, loss, confusion matrix).

## Tips
- For reproducibility, seeds are fixed to 42; data aug is simple gain/noise.
- `num_workers=0` is intentional to avoid librosa/dataloader issues on macOS.
- Keep dataset on fast storage; audio is trimmed/padded to 4s at 22.05 kHz with 64-bin mel spectrograms.
