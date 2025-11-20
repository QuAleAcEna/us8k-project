import argparse
import os
from pathlib import Path

import optuna
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# Reutilizamos dataset e modelos existentes
from train_iter1 import US8K as US8K_CNN, AudioCNN, make_splits as make_splits_cnn
from train_rnn_iter1 import US8KSeq as US8K_RNN, AudioGRU, make_splits as make_splits_rnn


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_data(model_type: str):
    if model_type == "cnn":
        tr_df, va_df, _ = make_splits_cnn()
        return US8K_CNN(tr_df, augment=True), US8K_CNN(va_df, augment=False)
    elif model_type == "rnn":
        tr_df, va_df, _ = make_splits_rnn()
        return US8K_RNN(tr_df, augment=True), US8K_RNN(va_df, augment=False)
    else:
        raise ValueError("model_type deve ser 'cnn' ou 'rnn'")


def build_model(model_type: str):
    if model_type == "cnn":
        return AudioCNN()
    else:
        return AudioGRU()


def objective(trial: optuna.trial.Trial, model_type: str, device: torch.device):
    batch = trial.suggest_categorical("batch", [16, 32, 48, 64])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    epochs = trial.suggest_int("epochs", 4, 20)

    train_ds, val_ds = get_data(model_type)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(model_type).to(device)
    # aplica dropout customizado para CNN head ou emb RNN
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    patience, waited = 2, 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, device=device, dtype=torch.long)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validação
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = torch.as_tensor(yb, device=device, dtype=torch.long)
                logits = model(xb)
                preds.extend(logits.argmax(1).cpu().tolist())
                gts.extend(yb.cpu().tolist())

        acc = accuracy_score(gts, preds)
        trial.report(acc, epoch)
        best_val = max(best_val, acc)

        # early stopping se sem ganho
        if epoch == 0:
            continue
        if acc + 1e-3 < best_val:
            waited += 1
            if waited >= patience:
                break
        else:
            waited = 0

        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val


def main():
    load_dotenv()
    assert os.getenv("US8K_ROOT"), "Define US8K_ROOT no .env"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "rnn"], default="cnn")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--study-name", default="us8k_optuna")
    parser.add_argument("--storage", default=None, help="Ex: sqlite:///optuna.db para persistir")
    parser.add_argument("--csv", default="runs/optuna_results.csv", help="Ficheiro CSV para guardar cada trial")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )

    # callback para guardar cada trial em CSV
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log_trial(study: optuna.Study, trial: optuna.Trial):
        row = {
            "study": args.study_name,
            "model": args.model,
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
        }
        row.update(trial.params)
        write_header = not csv_path.exists()
        import csv

        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    study.optimize(
        lambda t: objective(t, args.model, device),
        n_trials=args.trials,
        timeout=None,
        callbacks=[log_trial],
    )

    print("\n== Melhor trial ==")
    print("Acc:", study.best_value)
    print("Hparams:", study.best_params)


if __name__ == "__main__":
    main()
