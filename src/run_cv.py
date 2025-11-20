import os, json, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- imports dos teus treinos ----------
# CNN
from train_iter1 import (
    US8K as US8K_CNN, AudioCNN, N_CLASSES as N_CLASSES_CNN,
    BATCH as BATCH_CNN, EPOCHS as EPOCHS_CNN, LR as LR_CNN, DROPOUT as DROPOUT_CNN,
    DEVICE as DEVICE_CNN
)
# RNN
from train_rnn_iter1 import (
    US8KSeq as US8K_RNN, AudioGRU, N_CLASSES as N_CLASSES_RNN,
    BATCH as BATCH_RNN, EPOCHS as EPOCHS_RNN, LR as LR_RNN, DROPOUT as DROPOUT_RNN,
    DEVICE as DEVICE_RNN
)

# ---------- dataset meta ----------
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
US8K_ROOT = os.getenv("US8K_ROOT")
CSV = os.path.join(US8K_ROOT, "metadata", "UrbanSound8K.csv")
META = pd.read_csv(CSV)

def make_splits_for(test_fold:int, val_fold:int):
    train_folds = [f for f in range(1,11) if f not in (test_fold, val_fold)]
    tr = META[META.fold.isin(train_folds)]
    va = META[META.fold == val_fold]
    te = META[META.fold == test_fold]
    return tr, va, te

def run_one_fold(model_type:str, test_fold:int, val_fold:int, epochs:int=None,
                 batch:int=None, lr:float=None, dropout:float=None):
    """
    Treina e avalia num fold:
      - model_type: "cnn" ou "rnn"
      - test_fold: fold de teste
      - val_fold: fold de validação
    """
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    if model_type == "cnn":
        Dataset   = US8K_CNN
        Model     = AudioCNN
        BATCH     = batch or BATCH_CNN
        LR        = lr or LR_CNN
        DROPOUT   = dropout if dropout is not None else DROPOUT_CNN
        N_CLASSES = N_CLASSES_CNN
        DEVICE    = DEVICE_CNN
    elif model_type == "rnn":
        Dataset   = US8K_RNN
        Model     = AudioGRU
        BATCH     = batch or BATCH_RNN
        LR        = lr or LR_RNN
        DROPOUT   = dropout if dropout is not None else DROPOUT_RNN
        N_CLASSES = N_CLASSES_RNN
        DEVICE    = DEVICE_RNN
    else:
        raise ValueError("model_type deve ser 'cnn' ou 'rnn'")

    if epochs is None:
        epochs = EPOCHS_CNN if model_type == "cnn" else EPOCHS_RNN

    tr_df, va_df, te_df = make_splits_for(test_fold, val_fold)

    # num_workers=0 para evitar chatices no macOS com librosa
    tr_dl = DataLoader(Dataset(tr_df, augment=True),  batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
    va_dl = DataLoader(Dataset(va_df, augment=False), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
    te_dl = DataLoader(Dataset(te_df, augment=False), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    model = Model(dropout=DROPOUT).to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    def run_epoch(dl, train=False):
        model.train(train)
        losses, preds, gts = [], [], []
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = torch.as_tensor(yb, device=DEVICE, dtype=torch.long)
            logits = model(xb)
            loss = crit(logits, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            preds += logits.argmax(1).detach().cpu().tolist()
            gts   += yb.detach().cpu().tolist()
        return float(np.mean(losses)), accuracy_score(gts, preds), (gts, preds)

    best_va, best_state = -1, None
    for ep in range(1, epochs+1):
        tr_loss, tr_acc, _ = run_epoch(tr_dl, train=True)
        va_loss, va_acc, _ = run_epoch(va_dl, train=False)
        print(f"[{model_type.upper()} | fold{test_fold}] Epoch {ep:02d} | train {tr_acc:.3f}/{tr_loss:.3f} | val {va_acc:.3f}/{va_loss:.3f}")
        if va_acc > best_va:
            best_va, best_state = va_acc, {k:v.cpu() for k,v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

    te_loss, te_acc, (gts, preds) = run_epoch(te_dl, train=False)
    cm = confusion_matrix(gts, preds, labels=list(range(N_CLASSES)))
    rep = classification_report(gts, preds, digits=3, output_dict=True)

    return {
        "test_fold": test_fold,
        "val_fold": val_fold,
        "test_acc": float(te_acc),
        "test_loss": float(te_loss),
        "val_best_acc": float(best_va),
        "report": rep,
        "cm": cm
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn","rnn"], required=True, help="Qual modelo correr para 10 folds")
    parser.add_argument("--epochs", type=int, default=None, help="Override de epochs (senão usa os do treino original)")
    parser.add_argument("--batch", type=int, default=None, help="Override de batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override de learning rate")
    parser.add_argument("--dropout", type=float, default=None, help="Override de dropout")
    args = parser.parse_args()

    model_type = args.model
    epochs = args.epochs
    batch = args.batch
    lr = args.lr
    dropout = args.dropout

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("runs") / f"cv_{model_type}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> Guardar resultados em: {out_dir}")

    results = []
    # tamanho por classe (assumido igual para CNN/RNN)
    N_CLASSES = N_CLASSES_CNN if model_type=="cnn" else N_CLASSES_RNN
    cm_sum = np.zeros((N_CLASSES, N_CLASSES), dtype=int)

    for test_fold in range(1, 11):
        val_fold = (test_fold % 10) + 1  # “próximo” fold
        out = run_one_fold(model_type, test_fold, val_fold, epochs=epochs, batch=batch, lr=lr, dropout=dropout)

        # guardar por fold
        fold_path = out_dir / f"fold{test_fold}_metrics.json"
        with open(fold_path, "w") as f:
            json.dump({
                **{k:v for k,v in out.items() if k not in ["cm","report"]},
                "classification_report": out["report"]
            }, f, indent=2)

        # acumular
        results.append({k:v for k,v in out.items() if k not in ["cm","report"]})
        cm_sum += out["cm"]
        print(f"[{model_type.upper()}] fold {test_fold:02d} | test_acc={out['test_acc']:.3f} | best_val={out['val_best_acc']:.3f}")

    # resumo final
    df = pd.DataFrame(results)
    mean_acc = df["test_acc"].mean()
    std_acc  = df["test_acc"].std(ddof=1)
    print(f"\n[{model_type.upper()}] ACC média (10 folds): {mean_acc:.3f} ± {std_acc:.3f}")

    df.to_csv(out_dir / "summary.csv", index=False)
    np.savetxt(out_dir / "confusion_matrix_cumulative.csv", cm_sum, fmt="%d", delimiter=",")

    # plot CM cumulativa
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_sum, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão Cumulativa (10 folds) — {model_type.upper()}")
    plt.xlabel("Predito"); plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_cumulative.png", dpi=300)
    plt.close()

    # salvar também um JSON com o resumo
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "model": model_type,
            "acc_mean": float(mean_acc),
            "acc_std": float(std_acc),
            "folds": list(range(1,11)),
        }, f, indent=2)

if __name__ == "__main__":
    main()
