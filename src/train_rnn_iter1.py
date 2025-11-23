import os, json, csv, warnings, numpy as np, pandas as pd, librosa, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
load_dotenv()

# CONFIG
ROOT = os.getenv("US8K_ROOT")
assert ROOT and os.path.exists(ROOT), f"US8K_ROOT inválido: {ROOT}"
CSV  = os.path.join(ROOT, "metadata", "UrbanSound8K.csv")
assert os.path.exists(CSV), f"CSV não encontrado: {CSV}"

# Folds (1ª iteração): test=1, val=10, train=2-9
TEST_FOLD = 1
VAL_FOLD  = 10
TRAIN_FOLDS = [f for f in range(1, 11) if f not in (TEST_FOLD, VAL_FOLD)]

# Áudio / features 
SR, DUR = 22050, 4.0
N_MELS, N_FFT, HOP = 64, 1024, 512   # sequência ~173 passos 
USE_DB = True                        # usar log-power 

# Treino (pode ser sobreposto por env: RNN_BATCH, RNN_EPOCHS, RNN_LR, RNN_DROPOUT, RNN_HIDDEN, RNN_LAYERS)
BATCH    = int(os.getenv("RNN_BATCH", 32))
EPOCHS   = int(os.getenv("RNN_EPOCHS", 50))
LR       = float(os.getenv("RNN_LR", 1e-3))
PATIENCE = int(os.getenv("RNN_PATIENCE", 7))             # early stopping patience (epochs)
MIN_DELTA = float(os.getenv("RNN_MIN_DELTA", 1e-3))      # min val loss improvement
N_CLASSES = 10
HIDDEN   = int(os.getenv("RNN_HIDDEN", 128))
N_LAYERS = int(os.getenv("RNN_LAYERS", 2))
BIDIR    = True
DROPOUT  = float(os.getenv("RNN_DROPOUT", 0.2))

DEVICE = ("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Saída
RUNS = Path("runs")
RUNS.mkdir(exist_ok=True)

def prepare_run_dir():
    run = RUNS / f"rnn_iter1_fold{TEST_FOLD}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    return run

# DATASET
class US8KSeq(Dataset):
    """
    Devolve (seq, label):
      seq: Tensor [T, n_mels]  (T frames ao longo do tempo; n_mels features por frame)
      label: int
    """
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self): return len(self.df)

    def _load_audio(self, row):
        path = os.path.join(ROOT, "audio", f"fold{row.fold}", row.slice_file_name)
        y, _ = librosa.load(path, sr=SR, mono=True)
        target = int(SR * DUR)
        y = np.pad(y, (0, max(0, target - len(y))))[:target]
        if self.augment:
            if np.random.rand() < 0.3: y *= np.random.uniform(0.8, 1.2)
            if np.random.rand() < 0.3: y += 0.004 * np.random.randn(len(y))
        return y

    def _to_seq(self, y):
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                           n_fft=N_FFT, hop_length=HOP)
        if USE_DB:
            S = librosa.power_to_db(S, ref=np.max)
        S = librosa.util.normalize(S)                # (n_mels, T)
        seq = torch.tensor(S.T, dtype=torch.float32) # (T, n_mels)
        return seq

    def __getitem__(self, i):
        row = self.df.iloc[i]
        y = self._load_audio(row)
        x = self._to_seq(y)
        label = int(row["classID"])
        return x, label

# MODELO
class AudioGRU(nn.Module):
    """
    Entrada: batch de sequências [B, T, n_mels]
    GRU bidirecional - último estado - cabeça linear
    """
    def __init__(self, n_mels=N_MELS, hidden=HIDDEN, n_layers=N_LAYERS,
                 bidir=BIDIR, dropout=DROPOUT, n_classes=N_CLASSES):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_mels,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidir
        )
        feat = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(dropout),
            nn.Linear(feat, n_classes)
        )

    def forward(self, x):            # x: [B, T, n_mels]
        out, h = self.gru(x)         # out: [B, T, D]; h: [layers*dir, B, H]
        last = out[:, -1, :]         # usa o último passo temporal
        return self.head(last)

# UTILS
def make_splits():
    meta = pd.read_csv(CSV)
    tr = meta[meta.fold.isin(TRAIN_FOLDS)]
    va = meta[meta.fold == VAL_FOLD]
    te = meta[meta.fold == TEST_FOLD]
    return tr, va, te

def collate_pad(batch):
    """
    Como todas as sequências têm mesmo T (pad/crop 4s), podemos empilhar direto.
    Mantemos collate simples para desempenho.
    """
    xs, ys = zip(*batch)
    X = torch.stack(xs, dim=0)    # [B, T, n_mels]
    y = torch.tensor(ys, dtype=torch.long)
    return X, y

def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train(train)
    losses, preds, gts = [], [], []
    for xb, yb in tqdm(loader, disable=False):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits, yb)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        preds += logits.argmax(1).detach().cpu().tolist()
        gts   += yb.detach().cpu().tolist()
    return float(np.mean(losses)), accuracy_score(gts, preds), (gts, preds)

# MAIN
def main():
    print("Device:", DEVICE)
    RUN = prepare_run_dir()
    tr_df, va_df, te_df = make_splits()
    print(f"Train={len(tr_df)} | Val={len(va_df)} | Test(fold{TEST_FOLD})={len(te_df)}")

    # guardar config usada (paridade com 1ª iteração da CNN)
    config = {
        "dataset_root": ROOT, "csv": CSV,
        "folds": {"train": TRAIN_FOLDS, "val": VAL_FOLD, "test": TEST_FOLD},
        "audio": {"sr": SR, "dur": DUR, "n_mels": N_MELS, "n_fft": N_FFT, "hop": HOP, "log_db": USE_DB},
        "model": {"type": "GRU", "hidden": HIDDEN, "layers": N_LAYERS, "bidir": BIDIR, "dropout": DROPOUT},
        "train": {
            "batch": BATCH, "epochs": EPOCHS, "lr": LR, "dropout": DROPOUT,
            "early_stopping": {"monitor": "val_loss", "patience": PATIENCE, "min_delta": MIN_DELTA}
        },
        "device": DEVICE, "seed": SEED
    }
    with open(RUN / "config.json", "w") as f: json.dump(config, f, indent=2)

    if DEVICE == "mps":
        num_workers = 0
    else:
        num_workers = 4 
    # num_workers=0 para evitar chatices no macOS com librosa
    tr_dl = DataLoader(US8KSeq(tr_df, augment=True),  batch_size=BATCH, shuffle=True,  num_workers=num_workers,
                       pin_memory=True, collate_fn=collate_pad)
    va_dl = DataLoader(US8KSeq(va_df, augment=False), batch_size=BATCH, shuffle=False, num_workers=num_workers,
                       pin_memory=True, collate_fn=collate_pad)
    te_dl = DataLoader(US8KSeq(te_df, augment=False), batch_size=BATCH, shuffle=False, num_workers=num_workers,
                       pin_memory=True, collate_fn=collate_pad)

    model = AudioGRU().to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    # logs
    hist_csv = RUN / "history.csv"
    with open(hist_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_acc","train_loss","val_acc","val_loss"])

    best = None
    best_metrics = {"val_loss": float("inf"), "val_acc": 0.0}
    wait = 0
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc, _ = run_epoch(model, tr_dl, crit, opt)
        va_loss, va_acc, _ = run_epoch(model, va_dl, crit)
        print(f"Epoch {ep:02d} | train {tr_acc:.3f}/{tr_loss:.3f} | val {va_acc:.3f}/{va_loss:.3f}")
        with open(hist_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, tr_acc, tr_loss, va_acc, va_loss])
        # monitoriza val_loss para early stopping e checkpoint
        if va_loss + MIN_DELTA < best_metrics["val_loss"]:
            best_metrics = {"val_loss": va_loss, "val_acc": va_acc}
            best = {k:v.cpu() for k,v in model.state_dict().items()}
            torch.save(best, RUN / "model_best.pt")
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping: sem melhoria de val_loss por {PATIENCE} épocas (min_delta={MIN_DELTA})")
                break

    if best: model.load_state_dict({k:v.to(DEVICE) for k,v in best.items()})
    te_loss, te_acc, (gts, preds) = run_epoch(model, te_dl, crit)

    cm  = confusion_matrix(gts, preds, labels=list(range(N_CLASSES)))
    rep = classification_report(gts, preds, digits=3, output_dict=True)

    out = {
        "test":{"acc":float(te_acc), "loss":float(te_loss)},
        "best_val": {"acc": float(best_metrics["val_acc"]), "loss": float(best_metrics["val_loss"])},
        "confusion_matrix": cm.tolist(),
        "classification_report": rep,
        "config":{
            "folds":{"train":TRAIN_FOLDS, "val":VAL_FOLD, "test":TEST_FOLD},
            "audio":{"sr":SR, "dur":DUR, "n_mels":N_MELS, "n_fft":N_FFT, "hop":HOP, "log_db":USE_DB},
            "model":{"type":"GRU","hidden":HIDDEN,"layers":N_LAYERS,"bidir":BIDIR,"dropout":DROPOUT},
            "train":{
                "batch":BATCH,"epochs":EPOCHS,"lr":LR,"dropout":DROPOUT,
                "early_stopping":{"monitor":"val_loss","patience":PATIENCE,"min_delta":MIN_DELTA}
            },
            "device":DEVICE
        }
    }
    with open(RUN / "results.json","w") as f: json.dump(out, f, indent=2)
    np.savetxt(RUN / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    print(f"\n[TEST fold{TEST_FOLD}] acc {te_acc:.3f} | loss {te_loss:.3f}")
    print(f"Guardado em: {RUN.resolve()}")

if __name__ == "__main__":
    main()
