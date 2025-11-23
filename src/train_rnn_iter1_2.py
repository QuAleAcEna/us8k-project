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

# BANDA MEL 
N_MELS, N_FFT, HOP = 128, 1024, 512     
USE_DB = True

# Treino 
BATCH, EPOCHS, LR = 32, 15, 1e-3
PATIENCE = int(os.getenv("RNN2_PATIENCE", 7))             # early stopping patience (épocas)
MIN_DELTA = float(os.getenv("RNN2_MIN_DELTA", 1e-3))      # melhoria mínima em val_loss
N_CLASSES = 10

# MODELO
HIDDEN = 256           
N_LAYERS = 2
BIDIR = True
DROPOUT = 0.3          

DEVICE = ("mps" if torch.backends.mps.is_available()  else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Saída
RUNS = Path("runs")
RUNS.mkdir(exist_ok=True)
RUN = RUNS / f"rnn_iter1_fold{TEST_FOLD}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
RUN.mkdir(parents=True, exist_ok=True)


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

    def __len__(self): 
        return len(self.df)

    def _load_audio(self, row):
        path = os.path.join(ROOT, "audio", f"fold{row.fold}", row.slice_file_name)
        y, _ = librosa.load(path, sr=SR, mono=True)
        target = int(SR * DUR)
        y = np.pad(y, (0, max(0, target - len(y))))[:target]

        if self.augment:
            # ganho aleatório
            if np.random.rand() < 0.3:
                y *= np.random.uniform(0.8, 1.2)
            # ruído branco leve
            if np.random.rand() < 0.3:
                y += 0.004 * np.random.randn(len(y))
        return y

    def _spec_augment(self, S):
        """
        SpecAugment simples: masks em frequência e tempo.
        S: numpy array [n_mels, T]
        """
        n_mels, T = S.shape
        S_min = S.min()

        # mask em frequência
        if np.random.rand() < 0.5:
            f = np.random.randint(4, max(5, n_mels // 6))
            f0 = np.random.randint(0, max(1, n_mels - f))
            S[f0:f0+f, :] = S_min

        # mask em tempo
        if np.random.rand() < 0.5:
            t = np.random.randint(5, max(6, T // 6))
            t0 = np.random.randint(0, max(1, T - t))
            S[:, t0:t0+t] = S_min

        return S

    def _to_seq(self, y, apply_augment=False):
        S = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP
        )
        if USE_DB:
            S = librosa.power_to_db(S, ref=np.max)

        # SpecAugment só em treino
        if apply_augment:
            S = self._spec_augment(S)

        S = librosa.util.normalize(S)          # (n_mels, T)
        seq = torch.tensor(S.T, dtype=torch.float32)  # (T, n_mels)
        return seq

    def __getitem__(self, i):
        row = self.df.iloc[i]
        y = self._load_audio(row)
        x = self._to_seq(y, apply_augment=self.augment)
        label = int(row["classID"])
        return x, label



# MODELO
class AudioGRU(nn.Module):
    """
    Entrada: batch de sequências [B, T, n_mels]
    GRU bidirecional + head MLP
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

        # HEAD + REGULARIZAÇÃO
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(dropout),
            nn.Linear(feat, feat),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat, n_classes)
        )

    def forward(self, x):            # x: [B, T, n_mels]
        out, h = self.gru(x)         # out: [B, T, D]
        pooled = out.mean(dim=1)     # [B, D]
        return self.head(pooled)



# UTILS
def make_splits():
    meta = pd.read_csv(CSV)
    tr = meta[meta.fold.isin(TRAIN_FOLDS)]
    va = meta[meta.fold == VAL_FOLD]
    te = meta[meta.fold == TEST_FOLD]
    return tr, va, te

def collate_pad(batch):
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
    tr_df, va_df, te_df = make_splits()
    print(f"Train={len(tr_df)} | Val={len(va_df)} | Test(fold{TEST_FOLD})={len(te_df)}")

    if DEVICE == "mps":
        num_workers = 0
    else:
        num_workers = 4
    tr_dl = DataLoader(US8KSeq(tr_df, augment=True),  batch_size=BATCH, shuffle=True,
                       num_workers=num_workers, pin_memory=True, collate_fn=collate_pad)
    va_dl = DataLoader(US8KSeq(va_df, augment=False), batch_size=BATCH, shuffle=False,
                       num_workers=num_workers, pin_memory=True, collate_fn=collate_pad)
    te_dl = DataLoader(US8KSeq(te_df, augment=False), batch_size=BATCH, shuffle=False,
                       num_workers=num_workers, pin_memory=True, collate_fn=collate_pad)

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
        # early stopping com tolerância (monitoriza val_loss)
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

    if best:
        model.load_state_dict({k:v.to(DEVICE) for k,v in best.items()})
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
                "batch":BATCH,"epochs":EPOCHS,"lr":LR,
                "early_stopping":{"monitor":"val_loss","patience":PATIENCE,"min_delta":MIN_DELTA}
            },
            "device":DEVICE
        }
    }
    with open(RUN / "results.json","w") as f:
        json.dump(out, f, indent=2)
    np.savetxt(RUN / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    print(f"\n[TEST fold{TEST_FOLD}] acc {te_acc:.3f} | loss {te_loss:.3f}")
    print(f"Guardado em: {RUN.resolve()}")

if __name__ == "__main__":
    main()
