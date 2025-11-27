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
ROOT = os.getenv("US8K_ROOT")  # definido no .env
assert ROOT and os.path.exists(ROOT), f"US8K_ROOT inválido: {ROOT}"
CSV  = os.path.join(ROOT, "metadata", "UrbanSound8K.csv")
assert os.path.exists(CSV), f"CSV não encontrado: {CSV}"

# 1ª iteração fixa
TEST_FOLD = 1
VAL_FOLD  = 10
TRAIN_FOLDS = [f for f in range(1, 11) if f not in (TEST_FOLD, VAL_FOLD)]

# áudio / features
SR, DUR = 22050, 4.0
N_MELS, N_FFT, HOP = 64, 1024, 512
USE_MFCC, N_MFCC = False, 40

# Mixup (default on for treino)
MIXUP = bool(int(os.getenv("CNN_MIXUP", 1))) # enable via env CNN_MIXUP=1 or disable via CNN_MIXUP=0
MIXUP_ALPHA = float(os.getenv("CNN_MIXUP_ALPHA", 0.3))
MIXUP_PROB = float(os.getenv("CNN_MIXUP_PROB", 1.0))

# SpecAugment (default off; enable via env SPEC_AUG=1)
SPEC_AUG = bool(int(os.getenv("SPEC_AUG", 0)))
SPEC_AUG_PROB = float(os.getenv("SPEC_AUG_PROB", 0.25))
SPEC_AUG_TIME_MASKS = int(os.getenv("SPEC_AUG_TIME_MASKS", 1))
SPEC_AUG_FREQ_MASKS = int(os.getenv("SPEC_AUG_FREQ_MASKS", 1))
SPEC_AUG_MAX_TIME = int(os.getenv("SPEC_AUG_MAX_TIME", 6))
SPEC_AUG_MAX_FREQ = int(os.getenv("SPEC_AUG_MAX_FREQ", 6))

# treino (pode ser sobreposto por env: CNN_BATCH, CNN_EPOCHS, CNN_LR, CNN_DROPOUT)
BATCH   = int(os.getenv("CNN_BATCH", 32))
EPOCHS  = int(os.getenv("CNN_EPOCHS", 50))
LR      = float(os.getenv("CNN_LR", 1e-3))
DROPOUT = float(os.getenv("CNN_DROPOUT", 0.3))
WEIGHT_DECAY = float(os.getenv("CNN_WEIGHT_DECAY", 1e-4))
PATIENCE = int(os.getenv("CNN_PATIENCE", 7))             # early stopping patience (epochs)
MIN_DELTA = float(os.getenv("CNN_MIN_DELTA", 1e-3))      # min loss improvement to reset patience
N_CLASSES = 10

# device 
DEVICE = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# pastas de saída
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

def prepare_run_dir():
    run_name = f"cnn_iter1_fold{TEST_FOLD}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out = RUNS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out

# DATASET
class US8K(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self): return len(self.df)

    def _load(self, row):
        path = os.path.join(ROOT, "audio", f"fold{row.fold}", row.slice_file_name)
        y, _ = librosa.load(path, sr=SR, mono=True)
        target = int(SR * DUR)
        y = np.pad(y, (0, max(0, target - len(y))))[:target]
        if self.augment:
            if np.random.rand() < 0.3: y = y * np.random.uniform(0.8, 1.2)  # ganho
            if np.random.rand() < 0.3: y = y + 0.004 * np.random.randn(len(y))  # ruído
        return y

    def _spec_augment(self, F):
        F = F.copy()
        n_mels, n_frames = F.shape
        for _ in range(SPEC_AUG_FREQ_MASKS):
            width = np.random.randint(0, SPEC_AUG_MAX_FREQ + 1)
            if width == 0 or width >= n_mels: continue
            start = np.random.randint(0, n_mels - width + 1)
            F[start:start + width, :] = 0.0
        for _ in range(SPEC_AUG_TIME_MASKS):
            width = np.random.randint(0, SPEC_AUG_MAX_TIME + 1)
            if width == 0 or width >= n_frames: continue
            start = np.random.randint(0, n_frames - width + 1)
            F[:, start:start + width] = 0.0
        return F

    def _feat(self, y):
        if USE_MFCC:
            F = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
        else:
            S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
            F = librosa.power_to_db(S, ref=np.max)
        F = librosa.util.normalize(F).astype(np.float32)    # (freq,time)
        if self.augment and SPEC_AUG and (np.random.rand() < SPEC_AUG_PROB):
            F = self._spec_augment(F)
        return torch.from_numpy(F).unsqueeze(0)             # (1,f,t)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        y = self._load(row)
        x = self._feat(y)
        label = int(row["classID"])
        return x, label

# MODEL
class AudioCNN(nn.Module):
    def __init__(self, n_classes=10, dropout=DROPOUT):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.cls(self.fe(x))

# UTILS 
def make_splits():
    meta = pd.read_csv(CSV)
    tr = meta[meta.fold.isin(TRAIN_FOLDS)]
    va = meta[meta.fold == VAL_FOLD]
    te = meta[meta.fold == TEST_FOLD]
    return tr, va, te

def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(crit, preds, y_a, y_b, lam):
    return lam * crit(preds, y_a) + (1 - lam) * crit(preds, y_b)

def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train(train)
    losses, preds, gts = [], [], []
    for xb, yb in tqdm(loader, desc="train" if train else "eval", leave=False):
        xb = xb.to(DEVICE, non_blocking=True)
        yb = torch.as_tensor(yb, device=DEVICE, dtype=torch.long)
        use_mixup = train and MIXUP and (np.random.rand() < MIXUP_PROB)
        orig_yb = yb
        if use_mixup:
            xb, yb_a, yb_b, lam = mixup_data(xb, yb, MIXUP_ALPHA)
        logits = model(xb)
        if train and use_mixup:
            loss = mixup_criterion(crit, logits, yb_a, yb_b, lam)
        else:
            loss = crit(logits, yb)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        preds += logits.argmax(1).detach().cpu().tolist()
        gts   += orig_yb.detach().cpu().tolist()
    return float(np.mean(losses)), accuracy_score(gts, preds), (gts, preds)

# MAIN
def main():
    print("Device:", DEVICE)
    OUT = prepare_run_dir()
    # salvar config usada
    config = {
        "dataset_root": ROOT, "csv": CSV,
        "folds": {"train": TRAIN_FOLDS, "val": VAL_FOLD, "test": TEST_FOLD},
        "audio": {"sr": SR, "dur": DUR, "n_mels": N_MELS, "n_fft": N_FFT, "hop": HOP, "use_mfcc": USE_MFCC, "n_mfcc": N_MFCC},
        "augment": {
            "wave": {"gain_prob": 0.3, "noise_prob": 0.3},
            "specaugment": {
                "enabled": SPEC_AUG, "prob": SPEC_AUG_PROB,
                "time_masks": SPEC_AUG_TIME_MASKS, "freq_masks": SPEC_AUG_FREQ_MASKS,
                "max_time": SPEC_AUG_MAX_TIME, "max_freq": SPEC_AUG_MAX_FREQ
            },
            "mixup": {"enabled": MIXUP, "alpha": MIXUP_ALPHA, "prob": MIXUP_PROB}
        },
        "train": {
            "batch": BATCH, "epochs": EPOCHS, "lr": LR, "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
            "early_stopping": {"monitor": "val_loss", "patience": PATIENCE, "min_delta": MIN_DELTA}
        },
        "device": DEVICE, "seed": SEED, "model": "AudioCNN"
    }
    with open(OUT / "config.json", "w") as f: json.dump(config, f, indent=2)

    tr_df, va_df, te_df = make_splits()
    print(f"Train={len(tr_df)} | Val={len(va_df)} | Test(fold{TEST_FOLD})={len(te_df)}")
    if DEVICE == "cpu" or DEVICE == "cuda":
        num_workers = 6
    else:
        num_workers = 0
    # num_workers=0 
    tr_dl = DataLoader(US8K(tr_df, augment=True),  batch_size=BATCH, shuffle=True,  num_workers=num_workers, pin_memory=True)
    va_dl = DataLoader(US8K(va_df, augment=False), batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=True)
    te_dl = DataLoader(US8K(te_df, augment=False), batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = AudioCNN(N_CLASSES).to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # histórico
    hist_csv = OUT / "history.csv"
    with open(hist_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","train_acc","train_loss","val_acc","val_loss"])

    best = None
    best_metrics = {"val_loss": float("inf"), "val_acc": 0.0}
    wait = 0
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc, _ = run_epoch(model, tr_dl, crit, opt)
        va_loss, va_acc, _ = run_epoch(model, va_dl, crit)
        print(f"Epoch {ep:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f} | val acc {va_acc:.3f} loss {va_loss:.3f}")

        # log linha
        with open(hist_csv, "a", newline="") as f:
            w = csv.writer(f); w.writerow([ep, tr_acc, tr_loss, va_acc, va_loss])

        # checkpoint do melhor val (monitoriza loss)
        if va_loss + MIN_DELTA < best_metrics["val_loss"]:
            best_metrics = {"val_loss": va_loss, "val_acc": va_acc}
            best = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best, OUT / "model_best.pt")
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping: sem melhoria de val_loss por {PATIENCE} épocas (min_delta={MIN_DELTA})")
                break

    # carrega melhor antes de testar
    if best: model.load_state_dict({k: v.to(DEVICE) for k, v in best.items()})

    te_loss, te_acc, (gts, preds) = run_epoch(model, te_dl, crit)
    print(f"[TEST fold{TEST_FOLD}] acc {te_acc:.3f} | loss {te_loss:.3f}")

    cm = confusion_matrix(gts, preds, labels=list(range(N_CLASSES)))
    rep = classification_report(gts, preds, digits=3, output_dict=True)

    # guarda métricas finais
    out_json = {
        "test": {"acc": float(te_acc), "loss": float(te_loss)},
        "best_val": {"acc": float(best_metrics["val_acc"]), "loss": float(best_metrics["val_loss"])},
        "confusion_matrix": cm.tolist(),
        "classification_report": rep
    }
    with open(OUT / "results.json", "w") as f: json.dump(out_json, f, indent=2)

    # CSV da CM
    np.savetxt(OUT / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    print(f"\n Guardado em: {OUT.resolve()}")
    print(" - model_best.pt")
    print(" - history.csv")
    print(" - results.json")
    print(" - confusion_matrix.csv")

if __name__ == "__main__":
    main()
