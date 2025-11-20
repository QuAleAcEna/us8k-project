import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Encontrar última run RNN
runs = sorted(Path("runs").glob("rnn_iter1_fold1*"))
assert len(runs) > 0, "Nenhuma run RNN encontrada!"
RUN = runs[-1]
print("Plotting results for:", RUN)

plots_dir = RUN / "plots"
plots_dir.mkdir(exist_ok=True)

# Load history
hist = pd.read_csv(RUN / "history.csv")

# Accuracy plot
plt.figure(figsize=(8,4))
plt.plot(hist["train_acc"], label="Train Acc")
plt.plot(hist["val_acc"], label="Val Acc")
plt.title("RNN - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(plots_dir / "accuracy.png")
plt.close()

# Loss plot
plt.figure(figsize=(8,4))
plt.plot(hist["train_loss"], label="Train Loss")
plt.plot(hist["val_loss"], label="Val Loss")
plt.title("RNN - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(plots_dir / "loss.png")
plt.close()

# Confusion Matrix
with open(RUN / "results.json") as f:
    res = json.load(f)

cm = np.array(res["confusion_matrix"])

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("RNN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(plots_dir / "confusion_matrix.png")
plt.close()

print("Plots guardados em:", plots_dir)
