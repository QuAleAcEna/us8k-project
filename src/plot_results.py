import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_results(run_dir):
    run_dir = Path(run_dir)
    history_file = run_dir / "history.csv"
    results_file = run_dir / "results.json"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load files
    history = pd.read_csv(history_file)
    with open(results_file, "r") as f:
        results = json.load(f)
    cm = results["confusion_matrix"]

    # Plot ACC 
    plt.figure()
    plt.plot(history["epoch"], history["train_acc"], label="Train ACC")
    plt.plot(history["epoch"], history["val_acc"], label="Val ACC")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "accuracy.png", dpi=300)

    # Plot LOSS
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "loss.png", dpi=300)

    # Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)

    print(f" Saved plots in {plots_dir}")

if __name__ == "__main__":
    run_folder = sorted(Path("runs").glob("cnn_iter1_fold1*"))[-1]
    plot_results(run_folder)
