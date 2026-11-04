import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.metrics import calculate_ece

def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, save_path: str = "calibration_diagram.png"):
    """Generates and saves a model Reliability Diagram."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(0.0)
            bin_confs.append(bin_boundaries[i] + 0.5 / n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_boundaries[:-1], bin_accs, width=1.0/n_bins, align='edge', alpha=0.7, edgecolor='black', label='Outputs')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Calibration evaluation utility ready.")
