import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def run_exact_mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray):
    """Computes exact McNemar test between two classifiers."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = np.sum(correct_a & ~correct_b)  # Model A right, Model B wrong
    c = np.sum(~correct_a & correct_b)  # Model A wrong, Model B right

    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return result.pvalue, b, c

def paired_bootstrap_ci(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05):
    """Computes paired bootstrap 95% confidence intervals for accuracy difference."""
    n_samples = len(y_true)
    diffs = []
    
    np.random.seed(2024)
    for _ in range(n_resamples):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        acc_a = np.mean(pred_a[indices] == y_true[indices])
        acc_b = np.mean(pred_b[indices] == y_true[indices])
        diffs.append((acc_a - acc_b) * 100.0)

    lower = np.percentile(diffs, 100 * (alpha / 2))
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return np.mean(diffs), (lower, upper)

if __name__ == "__main__":
    y_true = np.random.randint(0, 5, 3057)
    p_a = y_true.copy()
    p_b = y_true.copy()
    p_a[::10] = (p_a[::10] + 1) % 5

    pval, b, c = run_exact_mcnemar_test(y_true, p_a, p_b)
    diff_mean, (ci_low, ci_high) = paired_bootstrap_ci(y_true, p_a, p_b)

    print(f"McNemar p-value: {pval:.5f} (Discordant pairs: {b}/{c})")
    print(f"Bootstrap Acc Diff: {diff_mean:.2f}% [95% CI: {ci_low:.2f}%, {ci_high:.2f}%]")
