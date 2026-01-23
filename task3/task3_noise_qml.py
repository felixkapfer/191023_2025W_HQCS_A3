"""
Task 3 (BONUS) – QML + Classical Shadows (Noise Study)
------------------------------------------------------
Goal:
- Introduce simple noise (depolarizing or amplitude damping) into a QML model
- Compare for the three village datasets:
  * training loss curves with/without noise
  * gradient norms over time
  * accuracy degradation due to noise
  * decision boundaries (2D)

Notes:
- Task 1 remains untouched.
- This script is standalone and reads CSVs from ./task1
- Saves plots to ./results/task3/plots
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as qnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = os.path.join("task1")
RESULTS_DIR = os.path.join("results", "task3")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

VILLAGES = {
    "gluehweindorf": os.path.join(DATA_DIR, "gluehweindorf.csv"),
    "krampuskogel": os.path.join(DATA_DIR, "krampuskogel.csv"),
    "lebkuchenstadt": os.path.join(DATA_DIR, "lebkuchenstadt.csv"),
}

FEATURES = ["carol_singing", "snowball_energy"]  # matches your Task 1 notebook
LABEL_COL = "label"

SEED = 7
TEST_SIZE = 0.25

EPOCHS = 80
BATCH_SIZE = 16
LR = 0.05

# Re-uploading values requested to discuss
REUPLOAD_COUNTS = [1, 3]

# Noise configs: choose one (or keep both)
NOISE_SETUPS = [
    ("noiseless", None, 0.0),
    ("depolarizing", "depolarizing", 0.02),
    ("amplitude_damping", "amplitude_damping", 0.05),
]


# -----------------------------
# Helper dataclasses
# -----------------------------
@dataclass
class TrainHistory:
    losses: List[float]
    grad_norms: List[float]
    train_acc: List[float]
    test_acc: List[float]


# -----------------------------
# Data loading / preprocessing
# -----------------------------
def load_dataset(csv_path: str) -> Tuple[qnp.ndarray, qnp.ndarray]:
    df = pd.read_csv(csv_path)
    X = df[FEATURES].values
    y = df[LABEL_COL].values.astype(int)
    return X, y


def preprocess_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(0.0, qnp.pi))  # good for angle encoding
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        qnp.array(X_train, requires_grad=False),
        qnp.array(X_test, requires_grad=False),
        qnp.array(y_train, requires_grad=False),
        qnp.array(y_test, requires_grad=False),
        scaler,
    )


# -----------------------------
# Model: Encoding + Trainable + Noise
# -----------------------------
def apply_angle_encoding(x, wires):
    # 2 features -> apply RX, RY on each qubit (simple, consistent with your notebook style)
    for w in wires:
        qml.RX(x[0], wires=w)
        qml.RY(x[1], wires=w)


def apply_entanglement(wires):
    # linear entanglement
    if len(wires) > 1:
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def apply_noise(noise_type: str | None, p: float, wires):
    if noise_type is None or p <= 0.0:
        return

    # Apply after each layer to mimic noisy dynamics in training
    for w in wires:
        if noise_type == "depolarizing":
            qml.DepolarizingChannel(p, wires=w)
        elif noise_type == "amplitude_damping":
            qml.AmplitudeDamping(p, wires=w)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")


def make_qnode(num_qubits: int, noise_type: str | None, noise_p: float):
    """
    Use default.mixed for both noiseless and noisy runs (noise_p=0 yields noiseless dynamics)
    to keep the simulation model consistent.
    """
    dev = qml.device("default.mixed", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights, x, reupload_count: int):
        wires = list(range(num_qubits))

        for layer in range(reupload_count):
            apply_angle_encoding(x, wires=wires)
            apply_entanglement(wires=wires)

            # Trainable layer: one RY per qubit per layer (simple & stable)
            for i, w in enumerate(wires):
                qml.RY(weights[layer, i], wires=w)

            apply_noise(noise_type, noise_p, wires=wires)

        # Binary classification via expectation value in [-1, 1]
        return qml.expval(qml.PauliZ(0))

    return circuit


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + qnp.exp(-z))


def model_prob(circuit, weights, x, reupload_count):
    # expval in [-1,1]; map to probability via sigmoid (simple calibration)
    z = circuit(weights, x, reupload_count)
    return sigmoid(z)


def bce_loss(circuit, weights, X, y, reupload_count):
    eps = 1e-9
    probs = qnp.array([model_prob(circuit, weights, x, reupload_count) for x in X])
    # y expected 0/1
    loss = -qnp.mean(y * qnp.log(probs + eps) + (1 - y) * qnp.log(1 - probs + eps))
    return loss


def predict_labels(circuit, weights, X, reupload_count, threshold=0.5):
    probs = qnp.array([model_prob(circuit, weights, x, reupload_count) for x in X])
    return (probs >= threshold).astype(int)


# -----------------------------
# Training loop + required metrics
# -----------------------------
def iterate_minibatches(X, y, batch_size: int):
    n = len(X)
    idx = qnp.random.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_one_setting(
    X_train, y_train, X_test, y_test,
    reupload_count: int,
    noise_type: str | None,
    noise_p: float,
) -> Tuple[TrainHistory, qnp.ndarray, Callable]:
    num_qubits = 2
    circuit = make_qnode(num_qubits=num_qubits, noise_type=noise_type, noise_p=noise_p)

    # weights: shape (reupload_count, num_qubits)
    weights = qnp.array(
        qnp.random.uniform(0.0, qnp.pi, size=(reupload_count, num_qubits)),
        requires_grad=True,
    )

    opt = qml.AdamOptimizer(stepsize=LR)

    history = TrainHistory(losses=[], grad_norms=[], train_acc=[], test_acc=[])

    # gradient of loss w.r.t. weights
    grad_fn = qml.grad(lambda w, xb, yb: bce_loss(circuit, w, xb, yb, reupload_count))

    for epoch in range(EPOCHS):
        # Mini-batch updates
        for xb, yb in iterate_minibatches(X_train, y_train, BATCH_SIZE):
            weights = opt.step(lambda w: bce_loss(circuit, w, xb, yb, reupload_count), weights)

        # Metrics after epoch on full sets
        train_loss = float(bce_loss(circuit, weights, X_train, y_train, reupload_count))
        grads = grad_fn(weights, X_train, y_train)
        grad_norm = float(qnp.linalg.norm(grads))

        yhat_train = predict_labels(circuit, weights, X_train, reupload_count)
        yhat_test = predict_labels(circuit, weights, X_test, reupload_count)

        train_acc = float(accuracy_score(y_train, yhat_train))
        test_acc = float(accuracy_score(y_test, yhat_test))

        history.losses.append(train_loss)
        history.grad_norms.append(grad_norm)
        history.train_acc.append(train_acc)
        history.test_acc.append(test_acc)

    return history, weights, circuit


# -----------------------------
# Plotting utilities (PNG export)
# -----------------------------
def save_loss_and_grad_plots(village: str, reupload: int, histories: Dict[str, TrainHistory]):
    # Loss curves
    plt.figure(figsize=(10, 4))
    for label, h in histories.items():
        plt.plot(h.losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title(f"{village} – Loss curves (reupload={reupload})")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{village}_reupload{reupload}_loss.png")
    plt.savefig(out, dpi=200)
    plt.close()

    # Gradient norms
    plt.figure(figsize=(10, 4))
    for label, h in histories.items():
        plt.plot(h.grad_norms, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient norm (L2)")
    plt.title(f"{village} – Gradient norms (reupload={reupload})")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{village}_reupload{reupload}_gradnorm.png")
    plt.savefig(out, dpi=200)
    plt.close()


def save_accuracy_plot(village: str, reupload: int, histories: Dict[str, TrainHistory]):
    plt.figure(figsize=(10, 4))
    for label, h in histories.items():
        plt.plot(h.test_acc, label=f"{label} (test)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{village} – Test accuracy over time (reupload={reupload})")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{village}_reupload{reupload}_testacc.png")
    plt.savefig(out, dpi=200)
    plt.close()


def save_decision_boundary(
    village: str,
    reupload: int,
    label: str,
    circuit,
    weights,
    scaler: MinMaxScaler,
    X_train,
    y_train,
):
    # grid in original feature space, then transform via scaler to [0, pi]
    x0_min, x0_max = float(qnp.min(X_train[:, 0])), float(qnp.max(X_train[:, 0]))
    x1_min, x1_max = float(qnp.min(X_train[:, 1])), float(qnp.max(X_train[:, 1]))

    # Expand slightly
    pad0 = 0.05 * (x0_max - x0_min + 1e-9)
    pad1 = 0.05 * (x1_max - x1_min + 1e-9)
    x0_min, x0_max = x0_min - pad0, x0_max + pad0
    x1_min, x1_max = x1_min - pad1, x1_max + pad1

    # Build grid in scaled space directly (since X_train already scaled to [0, pi])
    # Use that space for plotting boundaries consistently.
    gx0 = qnp.linspace(x0_min, x0_max, 120)
    gx1 = qnp.linspace(x1_min, x1_max, 120)

    Z = qnp.zeros((len(gx1), len(gx0)))

    for i, yy in enumerate(gx1):
        for j, xx in enumerate(gx0):
            p = model_prob(circuit, weights, qnp.array([xx, yy]), reupload)
            Z[i, j] = p

    plt.figure(figsize=(6, 5))
    plt.contourf(gx0, gx1, Z, levels=30)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=18)
    plt.xlabel(FEATURES[0] + " (scaled)")
    plt.ylabel(FEATURES[1] + " (scaled)")
    plt.title(f"{village} – Decision boundary ({label}, reupload={reupload})")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{village}_reupload{reupload}_boundary_{label}.png")
    plt.savefig(out, dpi=200)
    plt.close()


# -----------------------------
# Main experiment runner
# -----------------------------
def main():
    print("=== Task 3: QML Noise Study ===")
    print(f"Reading datasets from: {DATA_DIR}")
    print(f"Saving plots to       : {PLOTS_DIR}")

    summary_rows = []

    for village, path in VILLAGES.items():
        X, y = load_dataset(path)
        X_train, X_test, y_train, y_test, scaler = preprocess_split(X, y)

        for reupload in REUPLOAD_COUNTS:
            histories_by_label: Dict[str, TrainHistory] = {}
            final_models = {}

            for label, noise_type, noise_p in NOISE_SETUPS:
                print(f"\n[{village}] reupload={reupload} | setting={label} | noise_p={noise_p}")

                hist, w_opt, circ = train_one_setting(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    reupload_count=reupload,
                    noise_type=noise_type,
                    noise_p=noise_p,
                )

                histories_by_label[label] = hist
                final_models[label] = (w_opt, circ)

                summary_rows.append({
                    "village": village,
                    "reupload": reupload,
                    "setting": label,
                    "final_train_loss": hist.losses[-1],
                    "final_train_acc": hist.train_acc[-1],
                    "final_test_acc": hist.test_acc[-1],
                    "final_grad_norm": hist.grad_norms[-1],
                })

            # Required plots
            save_loss_and_grad_plots(village, reupload, histories_by_label)
            save_accuracy_plot(village, reupload, histories_by_label)

            # Decision boundaries (plot for noiseless vs one noisy setting)
            # Use noiseless + depolarizing if available
            for boundary_label in ["noiseless", "depolarizing"]:
                if boundary_label in final_models:
                    w_opt, circ = final_models[boundary_label]
                    save_decision_boundary(
                        village=village,
                        reupload=reupload,
                        label=boundary_label,
                        circuit=circ,
                        weights=w_opt,
                        scaler=scaler,
                        X_train=X_train,
                        y_train=y_train,
                    )

    # Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(RESULTS_DIR, "task3_summary.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Wrote summary to: {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
