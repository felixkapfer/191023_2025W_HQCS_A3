"""
Task 4.1 – Bernstein–Vazirani: Circuit Creation + Compilation (opt 0 vs opt 3)
-----------------------------------------------------------------------------
- 5 Qubits total: 4 data qubits + 1 ancilla
- Hidden string explicitly set
- Compilation via Qiskit's preset pass manager (IBM backend)
- Metrics + layout mapping printed for report/oral defense
- Circuit diagrams saved as PNG under ./results/task4/images/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure that ./src is on the Python path so that "utils" can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from utils.circuit_reporting import print_circuit_metrics, print_layout_mapping
from utils.circuit_rendering import save_circuit_png_with_text_fallback





# -----------------------------
# Task 4.1 - Configuration
# -----------------------------
N_DATA_QUBITS = 4                 # 4 data qubits
ANCILLA_QUBIT = N_DATA_QUBITS     # ancilla is the 5th qubit -> total 5 qubits
HIDDEN_STRING = "1011"            # length = 4

RESULTS_DIR = os.path.join("results", "task4", "images")
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# 1) Build Bernstein–Vazirani circuit
# -----------------------------
circuit = QuantumCircuit(N_DATA_QUBITS + 1, N_DATA_QUBITS)

# Prepare ancilla in |-> : |0> --X--> |1> --H--> |->
circuit.x(ANCILLA_QUBIT)
circuit.h(ANCILLA_QUBIT)

# Put data qubits into uniform superposition
for i in range(N_DATA_QUBITS):
    circuit.h(i)

# Oracle implementation: apply CX(i -> ancilla) for each s_i = 1
for i, bit in enumerate(HIDDEN_STRING):
    if bit == "1":
        circuit.cx(i, ANCILLA_QUBIT)

# Interference step: Hadamard on data qubits
for i in range(N_DATA_QUBITS):
    circuit.h(i)

# Measure only the data qubits
for i in range(N_DATA_QUBITS):
    circuit.measure(i, i)

print("\n=== ORIGINAL CIRCUIT (text) ===")
print(circuit.draw(output="text", fold=120))


# -----------------------------
# 2) Compile via preset pass manager (opt level 0 and 3)
# -----------------------------
backend = FakeManilaV2()

pass_manager_0 = generate_preset_pass_manager(backend=backend, optimization_level=0)
pass_manager_3 = generate_preset_pass_manager(backend=backend, optimization_level=3)

compiled_0 = pass_manager_0.run(circuit)
compiled_3 = pass_manager_3.run(circuit)


# -----------------------------
# 3) Print metrics and layout mapping
# -----------------------------
print_circuit_metrics("Compiled (optimization level 0)", compiled_0)
print_layout_mapping("Layout mapping (optimization level 0)", compiled_0)

print_circuit_metrics("Compiled (optimization level 3)", compiled_3)
print_layout_mapping("Layout mapping (optimization level 3)", compiled_3)


# -----------------------------
# 4) Save PNG diagrams for the report
# -----------------------------
save_circuit_png_with_text_fallback(
    circuit=circuit,
    output_path=os.path.join(RESULTS_DIR, "task4_1_bv_original.png"),
    title="Task 4.1 – Bernstein–Vazirani (Original)",
)

save_circuit_png_with_text_fallback(
    circuit=compiled_0,
    output_path=os.path.join(RESULTS_DIR, "task4_1_bv_compiled_opt0.png"),
    title="Task 4.1 – Bernstein–Vazirani (Compiled, opt level 0)",
)

save_circuit_png_with_text_fallback(
    circuit=compiled_3,
    output_path=os.path.join(RESULTS_DIR, "task4_1_bv_compiled_opt3.png"),
    title="Task 4.1 – Bernstein–Vazirani (Compiled, opt level 3)",
)

print(f"\nDone. Images saved to: {RESULTS_DIR}/")







# ============================================================
# Task 4.3 – Execute compiled circuits and create result plots
# ============================================================

import matplotlib.pyplot as plt
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import Statevector


# -----------------------------
# Task 4.3 - Configuration
# -----------------------------
SHOTS = 4096

PLOTS_DIR = os.path.join("results", "task4", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------
# 1) Ideal distribution (noise-free, statevector)
# -----------------------------
# Use the *logical* circuit without measurements for the ideal reference.
ideal_no_meas = circuit.remove_final_measurements(inplace=False)
sv = Statevector.from_instruction(ideal_no_meas)

# probabilities_dict returns bitstrings for the measured qargs.
ideal_probs = sv.probabilities_dict(qargs=list(range(N_DATA_QUBITS)))
ideal_dist = {k: float(v) for k, v in ideal_probs.items()}


# -----------------------------
# 2) Noisy "device-like" execution using FakeManila noise model (Aer)
# -----------------------------
noise_model = NoiseModel.from_backend(backend)
aer_sim = Aer.get_backend("aer_simulator")

job0 = aer_sim.run(compiled_0, noise_model=noise_model, shots=SHOTS)
job3 = aer_sim.run(compiled_3, noise_model=noise_model, shots=SHOTS)

counts0 = job0.result().get_counts()
counts3 = job3.result().get_counts()

dist0 = {k: v / SHOTS for k, v in counts0.items()}
dist3 = {k: v / SHOTS for k, v in counts3.items()}


# -----------------------------
# 3) Helper: consistent key set + plotting
# -----------------------------
def _all_keys(*dicts):
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    return sorted(keys)

def _barplot_distribution(ax, keys, dist, title):
    values = [dist.get(k, 0.0) for k in keys]
    ax.bar(keys, values)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probability")

keys = _all_keys(ideal_dist, dist0, dist3)


# -----------------------------
# 4) Single combined comparison plot (recommended for report)
# -----------------------------
x = list(range(len(keys)))
bar_width = 0.28

y_ideal = [ideal_dist.get(k, 0.0) for k in keys]
y_opt0  = [dist0.get(k, 0.0) for k in keys]
y_opt3  = [dist3.get(k, 0.0) for k in keys]

plt.figure(figsize=(12, 5))
plt.bar([i - bar_width for i in x], y_ideal, width=bar_width, label="Ideal (statevector)")
plt.bar([i for i in x],             y_opt0,  width=bar_width, label="Noisy run (compiled, opt 0)")
plt.bar([i + bar_width for i in x], y_opt3,  width=bar_width, label="Noisy run (compiled, opt 3)")

plt.xticks(x, keys)
plt.ylim(0, 1.0)
plt.xlabel("Measured bitstring")
plt.ylabel("Probability")
plt.title("Task 4.3 – BV results: Ideal vs compiled (FakeManila noise model)")
plt.legend()
plt.tight_layout()

combined_plot_path = os.path.join(PLOTS_DIR, "task4_3_bv_results_comparison.png")
plt.savefig(combined_plot_path, dpi=200, bbox_inches="tight")
print(f"[OK] Saved combined comparison plot: {combined_plot_path}")
plt.show()


# -----------------------------
# 5) Optional: three separate plots (nice for appendix)
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

_barplot_distribution(axes[0], keys, ideal_dist, "Ideal distribution (statevector)")
_barplot_distribution(axes[1], keys, dist0, "Noisy execution – compiled circuit (opt level 0)")
_barplot_distribution(axes[2], keys, dist3, "Noisy execution – compiled circuit (opt level 3)")

axes[2].set_xlabel("Measured bitstring")
plt.tight_layout()

separate_plot_path = os.path.join(PLOTS_DIR, "task4_3_bv_results_three_panels.png")
plt.savefig(separate_plot_path, dpi=200, bbox_inches="tight")
print(f"[OK] Saved three-panel plot: {separate_plot_path}")
plt.show()
