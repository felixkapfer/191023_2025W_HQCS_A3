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
