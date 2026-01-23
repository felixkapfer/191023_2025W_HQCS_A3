from __future__ import annotations

from typing import Dict, Any
from qiskit import QuantumCircuit


def ops_to_serializable_dict(ops: Dict[Any, Any]) -> Dict[str, int]:
    """Convert Qiskit ops counter to a plain JSON-like dict."""
    return {str(k): int(v) for k, v in ops.items()}


def print_circuit_metrics(title: str, circuit: QuantumCircuit) -> None:
    """
    Print a compact set of circuit metrics that are useful for reports and oral defense.
    """
    ops = circuit.count_ops()
    cx_count = int(ops.get("cx", 0))
    measure_count = int(ops.get("measure", 0))

    # Rough count for single-qubit gates: everything that is not cx/measure/barrier
    one_qubit_count = sum(int(v) for k, v in ops.items() if k not in ("cx", "measure", "barrier"))

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Depth            : {circuit.depth()}")
    print(f"Total gates      : {circuit.size()}")
    print(f"2Q gates (cx)     : {cx_count}")
    print(f"1Q gates (approx) : {one_qubit_count}")
    print(f"Measurements      : {measure_count}")
    print(f"Ops breakdown     : {ops_to_serializable_dict(ops)}")


def _format_logical_qubit_label(circuit: QuantumCircuit, qubit) -> str:
    """
    Return a robust logical label for a qubit (e.g., q[0]).
    This avoids relying on version-specific attributes such as qubit.index.
    """
    logical_idx = circuit.qubits.index(qubit)

    reg_name = None
    reg_index = None

    # Best effort to extract register info across Qiskit versions
    for attr in ("register", "_register"):
        if hasattr(qubit, attr):
            try:
                reg = getattr(qubit, attr)
                if reg is not None and hasattr(reg, "name"):
                    reg_name = reg.name
                    break
            except Exception:
                pass

    for attr in ("index", "_index"):
        if hasattr(qubit, attr):
            try:
                reg_index = int(getattr(qubit, attr))
                break
            except Exception:
                pass

    if reg_name is not None and reg_index is not None:
        return f"{reg_name}[{reg_index}]"

    return f"logical#{logical_idx}"


def print_layout_mapping(title: str, circuit: QuantumCircuit) -> None:
    """
    Print the logical-to-physical qubit mapping produced by transpilation.
    """
    print(f"\n{title} (logical -> physical)")
    print("-" * (len(title) + 21))

    if not hasattr(circuit, "layout") or circuit.layout is None:
        print("No layout information available.")
        return

    mapping = circuit.layout.input_qubit_mapping  # logical qubit object -> physical index
    items = sorted(mapping.items(), key=lambda kv: kv[1])

    for logical_qubit, physical_index in items:
        label = _format_logical_qubit_label(circuit, logical_qubit)
        print(f"  {label} -> phys {physical_index}")
