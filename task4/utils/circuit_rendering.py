from __future__ import annotations

import os
from qiskit import QuantumCircuit


def save_circuit_png(
    circuit: QuantumCircuit,
    output_path: str,
    title: str | None = None,
    dpi: int = 200,
) -> None:
    """
    Render a circuit with the Matplotlib drawer and save it as a PNG.

    Notes:
    - Requires optional dependencies:
      - matplotlib
      - pylatexenc
    - If those are missing, this function raises the underlying exception.
    """
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = circuit.draw(output="mpl", fold=-1)
    if title is not None:
        # Not all backends support suptitle; guard to avoid failing the whole run
        try:
            fig.suptitle(title, fontsize=14)
        except Exception:
            pass

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")


def save_circuit_png_with_text_fallback(
    circuit: QuantumCircuit,
    output_path: str,
    title: str,
    dpi: int = 200,
    text_fold: int = 120,
) -> None:
    """
    Save a PNG circuit diagram if possible; otherwise print a text representation.

    This is useful for robust execution in environments where optional
    visualization dependencies may not be installed.
    """
    try:
        save_circuit_png(circuit=circuit, output_path=output_path, title=title, dpi=dpi)
        print(f"[OK] Saved {output_path}")
    except Exception as exc:
        print(f"[WARN] PNG rendering failed for '{output_path}': {exc}")
        print(f"\n--- {title} (text fallback) ---")
        print(circuit.draw(output="text", fold=text_fold))
