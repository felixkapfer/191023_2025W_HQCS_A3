import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

mat1 = [1,2,1,0,1,9,2,2]   # 12101922  -> Rx/X layer
mat2 = [1,2,3,1,7,2,4,0]   # 12317240  -> Ry/Y layer
mat3 = [1,2,4,2,9,6,6,9]   # 12429669  -> Rz/Z layer

# 8-qubit device (statevector simulation)
dev_state = qml.device("default.qubit", wires=8)

# Device for snapshots: 1 shot (one sample) per snapshot
dev_shot = qml.device("default.qubit", wires=8, shots=1)


#2.2

def prob_layer(digs, rot_axis, fallback_pauli, rng):
    
    rot_axis = rot_axis.upper()
    fallback_pauli = fallback_pauli.upper()

    if rot_axis not in {"X", "Y", "Z"}:
        raise ValueError("rot_axis must be one of {'X','Y','Z'}")
    if fallback_pauli not in {"X", "Y", "Z"}:
        raise ValueError("fallback_pauli must be one of {'X','Y','Z'}")

    for j, ij in enumerate(digs):
        p_j = ij / 10.0
        theta_j = 2 * np.pi * (ij / 10.0)

        if rng.random() < p_j:
            if rot_axis == "X":
                qml.RX(theta_j, wires=j)
            elif rot_axis == "Y":
                qml.RY(theta_j, wires=j)
            else:  # "Z"
                qml.RZ(theta_j, wires=j)
        else:
            if fallback_pauli == "X":
                qml.PauliX(wires=j)
            elif fallback_pauli == "Y":
                qml.PauliY(wires=j)
            else:  # "Z"
                qml.PauliZ(wires=j)

def cz_pairwise_12_34_56_78():
    """Apply Controlled Z Gate so only |11> gets a minus sign on pairs  (0,1), (2,3), (4,5), (6,7)."""
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        qml.CZ(wires=[a, b])

def cz_pairwise_23_45_67_81():
    """Apply CZ on pairs (1,2), (3,4), (5,6), (7,0)."""
    for a, b in [(1, 2), (3, 4), (5, 6), (7, 0)]:
        qml.CZ(wires=[a, b])
        
@qml.qnode(dev_state)
def gift_state(seed_circuit: int = 0):
    """
    Implements steps (a)-(g) of the assignment for ONE probabilistic circuit copy.
    The randomness (RX vs X etc.) is controlled by seed_circuit.
    Returns the statevector of this one circuit instance. Containing the information of all 3 Mat numbers each one affects another axis
    """
    rng = np.random.default_rng(seed_circuit)

    # (a) H^{⊗8}
    for w in range(8):
        qml.Hadamard(wires=w)

    # (b) first matric number: RX(theta_j) w.p. p_j, else X
    prob_layer(mat1, rot_axis="X", fallback_pauli="X", rng=rng)

    # (c) CZ pairwise (1,2), (3,4), (5,6), (7,8)
    cz_pairwise_12_34_56_78()

    # (d) second matric number: RY(theta_j) w.p. p_j, else Y
    prob_layer(mat2, rot_axis="Y", fallback_pauli="Y", rng=rng)

    # (e) CZ pairwise starting at qubit 2: (2,3), (4,5), (6,7), (8,1)
    cz_pairwise_23_45_67_81()

    # (f) third matric number: RZ(theta_j) w.p. p_j, else Z
    prob_layer(mat3, rot_axis="Z", fallback_pauli="Z", rng=rng)

    # (g) H^{⊗8}
    for w in range(8):
        qml.Hadamard(wires=w)

    return qml.state()

def _rotate_to_measure_in_basis(basis_id, wire):
    """
    We always measure PauliZ at the end.
    To effectively measure:
      X: apply H, then measure Z
      Y: apply S† then H, then measure Z
      Z: do nothing, measure Z
    basis_id: 0->X, 1->Y, 2->Z
    """
    if basis_id == 0:          # X basis
        qml.Hadamard(wires=wire)
    elif basis_id == 1:        # Y basis
        qml.adjoint(qml.S)(wires=wire)   # S†
        qml.Hadamard(wires=wire)
    elif basis_id == 2:        # Z basis
        pass
    else:
        raise ValueError("basis_id must be 0 (X), 1 (Y), or 2 (Z)")


OBS = {
    "X1":   {0: "X"},
    "Y1":   {0: "Y"},
    "Z1":   {0: "Z"},
    "X1X2": {0: "X", 1: "X"},
    "Z1X2": {0: "Z", 1: "X"},
    "Y1Z2": {0: "Y", 1: "Z"},
}

@qml.qnode(dev_shot)
def _shadow_snapshot_qnode(seed_circuit: int, bases_id):
    """
    Executes:
      - gift circuit (a)-(g) with randomness controlled by seed_circuit
      - basis rotations given by bases_id (length 8, entries 0/1/2)
      - measures PauliZ on all qubits (shots=1)

    Returns: array of 8 outcomes in {+1, -1}
    """
    rng = np.random.default_rng(seed_circuit)

    # (a) H^{⊗8}
    for w in range(8):
        qml.Hadamard(wires=w)

    # (b) Rx or X
    prob_layer(mat1, rot_axis="X", fallback_pauli="X", rng=rng)

    # (c) CZ pairwise
    cz_pairwise_12_34_56_78()

    # (d) Ry or Y
    prob_layer(mat2, rot_axis="Y", fallback_pauli="Y", rng=rng)

    # (e) CZ shifted pairs
    cz_pairwise_23_45_67_81()

    # (f) Rz or Z
    prob_layer(mat3, rot_axis="Z", fallback_pauli="Z", rng=rng)

    # (g) final H^{⊗8}
    for w in range(8):
        qml.Hadamard(wires=w)

    # Apply measurement-basis rotations
    for w in range(8):
        _rotate_to_measure_in_basis(int(bases_id[w]), wire=w)

    # Measure Z on all wires (one shot -> one ±1 result per wire)
    return [qml.sample(qml.PauliZ(wires=w)) for w in range(8)]


def shadow_snapshot(seed_circuit: int, seed_meas: int):
    """
    User-facing helper:
      - draws random bases (X/Y/Z) using seed_meas
      - runs the QNode once using seed_circuit
      - returns (bases_as_strings, outcomes_as_ints)
    """
    rng_m = np.random.default_rng(seed_meas)

    # 0->X, 1->Y, 2->Z
    bases_id = rng_m.integers(low=0, high=3, size=8)

    # Run the circuit + measurement
    outcomes = _shadow_snapshot_qnode(seed_circuit, bases_id)

    # Convert bases to readable labels
    mapping = {0: "X", 1: "Y", 2: "Z"}
    bases = [mapping[int(x)] for x in bases_id]

    # outcomes is shape (8,) with values ±1
    outcomes = [int(x) for x in outcomes]

    # bases: welche Basis wurde pro Qubit gewählt (X/Y/Z) outcomes: was kam raus (±1)
    return bases, outcomes

def single_snapshot_estimator(obs_map, bases, outcomes):
    """
        A snapshot only contains information about O if we happened to measure
        all involved qubits in the correct bases (e.g., for X1X2 we need X on qubit0 and X on qubit1).
        If any required basis does NOT match, return 0 (this snapshot is useless for O).
        If all bases match, multiply the corresponding ±1 outcomes (prod).
        Multiply by 3^w (w = number of involved qubits) to compensate that a matching basis occurs
        only with probability (1/3)^w, so the estimator is unbiased when averaged over many snapshots.
    """
    prod = 1
    w = 0
    for wire, pauli in obs_map.items():
        w += 1
        if bases[wire] != pauli:
            return 0.0
        prod *= outcomes[wire]
    return (3 ** w) * prod

def estimate_all_observables(N=2000, seed=0):
    """
        estimate_all_observables:
        Repeats the classical-shadow experiment N times.
        For each iteration:
        draw a random circuit realization (probabilistic gates)
        draw random measurement bases (X/Y/Z per qubit)
        perform one shadow snapshot (bases + ±1 outcomes)
        compute a single-snapshot estimator for each observable
        and add it to a running sum

        Finally, divide by N to obtain the estimated expectation values.
        Averaging many noisy single-snapshot estimators yields convergence
        to the true expectation values of the observables.
    """
    
    rng = np.random.default_rng(seed)

    sums = {name: 0.0 for name in OBS.keys()}

    for t in range(N):
        seed_circuit = int(rng.integers(0, 2**31 - 1))
        seed_meas    = int(rng.integers(0, 2**31 - 1))

        bases, outcomes = shadow_snapshot(seed_circuit=seed_circuit, seed_meas=seed_meas)

        for name, obs_map in OBS.items():
            sums[name] += single_snapshot_estimator(obs_map, bases, outcomes)

    # average
    estimates = {name: sums[name] / N for name in sums}
    return estimates

est = estimate_all_observables(N=5000, seed=42)

for k,v in est.items():
    print(k, "=", v)
    # i measure the correlation btw qubit 1 and 2 with respect that 1 and 2 are entagled with the 3-8 qubits „Wie verhalten sich Qubit 1 und 2 innerhalb des gesamten Systems?“

# 2.4

def convergence_in_blocks(N_max=5000, block=100, seed=42):
    rng = np.random.default_rng(seed)

    # cumulative sums for each observable
    sums = {name: 0.0 for name in OBS.keys()}

    Ns = []
    traj = {name: [] for name in OBS.keys()}

    for t in range(1, N_max + 1):
        seed_circuit = int(rng.integers(0, 2**31 - 1))
        seed_meas    = int(rng.integers(0, 2**31 - 1)) # X,Y, or Z Observable chosen for a qubit

        bases, outcomes = shadow_snapshot(seed_circuit=seed_circuit, seed_meas=seed_meas)

        for name, obs_map in OBS.items():
            sums[name] += single_snapshot_estimator(obs_map, bases, outcomes)

        # every 'block' snapshots, record cumulative estimate; only mod 100
        if t % block == 0:
            Ns.append(t)
            for name in OBS.keys():
                traj[name].append(sums[name] / t) # mean value 

    return np.array(Ns), traj

# run and plot
Ns, traj = convergence_in_blocks(N_max=5000, block=100, seed=42)

plt.figure()
for name in OBS.keys():
    plt.plot(Ns, traj[name], label=name)
plt.axhline(0.0)
plt.xlabel("Number of snapshots N")
plt.ylabel("Shadow estimate  <O>_shadow(N)")
plt.legend()
plt.show()


# 2.5 

# We need the real Operators for Pennylane
OBS_OPS = {
    "X1":   qml.PauliX(0),
    "Y1":   qml.PauliY(0),
    "Z1":   qml.PauliZ(0),
    "X1X2": qml.PauliX(0) @ qml.PauliX(1),
    "Z1X2": qml.PauliZ(0) @ qml.PauliX(1),
    "Y1Z2": qml.PauliY(0) @ qml.PauliZ(1),
}

@qml.qnode(dev_state)
def exact_expvals(seed_circuit: int):
    rng = np.random.default_rng(seed_circuit)

    for w in range(8):
        qml.Hadamard(wires=w)
    prob_layer(mat1, rot_axis="X", fallback_pauli="X", rng=rng)
    cz_pairwise_12_34_56_78()
    prob_layer(mat2, rot_axis="Y", fallback_pauli="Y", rng=rng)
    cz_pairwise_23_45_67_81()
    prob_layer(mat3, rot_axis="Z", fallback_pauli="Z", rng=rng)
    for w in range(8):
        qml.Hadamard(wires=w)

    return [qml.expval(OBS_OPS[name]) for name in OBS_OPS.keys()]

def error_in_blocks(N_max=5000, block=100, seed=42):
    rng = np.random.default_rng(seed)

    shadow_sums = {name: 0.0 for name in OBS.keys()}
    exact_sums  = {name: 0.0 for name in OBS_OPS.keys()}

    Ns = []
    err_traj = {name: [] for name in OBS.keys()}

    for t in range(1, N_max + 1):
        seed_circuit = int(rng.integers(0, 2**31 - 1))
        seed_meas    = int(rng.integers(0, 2**31 - 1))

        # shadow update
        bases, outcomes = shadow_snapshot(seed_circuit=seed_circuit, seed_meas=seed_meas)
        for name, obs_map in OBS.items():
            shadow_sums[name] += single_snapshot_estimator(obs_map, bases, outcomes)

        # exact update (same circuit realization!)
        exact_vals = exact_expvals(seed_circuit)
        for name, val in zip(OBS_OPS.keys(), exact_vals):
            exact_sums[name] += float(val)

        if t % block == 0:
            Ns.append(t)
            for name in OBS.keys():
                shadow_avg = shadow_sums[name] / t
                exact_avg  = exact_sums[name]  / t
                err_traj[name].append(abs(exact_avg - shadow_avg))

    return np.array(Ns), err_traj

Ns_e, err = error_in_blocks(N_max=5000, block=100, seed=42)

plt.figure()
for name in OBS.keys():
    plt.plot(Ns_e, err[name], label=name)
plt.xlabel("Number of snapshots N")
plt.ylabel(r"$|\langle O\rangle_{\mathrm{exact}}-\langle O\rangle_{\mathrm{shadow}}|$")
plt.legend()
plt.show()
