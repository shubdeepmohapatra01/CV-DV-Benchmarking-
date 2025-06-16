import numpy as np
from math import pi, ceil
import scipy
from qutip import *
import c2qa
from collections import Counter
import matplotlib.pyplot as plt
from  qiskit.quantum_info import DensityMatrix

def collect_cvcircuit_metrics(circuit):
    from collections import Counter

    # Map each qubit to its register name
    qubit_to_reg = {}
    for reg in circuit.qregs:
        for q in reg:
            qubit_to_reg[q] = reg.name

    # Separate qubit and qumode registers
    qubit_regs = []
    qumode_regs = []

    for reg in circuit.qregs:
        name = reg.name.lower()
        if any(tag in name for tag in ['qmode', 'cv', 'osc','qumode','qmr']):
            qumode_regs.append(reg)
        else:
            qubit_regs.append(reg)

    # Count qubits and qumodes by number of *registers*, not physical bits
    num_qubits = sum(len(reg) for reg in qubit_regs)
    num_qumodes = sum(len(reg) for reg in qumode_regs)  # Each reg element is one qumode

    gate_counts = Counter()
    skip_instrs = {'barrier', 'measure', 'initialize', 'snapshot', 'delay'}

    for instr, qargs, cargs in circuit.data:
        if instr.name in skip_instrs:
            continue

        involved_regs = {qubit_to_reg.get(q, "").lower() for q in qargs}
        has_qubit = any(reg.name in [r.name for r in qubit_regs] for reg in qumode_regs if reg.name.lower() in involved_regs) \
                    or any(reg in [r.name for r in qubit_regs] for reg in involved_regs)
        has_qumode = any(reg.name in [r.name for r in qumode_regs] for reg in qumode_regs if reg.name.lower() in involved_regs) \
                     or any(reg in [r.name for r in qumode_regs] for reg in involved_regs)

        if has_qubit and has_qumode:
            gate_counts['hybrid_gates'] += 1
        elif has_qubit:
            gate_counts['qubit_gates'] += 1
        elif has_qumode:
            gate_counts['qumode_gates'] += 1
        else:
            gate_counts['unknown_gates'] += 1

    return {
        "Qubits": num_qubits,
        "Qumodes": num_qumodes,
        "Qubit Gates": gate_counts["qubit_gates"],
        "Qumode Gates": gate_counts["qumode_gates"],
        "Hybrid Gates": gate_counts["hybrid_gates"],
        "Total Gates": sum(gate_counts.values())
    }

    
    
def plot_radar_metrics(metrics_list, labels=None, title="CV-DV Radar Chart"):
    keys = ['Qubits', 'Qumodes', 'Qubit Gates', 'Qumode Gates', 'Hybrid Gates', 'Total Gates']
    N = len(keys)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    max_vals = {key: max(metric[key] for metric in metrics_list) for key in keys}

    data = []
    for metric in metrics_list:
        normalized = [metric[key] / max_vals[key] if max_vals[key] != 0 else 0 for key in keys]
        normalized += normalized[:1]
        data.append(normalized)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14)

    # Use a colormap for consistent and distinguishable colors
    colors = plt.cm.tab10.colors

    for i, d in enumerate(data):
        label = labels[i] if labels else f"Circuit {i+1}"
        color = colors[i % len(colors)]
        ax.plot(angles, d, label=label, color=color)
        ax.fill(angles, d, alpha=0.25, color=color)

        # Add text annotations for each metric value
        original_metrics = metrics_list[i]
        for j in range(N):
            angle = angles[j]
            r = d[j]
            value = original_metrics[keys[j]]
            ax.text(angle, r + 0.05, f"{value}", ha='center', va='center', fontsize=8, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()
    
    
def wigner_negativity(state, axes_min=-6, axes_max=6, axes_steps=200, g=np.sqrt(2), method="clenshaw"):
    """
    Compute Wigner negativity from the Wigner function of a quantum state.

    Args:
        state (array-like): State vector or density matrix.
        axes_min (int): Minimum axis value for phase space.
        axes_max (int): Maximum axis value for phase space.
        axes_steps (int): Resolution of the phase space grid.
        g (float): Scaling factor (default sqrt(2)).
        method (str): Method for Wigner function calculation.

    Returns:
        float: Wigner negativity.
    """
    xvec = np.linspace(axes_min, axes_max, axes_steps)
    W = c2qa.wigner._wigner(state, xvec, g=g, method=method)
    
    dx = dy = (axes_max - axes_min) / (axes_steps - 1)
    
    negative_volume = np.sum(np.abs(W[W < 0])) * dx * dy
    return negative_volume

def truncation_cost_approximate(state,n):
    diag_probs = np.real(np.diag(state.data))
    tail_fraction = sum(diag_probs[-n:])  # Last n levels
    
    return tail_fraction

def average_energy(circuit,stateop,cutoff,omega_qubit = 1,omega_qumode = 1):
    state_qumode = c2qa.util.trace_out_qubits(circuit,stateop)
    qumode_dm = DensityMatrix(state_qumode)
    
    state_qubit = c2qa.util.trace_out_qumodes(circuit,stateop)
    qubit_dm = DensityMatrix(state_qubit)
    
    n_op = num(cutoff).full()
    n_expect = np.trace(qumode_dm.data @ n_op).real
    
    sz = np.array([[1,0],[0,-1]])
    z_expect = np.trace(qubit_dm.data @ sz).real
    
    E_total = omega_qumode*n_expect + omega_qubit*z_expect
    
    return E_total
    
    
    