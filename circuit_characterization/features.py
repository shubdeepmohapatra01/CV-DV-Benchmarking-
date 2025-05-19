import numpy as np
from math import pi, ceil
import scipy
from qutip import *
import c2qa
from collections import Counter
import matplotlib.pyplot as plt

def collect_cvcircuit_metrics(circuit,cutoff):
    qubit_to_reg = {}
    for reg in circuit.qregs:
        for q in reg:
            qubit_to_reg[q] = reg.name

    qubit_regs = []
    qumode_regs = []

    for reg in circuit.qregs:
        name = reg.name.lower()
        if any(tag in name for tag in ['qmode', 'cv', 'osc','qumode']):
            qumode_regs.append(reg.name)
        else:
            qubit_regs.append(reg.name)

    num_qubits = sum(len(reg) for reg in circuit.qregs if reg.name in qubit_regs)
    num_qumodes = sum(len(reg) for reg in circuit.qregs if reg.name in qumode_regs)/ (np.ceil(np.log2(cutoff)))

    gate_counts = Counter()

    skip_instrs = {'barrier', 'measure', 'initialize', 'snapshot', 'delay'}

    for instr, qargs, cargs in circuit.data:
        if instr.name in skip_instrs:
            continue

        involved_regs = {qubit_to_reg.get(q, "").lower() for q in qargs}
        has_qubit = any(reg in qubit_regs for reg in involved_regs)
        has_qumode = any(reg in qumode_regs for reg in involved_regs)

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
        "Qumodes": int(num_qumodes),
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

    for i, d in enumerate(data):
        label = labels[i] if labels else f"Circuit {i+1}"
        ax.plot(angles, d, label=label)
        ax.fill(angles, d, alpha=0.25)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()