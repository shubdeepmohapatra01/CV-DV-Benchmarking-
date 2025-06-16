import os,sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PARENT_DIR)

import c2qa
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import DensityMatrix
from collections import Counter
from benchmarks_circuit import (cat_state_circuit, gkp_state_circuit, state_transfer_CVtoDV, state_transfer_DVtoCV, JCH_simulation_circuit_display
                                ,binary_knapsack_vqe_circuit, shors_circuit, qft_circuit)
from features import collect_cvcircuit_metrics, wigner_negativity, truncation_cost_approximate, average_energy

def characterize_circuit(name, circuit, cutoff=10):
    metrics = collect_cvcircuit_metrics(circuit)

    try:
        stateop, result, _ = c2qa.util.simulate(circuit,shots = 1024)
        print(f"Simulation for {name} done")
        state = c2qa.util.trace_out_qubits(circuit,stateop)
        metrics["Wigner Negativity"] = wigner_negativity(state, axes_min=-6, axes_max=6, axes_steps=100)
        metrics["Truncation Cost"] = truncation_cost_approximate(state, n=5)
        metrics["Avg Energy"] = average_energy(circuit, stateop, cutoff)
    except Exception as e:
        print(f"[{name}] Simulation failed: {e}")
        metrics["Wigner Negativity"] = 0
        metrics["Truncation Cost"] = 0
        metrics["Avg Energy"] = 0
        
    print(metrics)

    return metrics

def plot_radar_metrics(metrics_list, labels=None, title="CV-DV Radar Chart"):
    keys = ['Qubits', 'Qumodes', 'Qubit Gates', 'Qumode Gates',
            'Hybrid Gates', 'Total Gates', 'Wigner Negativity',
            'Truncation Cost', 'Avg Energy']
    N = len(keys)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    max_vals = {key: max(metric.get(key, 0) for metric in metrics_list) for key in keys}

    data = []
    for metric in metrics_list:
        normalized = [metric.get(key, 0) / max_vals[key] if max_vals[key] != 0 else 0 for key in keys]
        normalized += normalized[:1]
        data.append(normalized)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=16)

    colors = plt.cm.tab10.colors

    for i, d in enumerate(data):
        label = labels[i] if labels else f"Circuit {i+1}"
        color = colors[i % len(colors)]
        ax.plot(angles, d, label=label, color=color)
        ax.fill(angles, d, alpha=0.25, color=color)

        for j in range(N):
            angle = angles[j]
            r = d[j]
            value = metrics_list[i].get(keys[j], 0)
            ax.text(angle, r + 0.05, f"{value:.2f}", ha='center', va='center', fontsize=7, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.2))
    plt.tight_layout()
    plt.show()

def main():
    benchmarks = []
    labels = []

    # Example benchmark circuits
    cutoff = 2**6

    # Cat state
    from qiskit import QuantumRegister
    import c2qa
    qmr = c2qa.QumodeRegister(1,num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))), name='qumode')
    qbr = QuantumRegister(1, name='qbit')
    circuit = c2qa.CVCircuit(qmr, qbr)
    circuit = cat_state_circuit(cutoff, circuit, qbr, qmr, alpha=2)
    benchmarks.append(characterize_circuit("Cat", circuit, cutoff))
    labels.append("Cat State")

    # GKP state
    qmr = c2qa.QumodeRegister(1,num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))), name='qumode')
    qbr = QuantumRegister(1, name='qbit')
    circuit = c2qa.CVCircuit(qmr, qbr)
    circuit = gkp_state_circuit(cutoff, circuit, qbr, qmr)
    benchmarks.append(characterize_circuit("GKP", circuit, cutoff))
    labels.append("GKP State")

    # JCH simulation (vary Nsites)
    # cutoff = 2**6
    # for Nsites in [2, 4]:
    #     circuit = JCH_simulation_circuit_display(Nsites, Nqubits=Nsites, cutoff=cutoff, J=0.1,
    #                                              omega_r=1.0, omega_q=1.0, g=0.5, tau=0.1, timesteps=1)
    #     label = f"JCH N={Nsites}"
    #     benchmarks.append(characterize_circuit(label, circuit, cutoff))
    #     labels.append(label)

    # # VQE (vary depth)
    # from scipy.sparse import identity
    # H = identity(2**2)  # Dummy Hamiltonian
    # for depth in [1, 3]:
    #     circuit = binary_knapsack_vqe_circuit(H, ndepth=depth, nfocks=[cutoff, cutoff])
    #     label = f"VQE depth={depth}"
    #     benchmarks.append(characterize_circuit(label, circuit, cutoff))
    #     labels.append(label)

    # Shor’s algorithm
    # circuit = shors_circuit(N=15, m=3, R=3, a=2, delta=0.1, cutoff=cutoff)
    # benchmarks.append(characterize_circuit("Shor", circuit, cutoff))
    # labels.append("Shor")

    # QFT
    # circuit = qft_circuit(cutoff=cutoff, n=2, ancilla=1, delta=0.1)
    # benchmarks.append(characterize_circuit("QFT", circuit, cutoff))
    # labels.append("QFT")

    # Plot everything
    plot_radar_metrics(benchmarks, labels, title="CV-DV Quantum Benchmark Comparison")

if __name__ == "__main__":
    main()