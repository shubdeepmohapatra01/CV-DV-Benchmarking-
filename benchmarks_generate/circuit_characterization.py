import os, sys
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister
from qiskit.quantum_info import DensityMatrix
from qutip import *

from benchmarks_circuit import (
    cat_state_circuit, gkp_state_circuit, JCH_simulation_circuit_display,
    binary_knapsack_vqe, binary_knapsack_vqe_circuit, shors_circuit
)
from features import (
    collect_cvcircuit_metrics, evaluate_quantum_metrics, 
    wigner_negativity_all_modes, truncation_cost_all_modes, average_energy_all
)

from custom_gates import bosonic_vqe

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "circuit_characters")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRUCTURAL_KEYS = ['Qubits', 'Qumodes', 'Qubit Gates', 'Qumode Gates', 'Total Gates']
PERFORMANCE_KEYS = ['Hybrid Gates', 'Truncation Cost', 'Wigner Negativity', 'Average Energy']


def plot_radar_group(metrics, keys, label, filename, color_idx=0):
    N = len(keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    raw_vals = [metrics.get(k, 0) for k in keys]
    raw_vals += raw_vals[:1]

    max_vals = max(raw_vals)
    max_radius = max_vals * 1.1 if max_vals != 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_ylim(0, max_radius)
    ax.set_yticklabels([])
    ax.set_title(label, fontsize=14)

    color = plt.cm.tab10(color_idx)
    ax.plot(angles, raw_vals, label=label, color=color)
    ax.fill(angles, raw_vals, alpha=0.25, color=color)

    for j in range(N):
        angle = angles[j]
        r = raw_vals[j]
        ax.text(angle, r + 0.05 * max_radius, f"{r:.2f}",
                ha='center', va='center', fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[{label}] Radar chart saved to {filename}")


def average_over_timesteps(circuit_template, U1, qmr, qbr, cutoff, steps, dt,num_qumodes,num_qubits, sample_every=5):
    import c2qa
    trunc_costs, wigner_negs, energies = [], [], []
    circuit = circuit_template.copy()

    for i, t in enumerate(np.arange(0, steps * dt, dt)):
        circuit.append(U1, qmr[:] + qbr[:])
        if i % sample_every == 0:
            state, _, _ = c2qa.util.simulate(circuit)
            trunc, wneg, energy = evaluate_quantum_metrics(circuit, state, cutoff,num_qumodes,num_qubits)
            trunc_costs.append(trunc)
            wigner_negs.append(wneg)
            energies.append(energy)

    return {
        "Truncation Cost": np.mean(trunc_costs),
        "Wigner Negativity": np.mean(wigner_negs),
        "Average Energy": np.mean(energies)
    }


def characterize_circuit(name, circuit, cutoff,num_qubits=1,num_qumodes=1, stateop=None):
    metrics = collect_cvcircuit_metrics(circuit, cutoff)

    if stateop is not None:
        trunc, wneg, energy = evaluate_quantum_metrics(circuit, stateop, cutoff,num_qumodes,num_qubits)
        metrics.update({
            "Truncation Cost": trunc,
            "Wigner Negativity": wneg,
            "Average Energy": energy
        })

    return metrics


def main():
    import c2qa
    from scipy.sparse import identity

    cutoff = 2**6
    color_idx = 0

    # --- Cat state ---
    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))),name='qumode')
    qbr = QuantumRegister(1)
    circuit = c2qa.CVCircuit(qmr, qbr)
    circuit = cat_state_circuit(cutoff, circuit, qbr, qmr, alpha=4)
    state, _, _ = c2qa.util.simulate(circuit)
    metrics = characterize_circuit("Cat State", circuit, cutoff,stateop = state)
    plot_radar_group(metrics, STRUCTURAL_KEYS, "Cat State (Structure)", os.path.join(OUTPUT_DIR, "cat_struct.png"), color_idx)
    plot_radar_group(metrics, PERFORMANCE_KEYS, "Cat State (Quantum)", os.path.join(OUTPUT_DIR, "cat_quantum.png"), color_idx)
    color_idx += 1

    # --- GKP state ---
    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))),name='qumode')
    qbr = QuantumRegister(1)
    circuit = c2qa.CVCircuit(qmr, qbr)
    circuit = gkp_state_circuit(cutoff, circuit, qbr, qmr)
    state, _, _ = c2qa.util.simulate(circuit)
    metrics = characterize_circuit("GKP State", circuit, cutoff,stateop = state)
    plot_radar_group(metrics, STRUCTURAL_KEYS, "GKP State (Structure)", os.path.join(OUTPUT_DIR, "gkp_struct.png"), color_idx)
    plot_radar_group(metrics, PERFORMANCE_KEYS, "GKP State (Quantum)", os.path.join(OUTPUT_DIR, "gkp_quantum.png"), color_idx)
    color_idx += 1

    # --- JCH Simulation ---
    cutoff = 2**2
    Nsites = 3
    qmr = c2qa.QumodeRegister(Nsites, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))),name='qumode')
    qbr = QuantumRegister(Nsites)
    circuit_template = c2qa.CVCircuit(qmr, qbr)
    circuit_template.cv_initialize(2, qmr[0])
    U1 = JCH_simulation_circuit_display(Nsites, Nqubits=Nsites, cutoff=cutoff, J=0.1,
                                        omega_r=2 * np.pi * 2, omega_q=2 * np.pi * 3,
                                        g=2 * np.pi * 0.5, tau=0.1, timesteps=1)
    jch_metrics = average_over_timesteps(circuit_template, U1, qmr, qbr, cutoff, steps=50, dt=0.1,num_qumodes=Nsites,num_qubits=Nsites)
    circuit = JCH_simulation_circuit_display(Nsites, Nqubits=Nsites, cutoff=cutoff, J=0.1,
                                        omega_r=2 * np.pi * 2, omega_q=2 * np.pi * 3,
                                        g=2 * np.pi * 0.5, tau=0.1, timesteps=1)
    struct_metrics = collect_cvcircuit_metrics(circuit, cutoff)
    struct_metrics.update(jch_metrics)
    plot_radar_group(struct_metrics, STRUCTURAL_KEYS, f"JCH N={Nsites} (Structure)", os.path.join(OUTPUT_DIR, "jch_struct.png"), color_idx)
    plot_radar_group(struct_metrics, PERFORMANCE_KEYS, f"JCH N={Nsites} (Quantum)", os.path.join(OUTPUT_DIR, "jch_quantum.png"), color_idx)
    color_idx += 1

    # --- VQE Benchmark ---
    values = [2, 5, 7, 3]
    weights = [2.5, 3, 4, 3.5]
    max_weight = 7
    l_val = 3
    nfocks = [8,8]
    ndepth = 5

    bkp_fun1, bkp_list1 = bosonic_vqe.binary_knapsack_ham(l_val, values, weights, max_weight)
    bkp_list1 = bosonic_vqe.binary_to_pauli_list(bkp_fun1, bkp_list1)
    bkp_ham1 = Qobj( bosonic_vqe.qubit_op_to_ham(bkp_list1).full() )
    en, Xvec, int_results = binary_knapsack_vqe(bkp_ham1, ndepth, nfocks, maxiter=150,
                                    verb=1, threshold=1e-3)
    vqe_circuit = binary_knapsack_vqe_circuit(bkp_ham1,ndepth,nfocks,Xvec)
    state, _, _ = c2qa.util.simulate(vqe_circuit)
    vqe_metrics = characterize_circuit(f"VQE depth={ndepth}", vqe_circuit, cutoff, state)
    plot_radar_group(vqe_metrics, STRUCTURAL_KEYS, "VQE (Structure)", os.path.join(OUTPUT_DIR, "vqe_struct.png"), color_idx)
    plot_radar_group(vqe_metrics, PERFORMANCE_KEYS, "VQE (Quantum)", os.path.join(OUTPUT_DIR, "vqe_quantum.png"), color_idx)
    

if __name__ == "__main__":
    main()