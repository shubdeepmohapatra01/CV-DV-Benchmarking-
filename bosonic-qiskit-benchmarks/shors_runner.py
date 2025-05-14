import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fractions import Fraction
from math import gcd
from qutip import *

from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PARENT_DIR)

from custom_gates import shors, state_generation
import c2qa

# === SETTINGS ===
RUN_ON_SERVER = False  # Set to False for local debug
RESULTS_DIR = "results_logs"
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file = os.path.join(RESULTS_DIR, f"log_{timestamp}.txt")
factors_file = os.path.join(RESULTS_DIR, f"factors_{timestamp}.txt")

# Redirect print to log file
def write_log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")

def write_factors(msg):
    with open(factors_file, "a") as f:
        f.write(msg + "\n")

# === UTILITIES ===

def try_find_factors(N, r, a):
    if r % 2 != 0:
        return None
    candidate = pow(a, r//2, N)
    if candidate == N-1 or candidate == 1:
        return None
    p = np.gcd(candidate-1, N)
    q = np.gcd(candidate+1, N)
    if p*q == N and p != 1 and q != 1:
        return (p, q)
    return None

def find_valid_a_values(N):
    valid_a = []
    for a in range(2, N):
        if gcd(a, N) != 1:
            continue
        r = 1
        while pow(a, r, N) != 1 and r < N:
            r += 1
        if r % 2 == 0 and pow(a, r // 2, N) != N - 1:
            valid_a.append((a, r))
    return valid_a

def sample_p_and_estimate_period(p_dist, paxis, max_denominator=100):
    # prob_dist = np.nan_to_num(prob_dist, nan=0.0, posinf=0.0, neginf=0.0)
    prob_dist = p_dist / np.sum(p_dist)
    p_sample = np.random.choice(paxis, p=prob_dist)
    s_over_r = p_sample / (2 * np.pi)
    frac = Fraction(s_over_r).limit_denominator(max_denominator)
    j, r = frac.numerator, frac.denominator
    return (r if gcd(j, r) == 1 else None), (j, r), p_sample

def generate_gkp_codeword(cutoff, delta=0.3, kappa=1.0, logical=0, num_peaks=7):
    a = destroy(cutoff)
    sq = squeeze(cutoff, -np.log(delta))

    state = 0
    spacing = np.sqrt(np.pi)
    for k in range(-num_peaks//2, num_peaks//2 + 1):
        shift = (2 * k + logical) * spacing
        envelope = np.exp(-0.5 * (k * kappa)**2)
        disp = displace(cutoff, shift)
        peak = disp * sq * basis(cutoff, 0)
        state += envelope * peak

    state = state.unit()
    return state

# === MAIN FUNCTION ===

def estimate_success_probability(N, m, R, delta, cutoff, trials=30, shots=1024):
    qmr1 = c2qa.QumodeRegister(num_qumodes=3, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))), name='qumode')
    qbr = QuantumRegister(1)
    cr = ClassicalRegister(1)

    valid_a_r_pairs = find_valid_a_values(N)
    if not valid_a_r_pairs:
        write_log(f"[N={N}] No valid a values found.")
        return 0.0, [], 0, 0

    total_successes = 0
    total_shots = 0
    all_factors = set()
    alpha = np.sqrt(np.pi)
    gkp0 = generate_gkp_codeword(cutoff, logical=0)
    R_gate = shors.translation_R(cutoff, R)
    M_gate = shors.multiplication(cutoff, N)
    UaNm_cache = {}

    for trial in range(trials):
        a, true_r = random.choice(valid_a_r_pairs)
        write_log(f"[Trial {trial+1}] a = {a}, expected r = {true_r}")

        circuit = c2qa.CVCircuit(qmr1, qbr, cr)

        # circuit.cv_initialize(gkp0.full(), qmr1[0])
        # circuit.cv_initialize(gkp0.full(), qmr1[1])
        # circuit.cv_sq(-np.log(delta), qmr1[2])
        
        for i in range(3):
            circuit.cv_sq(-np.log(delta), qmr1[i])

        for k in range(1, 9):
            circuit.cv_c_d(alpha, qmr1[0], qbr[0])
            Uxk = state_generation.Ux_operator(cutoff, alpha, 4 * k, delta)
            Ux_gate = UnitaryGate(Uxk.full(), label=f'Ux_{k}')
            circuit.append(Ux_gate, qmr1[0] + qbr[:])

        for k in range(1, 9):
            circuit.cv_c_d(alpha, qmr1[1], qbr[0])
            Uxk = state_generation.Ux_operator(cutoff, alpha, 4 * k, delta)
            Ux_gate = UnitaryGate(Uxk.full(), label=f'Ux_{k}')
            circuit.append(Ux_gate, qmr1[1] + qbr[:])

        # Cache reuse
        if a in UaNm_cache:
            circuit = UaNm_cache[a]
        else:
            circuit.append(UnitaryGate(R_gate.full(), label='R'), qmr1[0])
            circuit.append(UnitaryGate(M_gate.full(), label=f'M{N}'), qmr1[1])
            circuit = shors.U_aNm(cutoff, circuit, qmr1, qbr, a, N, m)
            UaNm_cache[a] = circuit.copy()

        # Run once to get momentum distribution
        stateop, _, _ = c2qa.util.simulate(circuit, shots=1)
        rho_qumode_0 = shors.get_reduced_qumode_density_matrix(stateop, qumode_index=0, num_qumodes=3, cutoff=cutoff)
        x_dist, xaxis = shors.momentum_plotting(rho_qumode_0, cutoff, ax_min=-30, ax_max=30, steps=200)

        for _ in range(shots):
            estimated_r, (j, r), p_sample = sample_p_and_estimate_period(x_dist.flatten(), xaxis)

            if estimated_r is not None:
                factors = try_find_factors(N, estimated_r, a)
                if factors:
                    total_successes += 1
                    all_factors.update(factors)
                    write_log(f"Shot success: r={estimated_r}, factors={factors}")
                    write_factors(f"N={N}, a={a}, r={estimated_r}, factors={factors}")
                else:
                    write_log(f"Shot fail: r={estimated_r} gave no valid factors.")
            else:
                write_log("Shot fail: Could not extract valid r.")
            total_shots += 1

    success_rate = total_successes / total_shots if total_shots > 0 else 0.0
    write_log(f"[N={N}, cutoff={cutoff}] Total success probability: {success_rate:.4f}")
    return success_rate, sorted(all_factors), total_successes, total_shots



# === DRIVER ===

def run():
    settings = [
        (128, [15])
    ] if not RUN_ON_SERVER else [
        (128, [15, 21]),
        (256, [15, 21, 33, 35]),
        (512, [15, 21, 33, 35, 51]),
    ]

    Ns = []
    probabilities = []
    labels = []

    for cutoff, numbers in settings:
        for N in numbers:
            write_log(f"\n=== Running for N={N}, cutoff={cutoff} ===")
            success_rate, factors, total_successes, total_shots = estimate_success_probability(
                N=N, m=2, R=15, delta=0.222, cutoff=cutoff, trials=5, shots=1024
            )
            Ns.append(N)
            probabilities.append(success_rate)
            label = ', '.join(map(str, factors)) if factors else "None"
            labels.append(label)

    # Plotting all in one bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(n) for n in Ns], probabilities, color='skyblue')
    plt.ylabel("Success Probability (across all trials)")
    plt.xlabel("Composite Number N")
    plt.title("Period Finding Success Probability Across Trials")

    # Annotate bars with factors
    for bar, label in zip(bars, labels):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, label,
                 ha='center', va='bottom', fontsize=8, rotation=45)

    plot_path = os.path.join(RESULTS_DIR, f"summary_prob_plot3.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()
    write_log(f"Summary plot saved to {plot_path}")


if __name__ == "__main__":
    run()
