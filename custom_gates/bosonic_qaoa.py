import c2qa
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RGate
from qiskit.converters import circuit_to_gate
from qutip import *
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from scipy.stats.contingency import margins
from scipy.optimize import minimize
from qiskit.quantum_info import state_fidelity, Statevector

def cost(cutoff, a,n, eta):
    x = position(cutoff)  # Position operator
    aI = a * qeye(cutoff)  
    Hc = (x - aI) ** n   # Cost Hamiltonian
    costH = (-1j * eta * Hc).expm()  # Unitary evolution
    return costH

def kinetic_mixer(cutoff, gamma):
    p = momentum(cutoff)  # Momentum operator
    Hm = 0.5*(p ** 2)  # Mixer Hamiltonian
    mixerH = (-1j * gamma * Hm).expm()  # Unitary evolution
    return mixerH

def cvQAOA(params,cutoff,depth,s,n,a,costval,estval):
    gamma_list = params[:depth]  
    eta_list = params[depth:]    

    # Define Qumode and Classical Registers
    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))))
    cr = ClassicalRegister(1)

    # Initialize the circuit
    circuit = c2qa.CVCircuit(qmr, cr)
    circuit.cv_initialize(0, qmr[0])
    circuit.cv_sq(-s, qmr[0])  

    for i in range(depth):
        gamma = gamma_list[i]
        eta = eta_list[i]

        costH = cost(cutoff, a,n, eta)
        gate1 = UnitaryGate(costH.full(), label=f'Uc_{eta}')
        circuit.append(gate1, qmr[0])
        
        mixH = kinetic_mixer(cutoff, gamma)
        gate1 = UnitaryGate(mixH.full(), label=f'Um_{gamma}')
        circuit.append(gate1, qmr[0])

    # Simulate the circuit
    x = position(cutoff)
    # st0 = Statevector.from_instruction(circuit) 
    state, _, _ = c2qa.util.simulate(circuit)

    expval = (expect(x, Qobj(state)))

    # Compute the cost function value
    cost_val = (expval - a) ** n 

    # Append to costval for tracking
    estval.append(expval)
    costval.append(cost_val)
    return (cost_val)

def results_final(params, cutoff, depth, s, n, a):
    from c2qa.wigner import wigner
    from c2qa.util import simulate
    from qutip import Qobj, expect

    gamma_list = params[:depth] 
    eta_list = params[depth:]  

    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))))
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr, cr)

    circuit.cv_initialize(0, qmr[0])       # vacuum
    circuit.cv_sq(-s, qmr[0])              # squeezing gate

    # Simulate initial state
    init_state, _, _ = simulate(circuit)

    # QAOA unitaries
    for i in range(depth):
        costH = cost(cutoff, a, n, eta_list[i])
        cost_gate = UnitaryGate(costH.full(), label=f'Uc_{eta_list[i]}')
        circuit.append(cost_gate, qmr[0])

        mixH = kinetic_mixer(cutoff, gamma_list[i])
        mixer_gate = UnitaryGate(mixH.full(), label=f'Um_{gamma_list[i]}')
        circuit.append(mixer_gate, qmr[0])

    # Final simulation
    final_state, _, _ = simulate(circuit)
    x_op = position(cutoff)
    expval = expect(x_op, Qobj(final_state))

    # Position basis axis
    ax_min, ax_max, steps = -6, 6, 200
    xaxis = np.linspace(ax_min, ax_max, steps)

    # Initial state distribution
    w_init = wigner(init_state, axes_max=ax_max, axes_min=ax_min, axes_steps=steps)
    x_init, _ = margins(w_init.T)
    x_init *= (ax_max - ax_min) / steps

    # Final state distribution
    w_final = wigner(final_state, axes_max=ax_max, axes_min=ax_min, axes_steps=steps)
    x_final, _ = margins(w_final.T)
    x_final *= (ax_max - ax_min) / steps

    return expval, x_init, x_final, xaxis
