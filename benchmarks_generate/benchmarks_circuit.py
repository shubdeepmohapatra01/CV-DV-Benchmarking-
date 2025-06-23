import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PARENT_DIR)
import c2qa
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RGate
from qiskit.converters import circuit_to_gate
from qutip import *
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from custom_gates import state_generation,state_transfer,jch_sim,bosonic_vqe_new,shors_bq

def cat_state_circuit(cutoff,circuit,qbr,qmr,alpha):
    circuit.h(qbr[0])
    circuit.cv_c_d(alpha / np.sqrt(2),qmr[0],qbr[0])
    circuit.h(qbr[0])

    circuit.s(qbr[0])
    circuit.h(qbr[0])
    circuit.cv_c_d(1j*np.pi/(8*alpha*np.sqrt(2)),qmr[0],qbr[0])
    circuit.h(qbr[0])
    circuit.s(qbr[0])
    
    return circuit

def gkp_state_circuit(cutoff,circuit,qbr,qmr,N_rounds=9,r=0.222):
    alpha = np.sqrt(np.pi)
    circuit.cv_sq(r,qmr[0])
    for k in range(1,N_rounds):
        circuit.h(qbr[0])
        circuit.cv_c_d(alpha / np.sqrt(2),qmr[0],qbr[0])
        circuit.h(qbr[0])

        circuit.s(qbr[0])
        circuit.cv_c_d(1j*np.pi/(8*alpha*np.sqrt(2)),qmr[0],qbr[0])
        circuit.h(qbr[0])
        circuit.h(qbr[0])
        circuit.sdg(qbr[0])
        
    return circuit

def apply_basis_transformation(circuit, qbr1):
    num_qubits = len(qbr1)
    for i in range(num_qubits):
        circuit.h(qbr1[i])
        if i == num_qubits - 1:  # MSB
            circuit.x(qbr1[i])
            circuit.z(qbr1[i])
        elif i == 0:  # LSB
            circuit.z(qbr1[i])
        else:  # Middle qubits
            circuit.x(qbr1[i])
            
def apply_basis_transformation_reverse(circuit, qbr1):
    num_qubits = len(qbr1)
    for i in range(num_qubits):
        if i == num_qubits - 1:  # MSB
            circuit.z(qbr1[i])
            circuit.x(qbr1[i])
            circuit.h(qbr1[i])
        elif i == 0:  # LSB
            circuit.z(qbr1[i])
            circuit.h(qbr1[i])
        else:  # Middle qubits
            circuit.x(qbr1[i])
            circuit.h(qbr1[i])

def state_transfer_CVtoDV(cutoff,circuit,qmr,qbr,cr,n,lmbda=0.29):
    for j in range(1,n+1):
        V_j = state_transfer.Vj(lmbda,j,4,cutoff)
        gate1 = UnitaryGate(V_j.full(), label=f'V{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in p direction
        
        W_j = state_transfer.Wj(lmbda,j,4,cutoff)
        gate1 = UnitaryGate(W_j.full(), label=f'W{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in x direction
        
    apply_basis_transformation(circuit, qbr)

    # Simulate and measure
    for i in range(n):
        circuit.measure(qbr[i], cr[-(i + 1)])
        
    return circuit

def state_transfer_DVtoCV(cutoff,circuit,qmr,qbr,cr,n,lmbda=0.29):
    for j in range(n+1,0,-1):
        W_j = state_transfer.Wj(lmbda,j,4,cutoff)
        gate1 = UnitaryGate(W_j.full(), label=f'W{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in x direction 
        
        V_j = state_transfer.Vj(lmbda,j,4,cutoff)
        gate1 = UnitaryGate(V_j.full(), label=f'V{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in p direction
        
    # Simulate and measure
    for i in range(n):
        circuit.measure(qbr[i], cr[-(i + 1)])
        
    return circuit

def JCH_simulation_circuit_unitary(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau):
    U1 = jch_sim.createCircuit(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau)
    
    return U1

def JCH_simulation_circuit_display(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau,timesteps):
    circuit = jch_sim.circuit_display(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau,timesteps)
    
    return circuit

def binary_knapsack_vqe(H, ndepth, nfocks, maxiter=100, method='COBYLA', verb=0,threshold=1e-08, print_freq=10, Xvec=[]):
    en, Xvec, int_results = bosonic_vqe_new.ecd_opt_vqe(H, ndepth, nfocks, maxiter=maxiter, method='BFGS',
                                    verb=1, threshold=1e-9)
    
    return en,Xvec,int_results

def binary_knapsack_vqe_circuit(H, ndepth, nfocks,Xvec=[]):
    # Bound parameters
    beta_mag_min = 0.0
    beta_mag_max = 10.0
    beta_arg_min = 0.0
    beta_arg_max = 2 * np.pi
    theta_min = 0.0
    theta_max = np.pi
    phi_min = 0.0
    phi_max = 2 * np.pi

    # Define bounds
    size = ndepth * 2
    beta_mag_bounds = [(beta_mag_min, beta_mag_max)] * size
    beta_arg_bounds = [(beta_arg_min, beta_arg_max)] * size
    theta_bounds = [(theta_min, theta_max)] * size
    phi_bounds = [(phi_min, phi_max)] * size
    bounds = beta_mag_bounds + beta_arg_bounds + theta_bounds + phi_bounds

    # Random Initialization
    if len(Xvec) == 0:
        beta_mag = np.random.uniform(0, 3, size=(ndepth, 2))
        beta_arg = np.random.uniform(0, np.pi, size=(ndepth, 2))
        theta = np.random.uniform(0, np.pi, size=(ndepth, 2))
        phi = np.random.uniform(0, np.pi, size=(ndepth, 2))
        Xvec = bosonic_vqe_new.pack_variables(beta_mag, beta_arg, theta, phi)
        
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[0]))),name = 'qumode')
    qmr1 = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[1]))),name = 'qmr')
    qbr = QuantumRegister(1,name = 'qbit')
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr1,qmr, qbr)

    beta_mag, beta_arg, theta, phi = bosonic_vqe_new.unpack_variables(Xvec, ndepth)
    circuit = bosonic_vqe_new.ecd_rot_ansatz(beta_mag, beta_arg, theta, phi, nfocks,circuit,qmr,qmr1,qbr)
    
    return circuit

def shors_circuit(N, m, R, a, delta, cutoff):
    qmr1 = c2qa.QumodeRegister(num_qumodes=3, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))), name='qumode')
    qbr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr1, qbr, cr)
    
    circuit = gkp_state_circuit(cutoff,circuit,qbr,qmr1[0])
    circuit = gkp_state_circuit(cutoff,circuit,qbr,qmr1[1])
    circuit.cv_sq(-np.log(delta), qmr1[2])
    
    circuit = shors_bq.translation_R(cutoff,R,circuit,qmr1,0)
    circuit = shors_bq.multiplication(cutoff,N,qmr1,1)
    circuit = shors_bq.U_aNm(cutoff, circuit, qmr1, qbr, a, N, m)

    return circuit

def apply_basis_transformation_qft(circuit, combined_register):
    num_qubits = len(combined_register)
    for i in range(num_qubits):
        circuit.h(combined_register[i])
        if i == num_qubits-1:  # MSB
            circuit.x(combined_register[i])
            circuit.z(combined_register[i])
        elif i == 0:  # LSB
            circuit.z(combined_register[i])
        else:  # Middle qubits
            circuit.x(combined_register[i])
            
    for i in range(num_qubits // 2):
        circuit.swap(combined_register[i], combined_register[num_qubits - i - 1])

def apply_reverse_basis_transformation_qft(circuit, combined_register):
    num_qubits = len(combined_register)
    
    for i in range(num_qubits // 2):
        circuit.swap(combined_register[i], combined_register[num_qubits - i - 1])
    
    for i in range(num_qubits):
        if i == 0:  # LSB
            circuit.z(combined_register[i])
            circuit.h(combined_register[i])
        elif i == num_qubits-1:  # MSB
            circuit.z(combined_register[i])
            circuit.x(combined_register[i])
            circuit.h(combined_register[i])
        else:  # Middle qubits
            circuit.x(combined_register[i])
            circuit.h(combined_register[i])

def qft_circuit(cutoff,n,ancilla,delta):
    append = 2
    delta_prime = (2*np.pi)/(2**n+ancilla+append*delta)

    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))), name='qumode')
    qbr1 = QuantumRegister(n,name='qbits')
    ancilla_reg = QuantumRegister(ancilla,name = 'ancilla')
    append_reg = QuantumRegister(append, name='append')
    cr1 = ClassicalRegister(1, name='creg')
    circuit = c2qa.CVCircuit(qmr, append_reg, qbr1, ancilla_reg, cr1)

    total_register = append_reg[:]+ qbr1[:] + ancilla_reg[:]

    # Initialization

    # circuit.h(qbr1[0])
    circuit.barrier()

    # Basis transformation
    apply_basis_transformation_qft(circuit, total_register)

    circuit.barrier()

    # CV gates
    circuit = state_transfer_DVtoCV(cutoff,circuit,qmr,total_register,cr1,n+ancilla+append,delta)
    circuit.cv_d(delta / 2, qmr[0])

    circuit.cv_r(np.pi/2,qmr[0])

    circuit.cv_d(-delta_prime / 2, qmr[0])

    circuit = state_transfer_CVtoDV(cutoff,circuit,qmr,total_register,cr1,n+ancilla+append,delta)

    circuit.barrier()

    # Reverse basis transformation
    apply_reverse_basis_transformation_qft(circuit, total_register)

    circuit.barrier()

    # Measurements
    # for i, qubit in enumerate(total_register):
    #     if(i == 0 or i ==1):
    #         continue
    #     if(i>1 and i<=n+1):
    #         circuit.measure(qubit, cr1[-(i + 1)])
    # circuit.measure(ancilla_reg[1], cr1[0])
    # circuit.measure(append_reg[0], cr1[0])

    return circuit
        
        
        
        