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

def qproj00():
    return basis(2, 0).proj()


def qproj11():
    return basis(2, 1).proj()


def qproj01():
    op = np.array([[0, 1], [0, 0]])
    return Qobj(op)


def qproj10():
    op = np.array([[0, 0], [1, 0]])
    return Qobj(op)

def hadamard():
    op = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return Qobj(op)

def Q_displacement_plus1(cutoff):
    return (-1j*momentum(cutoff)).expm()
def Q_displacement_minus1(cutoff):
    return (1j*momentum(cutoff)).expm()
def bosonic_sum(cutoff1,cutoff2):
    return (-1j*tensor(position(cutoff1),momentum(cutoff2))).expm()
def single_mode_squeeze(cutoff,squeeze_param):
    return (1j*squeeze_param*(position(cutoff)*momentum(cutoff) + momentum(cutoff)*position(cutoff))/2).expm()
def Q_control_plus1(cutoff):
    return tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(-1j*momentum(cutoff)).expm())
def Q_control_minus1(cutoff):
    return tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(1j*momentum(cutoff)).expm())
def P_displacement_pi(cutoff,sign):
    return tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(1j*np.pi*sign*position(cutoff)).expm())
def rotation_control(cutoff,sign):
    return tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(1j*np.pi/2*sign*num(cutoff)).expm())

def multiplication(cutoff,alpha):
    M_alpha = single_mode_squeeze(cutoff,-np.log(alpha))
    return M_alpha

def extractLSB(cutoff):
    qumode_gate = tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(1j*np.pi*position(cutoff)).expm())
    qubit_gate = tensor(hadamard(),qeye(cutoff))
    return qubit_gate * qumode_gate * qubit_gate

def translation_R(cutoff,R):
    return (-1j * R * momentum(cutoff)).expm()

def control_multiplication(cutoff,alpha):
    rotation_plus = rotation_control(cutoff,1)
    # control_rotation_plus = tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),rotation_plus)
    M_sqrt_alpha_dag = tensor(qeye(2),multiplication(cutoff,np.sqrt(alpha)).dag())
    rotation_minus = rotation_control(cutoff,-1)
    # control_rotation_minus = tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),rotation_minus)
    M_sqrt_alpha = tensor(qeye(2),multiplication(cutoff,np.sqrt(alpha)))
    
    return rotation_plus * M_sqrt_alpha_dag * rotation_minus * M_sqrt_alpha

def V_alpha(cutoff,circuit,qumode_register,qubit_register,alpha):
    M_2 = multiplication(cutoff,2)
    gate1 =UnitaryGate(M_2.full(), label='M2')
    circuit.append(gate1, qumode_register[2])
    
    LSB_extract = extractLSB(cutoff)
    gate1 =UnitaryGate(LSB_extract.full(), label='LSB1')
    circuit.append(gate1, qumode_register[0]+qubit_register[:])
    
    control_subtraction = Q_control_minus1(cutoff)
    gate1 =UnitaryGate(control_subtraction.full(), label='Q-1')
    circuit.append(gate1, qumode_register[0]+qubit_register[:])
    
    Malpha = control_multiplication(cutoff,alpha)
    gate1 =UnitaryGate(Malpha.full(), label=f'M_{alpha}')
    circuit.append(gate1, qumode_register[1]+qubit_register[:])
    
    control_addition = Q_control_plus1(cutoff)
    gate1 =UnitaryGate(control_addition.full(), label='Q+1')
    circuit.append(gate1, qumode_register[2]+qubit_register[:])
    
    LSB_extract = extractLSB(cutoff)
    gate1 =UnitaryGate(LSB_extract.full(), label='LSB2')
    circuit.append(gate1, qumode_register[2]+qubit_register[:])
    
    M_half = multiplication(cutoff,2)
    gate1 =UnitaryGate(M_half.full(), label='M1/2')
    circuit.append(gate1, qumode_register[0])
    
    return circuit

def V_alpha_dag(cutoff,circuit,qumode_register,qubit_register,alpha):
    M_half = multiplication(cutoff, 2)
    gate1 = UnitaryGate(M_half.full(), label='M1/2')
    circuit.append(gate1, qumode_register[0])

    LSB_extract = extractLSB(cutoff)
    gate1 = UnitaryGate(LSB_extract.full(), label='LSB2')
    circuit.append(gate1, qumode_register[2] + qubit_register[:])

    control_addition = Q_control_plus1(cutoff)
    gate1 = UnitaryGate(control_addition.full(), label='Q+1')
    circuit.append(gate1, qumode_register[2] + qubit_register[:])

    Malpha = control_multiplication(cutoff, alpha)
    gate1 = UnitaryGate(Malpha.full(), label=f'M_{alpha}')
    circuit.append(gate1, qumode_register[1] + qubit_register[:])

    control_subtraction = Q_control_minus1(cutoff)
    gate1 = UnitaryGate(control_subtraction.full(), label='Q-1')
    circuit.append(gate1, qumode_register[0] + qubit_register[:])

    LSB_extract = extractLSB(cutoff)
    gate1 = UnitaryGate(LSB_extract.full(), label='LSB1')
    circuit.append(gate1, qumode_register[0] + qubit_register[:])

    M_2 = multiplication(cutoff, 2)
    gate1 = UnitaryGate(M_2.full(), label='M2')
    circuit.append(gate1, qumode_register[2])
    
    return circuit

def V_aNm(cutoff,circuit,qumode_register,qubit_register,a,N,m):
    for i in range(m):
        alpha = pow(a,2**i) % N
        circuit = V_alpha(cutoff,circuit,qumode_register,qubit_register,alpha)
        circuit.barrier()
        
    for _ in range(m):
        circuit = V_alpha_dag(cutoff,circuit,qumode_register,qubit_register,1)
        circuit.barrier()
        
    return circuit

def V_aNm_dagger(cutoff, circuit, qumode_register, qubit_register, a, N, m):
    for _ in range(m):
        V_alpha(cutoff, circuit, qumode_register, qubit_register, 1)
        circuit.barrier()

    for i in reversed(range(m)):
        alpha = pow(a, 2**i, N)
        V_alpha_dag(cutoff, circuit, qumode_register, qubit_register, alpha)
        circuit.barrier()
    
    return circuit

def U_aNm(cutoff, circuit, qumode_register, qubit_register, a, N, m):
    circuit = V_aNm_dagger(cutoff, circuit, qumode_register, qubit_register, a, N, m)
    circuit.barrier()
    circuit.barrier()
    
    Q_addition = Q_displacement_plus1(cutoff)
    gate1 = UnitaryGate(Q_addition.full(), label='Q+1')
    circuit.append(gate1, qumode_register[1])
    circuit.barrier()
    circuit.barrier()
    
    circuit = V_aNm(cutoff, circuit, qumode_register, qubit_register, a, N, m)
    circuit.barrier()
    circuit.barrier()
    
    return circuit
     

def position_plotting(state,cutoff):
    x = position(cutoff)
    expval = expect(x, Qobj(state))

    ax_min, ax_max, steps = -6, 6, 200
    w = c2qa.wigner.wigner(state, axes_max=ax_max, axes_min=ax_min, axes_steps=steps)
    x_dist, _ = margins(w.T)  # Marginalize over y-axis

    x_dist *= (ax_max - ax_min) / steps
    xaxis = np.linspace(ax_min, ax_max, steps)
    return x_dist,xaxis

def trace_out_qumode_index(circuit,state,qumode_register,qubit_register,qumode_index='0'):
    if(qumode_index == '0'):
        trace = c2qa.util.cv_partial_trace(circuit, state, qubit_register[0])
        trace = c2qa.util.cv_partial_trace(circuit, trace, qumode_register[1]+qumode_register[2])
    elif(qumode_index == '1'):
        trace = c2qa.util.cv_partial_trace(circuit, state, qubit_register[0])
        trace = c2qa.util.cv_partial_trace(circuit, trace, qumode_register[0]+qumode_register[2])
    else:
        trace = c2qa.util.cv_partial_trace(circuit, state, qubit_register[0])
        trace = c2qa.util.cv_partial_trace(circuit, trace, qumode_register[0]+qumode_register[1])
        
    return trace