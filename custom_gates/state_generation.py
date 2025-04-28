import c2qa
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RGate
from qiskit.converters import circuit_to_gate
from qutip import *
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate

def CD_real(cutoff,alpha):
    p = momentum(cutoff)
    cdReal = tensor(sigmax(),-1j*np.sqrt(2)*alpha*p).expm()
    return cdReal

def CD_imaginary(cutoff,alpha):
    x = position(cutoff)
    cdImaginary = tensor(sigmay(),-1j*np.pi*x/(4*alpha)).expm()
    return cdImaginary