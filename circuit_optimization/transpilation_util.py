# basics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
# penny
import pennylane as qml
from pennylane.transforms import compile
from pennylane.math import fidelity_statevector as fidelity_penny
# qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit_aer import Aer, AerSimulator
from qiskit.visualization import plot_circuit_layout, circuit_drawer
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import state_fidelity as fidelity_qiskit
from math import pi
import copy

def extract_info_from_qnode(qnode):
    """Extracts gate info from a QNode by tracing its quantum function."""
    quantum_fn = qnode.func

    with qml.tape.QuantumTape() as tape:
        quantum_fn()

    gate_info = []
    for op in tape.operations:
        safe_params = []
        for p in op.parameters:
            try:
                safe_params.append(float(p))  
            except (TypeError, ValueError):
                safe_params.append(np.array(p).tolist()) 

        gate_info.append({
            "name": op.name,
            "wires": list(op.wires),
            "params": safe_params
        })
    
    return gate_info

def summary_penny(circuit):
    obj = qml.specs(circuit)()['resources']
    summary =  [obj.num_wires, obj.num_gates, obj.gate_sizes[1], obj.gate_sizes[2], obj.depth]
    df = pd.DataFrame(summary, index=['num_qubit', 'num_gate', 'num_1q_gate', 'num_2q_gate', 'depth'])
    df = df.T
    return df

def summary_qiskit(qc):
    counts = {"1-qubit": 0, "2-qubit": 0}
    for inst in qc.data:
        if isinstance(inst.operation, Gate):  
            num_qubits = len(inst.qubits)
            if num_qubits == 1:
                counts["1-qubit"] += 1
            elif num_qubits == 2:
                counts["2-qubit"] += 1
    summary = [qc.num_qubits, qc.size(), counts['1-qubit'], counts['2-qubit'], qc.depth()]
    df = pd.DataFrame(summary, index=['num_qubit', 'num_gate', 'num_1q_gate', 'num_2q_gate', 'depth'])
    df = df.T
    return df