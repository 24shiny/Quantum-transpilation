# basics
import numpy as np
import pandas as pd
# penny
import pennylane as qml
from pennylane.transforms import compile
from pennylane.math import fidelity_statevector as fidelity_penny
# qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import state_fidelity as fidelity_qiskit
from math import pi

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

def map_to_qiskit(qc, dict_elem): # pennylane to qiskit using the dictionary to map
    name = dict_elem['name']
    wires = dict_elem['wires']
    params = dict_elem['params']

    gate_map = {
        'Hadamard': lambda : qc.h(wires[0]),
        'PauliX': lambda: qc.x(wires[0]),
        'CNOT': lambda: qc.cx(wires[0], wires[1]),
        'CZ': lambda: qc.cz(wires[0], wires[1]),
        'RY': lambda: qc.ry(params[0], wires[0]),
        'QubitUnitary': lambda: qc.append(UnitaryGate(params[0]), wires),
        'U1': lambda: qc.p(params[0], wires[0]),
        'U2': lambda: qc.u(pi/2, params[0], params[1], wires[0])
    }

    if name in gate_map:
        gate_map[name]()
    else:
        raise ValueError(f"Unsupported gate: {name}")
    
def to_qiskit(circuit): # qnote to qc
    circuit_info = extract_info_from_qnode(circuit)
    num_q = max(wire for op in circuit_info for wire in op['wires'])
    qc = QuantumCircuit(num_q)
    for dict_elem in circuit_info:
        map_to_qiskit(qc, dict_elem)
    return qc
    
def summary_penny(circuit):
    obj = qml.specs(circuit)()['resources']
    temp = qml.specs(circuit)()['resources'].gate_types # dict
    summary =  [obj.num_wires, obj.num_gates, obj.gate_sizes[1], temp['CZ']+temp['CNOT'], temp['QubitUnitary'], obj.depth]
    df = pd.DataFrame(summary, index=['num_qubit', 'num_gate', 'num_1q_gate', 'num_2q_gate', 'unitary','depth'])
    # df = df.T
    print(df)

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

def info_to_qnode(circuit_info):
    dev = qml.device('default.qubit')
    def circuit():
        for gate in circuit_info:
            name = gate['name']
            wires = gate['wires']
            params = gate['params']
            if name == 'Hadamard':
                qml.Hadamard(wires=wires[0])
            elif name == 'X':
                qml.PauliX(wires=wires[0])
            elif name == 'U2':
                qml.U2(params[0], params[1], wires=wires[0])
            elif name == 'CNOT':
                qml.CNOT(wires=wires)            
            elif name == 'CZ':
                qml.CZ(wires=wires)
            elif name == 'QubitUnitary':
                matrix = np.array(params[0])
                qml.QubitUnitary(matrix, wires=wires)
            else:
                raise ValueError(f"Unsupported gate: {name}")
        return qml.state()
    qnode = qml.QNode(circuit, dev)
    return qnode
