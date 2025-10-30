import pennylane as qml
from pennylane.transforms import compile
import pandas as pd
import copy
import numpy as np
from qiskit_aer import Aer, AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qco_level_0 import extract_info_from_qnode
import matplotlib.pyplot as plt

def to_qiskit(qc, dict_elem):
    name = dict_elem['name']
    wire = dict_elem['wires']
    param = dict_elem['params']
    if name == 'Hadamard':
        qc.h(wire[0])
    if name == 'PauliX':
        qc.x(wire[0])
    if name == 'PauliY':
        qc.y(wire[0])
    if name == 'PauliZ':
        qc.z(wire[0])
    if name == 'CNOT':
        qc.cx(wire[0],wire[1])
    if name == 'CX':
        qc.cx(wire[0],wire[1])  
    if name == 'CY':
        qc.cy(wire[0],wire[1])  
    if name == 'CZ':
        qc.cz(wire[0],wire[1])    
    if name == 'QubitUnitary':
        qc.append(UnitaryGate(param[0]),wire)
    if name == 'U2':
        qc.u(np.pi/2, param[0], param[1], wire[0])

def to_qc(circuit):
    sample_q_num = qml.specs(circuit)()['resources'].num_wires
    qc = QuantumCircuit(sample_q_num)
    for dict_elem in extract_info_from_qnode(circuit):
        to_qiskit(qc, dict_elem)
    # Use AerSimulator to extend the circuit
    simulator = AerSimulator()
    qc.save_statevector()  # Now this works!

    # Run the simulation
    result = simulator.run(qc).result()
    statevector = result.data()['statevector']
    ref_state = statevector.data
    return qc

def summary_penny(circuit):
    obj = qml.specs(circuit)()['resources']
    temp = qml.specs(circuit)()['resources'].gate_types # dict
    summary =  [obj.num_wires, obj.num_gates, obj.gate_sizes[1], temp['CZ']+temp['CNOT'], temp['QubitUnitary'], obj.depth]
    df = pd.DataFrame(summary, index=['num_qubit', 'num_gate', 'num_1q_gate', 'num_2q_gate', 'unitary','depth'])
    return df.iloc[:,0].tolist()

def summary_qiskit(qc):
    counts = {'1q': 0, '2q': 0, 'U':0}
    for gate in qc.data:
        if len(gate.qubits) == 1:
            counts['1q'] += 1
        elif len(gate.qubits) == 2 and gate.name != 'unitary':
            counts['2q'] += 1
        elif len(gate.qubits) == 2 and gate.name == 'unitary':
            counts['U'] += 1
    return [qc.num_qubits, qc.size(), counts['1q'], counts['2q'], counts['U'], qc.depth()]

def qiskit_transpiler(qc, level=0):
    backend = Aer.get_backend('qasm_simulator')
    transpiled_qiskit = transpile(qc, backend, optimization_level=level)
    result = backend.run(transpiled_qiskit).result()
    statevector = result.get_statevector(qc)
    return  transpiled_qiskit, statevector

# main
def make_spec_table(circuit, optimized_circuit): # takes qiskit circuits
    qc = to_qc(circuit)

    df = pd.DataFrame(columns=['original','qiskit_0','qiskit_1','penny'])
    
    df['original'] = summary_qiskit(qc)
    qc0 = copy.deepcopy(qc)
    transpiled_qiskit_0, statevector_0 = qiskit_transpiler(qc0,0)
    df['qiskit_0'] = list(map(int,summary_qiskit(transpiled_qiskit_0)))

    qc1 = copy.deepcopy(qc)
    transpiled_qiskit_1, statevector_1 = qiskit_transpiler(qc1,1)
    df['qiskit_1'] = list(map(int,summary_qiskit(transpiled_qiskit_1)))

    
    df['penny'] = summary_penny(compile(circuit))
    df['mine'] = summary_penny(optimized_circuit)

    df.index = ['qubits', 'gates', '1q gates', '2q gates', 'unitary','depth']
    print(df)

def show_circuit(circuit):
    qml.draw_mpl(circuit, style='pennylane')()
    plt.show()