import pennylane as qml
from pennylane.transforms import *
import pandas as pd
import copy
import numpy as np
from qiskit_aer import Aer, AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qco_level_0 import extract_info_from_qnode, wire_range
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation, Collect2qBlocks, UnitarySynthesis, RemoveDiagonalGatesBeforeMeasure, OptimizeSwapBeforeMeasure, TemplateOptimization
import matplotlib.pyplot as plt

def to_qiskit(qc, dict_elem):
    name = dict_elem['name']
    wire = dict_elem['wires']
    param = dict_elem['params']
    if name == 'Hadamard':
        qc.h(wire[0])
    if name == 'PauliX':
        qc.x(wire[0])
    if name == 'CNOT':
        qc.cx(wire[0],wire[1])
    if name == 'CZ':
        qc.cz(wire[0],wire[1])    
    if name == 'QubitUnitary':
        qc.append(UnitaryGate(param[0]),wire)
    if name == 'RY':
        qc.ry(param[0], wire[0])
    if name == 'U1':
        qc.p(param[0], wire[0])
    if name == 'U2':
        qc.u(np.pi/2, param[0], param[1], wire[0])
    
def to_qc(circuit):
    circuit_info = extract_info_from_qnode(circuit)
    sample_q_num = wire_range(circuit_info)[1] + 1  # qml.specs(circuit)()['resources'].num_wires
    qc = QuantumCircuit(sample_q_num)
    for dict_elem in circuit_info:
        to_qiskit(qc, dict_elem)
    # Use AerSimulator to extend the circuit
    # simulator = AerSimulator()
    # qc.save_statevector()  # Now this works!
    # # Run the simulation
    # result = simulator.run(qc).result()
    # statevector = result.data()['statevector']
    # ref_state = statevector.data
    return qc

def summary_penny(circuit):
    obj = qml.specs(circuit)()['resources']
    cz = obj.gate_types['CZ']
    cnot = obj.gate_types['CNOT']
    swap = obj.gate_types['SWAP']
    num_type = len(qml.specs(circuit)()['resources'].gate_types.keys())
    summary =  [obj.num_gates, obj.gate_sizes[1], obj.gate_sizes[2]-(cz+cnot+swap), obj.depth, num_type]
    df = pd.DataFrame(summary, index=['num_gate', 'num_1q_gate', 'num_2q_gate','depth','num_type'])
    return df.iloc[:,0].tolist()

def summary_qiskit(qc):
    counts = {'1q': 0, '2q': 0}
    for gate in qc.data:
        if len(gate.qubits) == 1:
            counts['1q'] += 1
        elif len(gate.qubits) == 2 and gate.name != 'unitary':
            counts['2q'] += 1
    return [qc.size(), counts['1q'], counts['2q'], counts['U'], qc.depth(), len(qc.count_oFps().keys())]

def qiskit_transpiler(qc):
    # backend = Aer.get_backend('qasm_simulator')
    pm = PassManager()
    pm.append([Optimize1qGates(), TemplateOptimization(), Collect2qBlocks()])
    # pm.append([Optimize1qGates(), Collect2qBlocks(), 
    #            RemoveDiagonalGatesBeforeMeasure(), OptimizeSwapBeforeMeasure(), TemplateOptimization()]) # CommutativeCancellation()
    transpiled_qiskit = pm.run(qc)
    return  transpiled_qiskit

def qiskit_level_3(qc):
    return transpile(qc, optimization_level=3, basis_gates=['rx','ry','rz','cx'])

# main
def make_spec_table(circuit, optimized_circuit): # takes qiskit circuits
    df = pd.DataFrame(columns=['original','qiskit','penny','mine'])

    qc = to_qc(circuit)
    df['original'] = summary_qiskit(qc)

    qc = copy.deepcopy(qc)
    transpiled_qiskit = qiskit_transpiler(qc)
    df['qiskit'] = list(map(int,summary_qiskit(transpiled_qiskit)))

    # pipeline = [cancel_inverses, merge_rotations, single_qubit_fusion, combine_global_phases] # commute_controlled,
    pipeline = [cancel_inverses, merge_rotations, single_qubit_fusion]
    transpiled_penny = compile(circuit, pipeline=pipeline)
    df['penny'] = summary_penny(transpiled_penny)

    df['mine'] = summary_penny(optimized_circuit)

    df.index = ['qubits', 'gates', '1q gates', '2q gates', 'unitary','depth']
    return df

def show_circuit(circuit):
    qml.draw_mpl(circuit, style='pennylane')()
    plt.show()