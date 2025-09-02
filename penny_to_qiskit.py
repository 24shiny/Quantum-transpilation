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

class Penny_to_qiskit:
    """convert a qnode in PennyLane to a qiskit object"""
    def __init__(self, qnode):
        self.qnode = qnode
        self.gate_info = None # list of dictionaries liek {'name': 'PauliX', 'wires': [0], 'params': []}
        self.num_q = None
        self.qc = None # qiskit object
        self.ref_state = None      
        self.transpiled_circuit = None
        self.transpiled_qiskit_0 = None
        self.transpiled_qiskit_1 = None
        self.transpiled_qiskit_2 = None
        self.transpiled_qiskit_3 = None
        self.transpiled_penny = None
        self.df = None # comparison table

        self.extract_info_from_qnode()
        
    def extract_info_from_qnode(self):
        """Extracts gate info from a QNode by tracing its quantum function."""
        quantum_fn = self.qnode.func
    
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
        
        self.gate_info = gate_info
        
    def convert_to_qiskit(self,dict_elem): # pennylane to qiskit using the dictionary to map
        """Quantum circuits from KetGPT consist of the following eight gates. Add more lines for others, if necessary"""
        name = dict_elem['name']
        wires = dict_elem['wires']
        params = dict_elem['params']
    
        gate_map = {
            'Hadamard': lambda : self.qc.h(wires[0]),
            'PauliX': lambda: self.qc.x(wires[0]),
            'CNOT': lambda: self.qc.cx(wires[0], wires[1]),
            'CZ': lambda: self.qc.cz(wires[0], wires[1]),
            'RY': lambda: self.qc.ry(params[0], wires[0]),
            'QubitUnitary': lambda: self.qc.append(UnitaryGate(params[0]), wires),
            'U1': lambda: self.qc.p(params[0], wires[0]),
            'U2': lambda: self.qc.u(pi/2, params[0], params[1], wires[0])
        }
    
        if name in gate_map:
            gate_map[name]()
        else:
            raise ValueError(f"Unsupported gate: {name}")

    def conversion_loop(self):
        self.num_q = max(wire for gate in self.gate_info for wire in gate['wires'])
        self.qc = QuantumCircuit(self.num_q)
        for dict_elem in self.gate_info:
            self.convert_to_qiskit(dict_elem)
        self.get_ref_state()

    def get_ref_state(self):
        simulator = AerSimulator()
        self.qc.save_statevector()
        result = simulator.run(self.qc).result()
        statevector = result.data()['statevector']
        self.ref_state = statevector.data
            
    def summary_qiskit(self):
        counts = {"1-qubit": 0, "2-qubit": 0}
        for inst in self.qc.data:
            if isinstance(inst.operation, Gate):  
                num_qubits = len(inst.qubits)
                if num_qubits == 1:
                    counts["1-qubit"] += 1
                elif num_qubits == 2:
                    counts["2-qubit"] += 1
        return [self.qc.num_qubits, self.qc.size(), counts['1-qubit'], counts['2-qubit'], self.qc.depth()]
    
    def qiskit_transpiler(self, qc_inst, level=2):
        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(qc_inst, backend, optimization_level=level)
        result = backend.run(transpiled_qiskit).result()
        transpiled_state = result.get_statevector()
        self.transpiled_circuit = transpiled_circuit

    def summary_qiskit_fidelity(self, transpiled_circuit):
        backend = Aer.get_backend('qasm_simulator')
        result = backend.run(transpiled_circuit).result()
        qiskit_state = result.get_statevector()
        return int(fidelity_qiskit(self.ref_state, qiskit_state.data))
        
    def make_table_prep(self): # takes qiskit circuits
        self.df = pd.DataFrame(columns=['original','qiskit_0','qiskit_1','qiskit_2','qiskit_3','penny'])
        
        self.df['original'] = self.summary_qiskit(self.qc)
        qc0 = copy.deepcopy(self.qc)
        qc1 = copy.deepcopy(self.qc)
        qc2 = copy.deepcopy(self.qc)
        qc3 = copy.deepcopy(self.qc)
    
        self.transpiled_qiskit_0 = self.qiskit_transpiler(qc0,0)
        self.df['qiskit_0'] = list(map(int,self.summary_qiskit(self.transpiled_qiskit_0)))
        self.transpiled_qiskit_1 = self.qiskit_transpiler(qc1,1)
        self.df['qiskit_1'] = list(map(int,self.summary_qiskit(self.transpiled_qiskit_1)))
        self.transpiled_qiskit_2 = self.qiskit_transpiler(qc2,2)
        self.df['qiskit_2'] = list(map(int,self.summary_qiskit(self.transpiled_qiskit_2)))
        self.transpiled_qiskit_3 = self.qiskit_transpiler(qc3,3)
        self.df['qiskit_3'] = list(map(int,self.summary_qiskit(self.transpiled_qiskit_3)))
        
        penny = self.qnode
        self.transpiled_penny  = compile(penny)
        self.df['penny'] = self.summary_penny(self.transpiled_penny)
    
        self.df.index = ['qubits', 'gates', '1q gates', '2q gates', 'depth']

    def make_table(df): # add a row for fidelity
        self.make_table_prep()
        # New row as a dictionary
        new_row = {'original':'-', 
                   'qiskit_0':self.summary_qiskit_fidelity(self.transpiled_qiskit_0), 
                   'qiskit_1':self.summary_qiskit_fidelity(self.transpiled_qiskit_1), 
                   'qiskit_2':self.summary_qiskit_fidelity(self.transpiled_qiskit_2), 
                   'qiskit_3':self.summary_qiskit_fidelity(self.transpiled_qiskit_3), 
                   'penny':'-'}
        self.df = pd.concat([self.df, pd.DataFrame([new_row], index=['fidelity'])], ignore_index=False)

    def main(self):
        self.conversion_loop()
        self.make_table()
        