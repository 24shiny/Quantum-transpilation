# Quantum-transpilation
## 1. PennyLane to Qiskit Converter (comparison/Penny_to_qiskit.py)
<div align="justify"> 
  While PennyLane supports converting circuits from Qiskit to its own format, it does not currently offer a method for converting PennyLane circuits back to Qiskit. So I have made one myself!😊
  Designed to transform KetGPT circuits—which consist of only eight predefined gates—this class cannot handle arbitrary gates by default. However, you can easily tailor it by adding Qiskit–PennyLane gate correspondences in the convert_to_qiskit() function. In addition, you can print the comparison table, compares the performance of Qiskit's and PennyLane's transpilers.
</div>

## 2. Quantum Transpiler Performance Comparison (comparison/compare_decomposition.ipynb)
<div align="justify"> 
  <p align='center'><img src="img/comparison_table.png" width="500"/></p>
  <p align='center'>Fig. 1. Performance of Qiskit's and PennyLane's transpilers</p>
  This code prints a table comparing the performance of different quantum transpilers. Specifically, Qiskit's transpiler across optimization levels 0 to 3, and PennyLane's default compilation   strategy. Note that the number of unitary gates drops to zero at optimization levels 2 and 3 due to unitary synthesis, which decomposes unitary gates into 1- or 2-qubit primitive gates. Additionally, PennyLane's default compilation applies only few techniques such as commutation cancellation, inverse cancellation, rotation merging, and barrier removal and therefore does not show outstanding performance as it is. However, PennyLane offers <a href='https://docs.pennylane.ai/en/stable/introduction/compiling_circuits.html'> a number of optimization functions</a>, with which users can build up their own pipelines. Lastly, the fidelity for PennyLane's compiler cannot be calculated due to the decrease in the number of qubits; if we figure out which qubit is discarded, we can compute it, though.
</div>

## 3. Graph representation of quantum circuits (circuit_optimization/to_graph_networkx.ipynb)
<div align="justify"> 
  <p align='center'><img src="img/original_circuits.png" width="600"/></p>
  <p align='center'>Fig. 1. Quantum circuit generate by KetGPT</p>
  <p align='center'><img src="img/graph_representation_of_circuits.png" width="500"/></p>
  <p align='center'>Fig. 2.Graph representation of quantum circuits</p>
  Quantum circuits are often transformed into graphs in quantum circuit optimization. I am currently developing an interface between PennyLane and NetworkX !👍
</div>
