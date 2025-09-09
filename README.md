# Quantum-transpilation
## 1. PennyLane to Qiskit Converter (comparison/Penny_to_qiskit.py)
<div align="justify"> 
  While PennyLane supports converting circuits from Qiskit to its own format, it does not currently offer a method for converting PennyLane circuits back to Qiskit. So I have made one myself!😊
  Designed to transform KetGPT circuits—which consist of only eight predefined gates—this class cannot handle arbitrary gates by default. However, you can easily tailor it by adding Qiskit–PennyLane gate correspondences in the convert_to_qiskit() function. In addition, you can print the comparison table, compares the performance of Qiskit's and PennyLane's transpilers.
</div>

## 2. Quantum Transpiler comarison (comparison/compare_decomposition.ipynb)
<div align="justify"> 
  <p align='center'><img src="img/comparison_table.png" width="400"/></p>
  This code prints a table comparing the performance of different quantum transpilers. Specifically, Qiskit's transpiler across optimization levels 0 to 3, and PennyLane's default compilation   strategy. Note that the number of unitary gates drops to zero at optimization levels 2 and 3 due to unitary synthesis, which decomposes unitary gates into 1- or 2-qubit primitive gates.
  Additionally, PennyLane's default compilation applies techniques such as commutation cancellation, inverse cancellation, rotation merging, and barrier removal and therefore does not alter any unitary gates. Lastly, the fidelity for PennyLane's compiler is omitted since it cannot be calculated due to the decrease in the number of qubits.
</div>
