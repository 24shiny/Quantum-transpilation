# Quantum-transpilation

## 1. PennyLane to Qiskit Converter (comparison/Penny_to_qiskit.py)
<div align="justify"> 
  While PennyLane supports converting circuits from Qiskit to its own format, it does not currently offer a method for converting PennyLane circuits back to Qiskit. So I have made one myself!😊
  Designed to transform KetGPT circuits—which consist of only eight predefined gates—this class cannot handle arbitrary gates by default. However, you can easily tailor it by adding Qiskit–PennyLane gate correspondences in the convert_to_qiskit() function. In addition, you can print the comparison table, compares the performance of Qiskit's and PennyLane's transpilers.
</div><br>

## 2. Quantum Transpiler Performance Comparison (comparison/compare_decomposition.ipynb)
<div align="justify"> 
  <p align='center'><img src="img/comparison_table.png" width="500"/></p>
  <p align='center'>Fig. 1. Performance of Qiskit's and PennyLane's transpilers</p>
  This code prints a table comparing the performance of different quantum transpilers. Specifically, Qiskit's transpiler across optimization levels 0 to 3, and PennyLane's default compilation   strategy. Note that the number of unitary gates drops to zero at optimization levels 2 and 3 due to unitary synthesis, which decomposes unitary gates into 1- or 2-qubit primitive gates. Additionally, PennyLane's default compilation applies only few techniques such as commutation cancellation, inverse cancellation, rotation merging, and barrier removal and therefore does not show outstanding performance as it is. However, PennyLane offers <a href='https://docs.pennylane.ai/en/stable/introduction/compiling_circuits.html'> a number of optimization functions</a>, with which users can build up their own pipelines. Lastly, the fidelity for PennyLane's compiler cannot be calculated due to the decrease in the number of qubits; if we figure out which qubit is discarded, we can compute it, though.
</div><br>

## 3. Graph representation of quantum circuits (circuit_optimization/graph_representation/penny_to_graph.py)
<div align="justify"> 
  <p align='center'><img src="img/original_circuits.png" width="600"/></p>
  <p align='center'>Fig. 2. Quantum circuit generate by KetGPT</p>
  <p align='center'><img src="img/graph_representation_of_circuits.png" width="500"/></p>
  <p align='center'>Fig. 3.Graph representation of the quantum circuit</p>
  Quantum circuit optimization aims to reduce the number of gates and the depth, while maintaing the fidelity. Once quantum circuits are represented as graphs, we can apply graph algorithms to them!
  As a major example, Qiskit's transpiler converts quantum circuits to graphs, perform the Dijkstra search over them to replace known gate patterns with more efficient ones. Here I present a module to bridge PennyLane and NetworkX, which converts a quantum circuit in PennyLane to a graph object in NetworkX.
</div><br>

## 4. Hardware-independent quantum circuit optimization on graphs (circuit_optimization/hw_independent/)
<div align="justify"> 
  Most conventional approaches rely on pattern matching, which typically involves the following steps: (1) template preparation, (2) pattern search across quantum circuits, (3) subcircuit replacement, and (4) iteration of steps (2) and (3) until no further matches are     found. Successful optimization hinges on complete templates and efficient search strategies. However, manually crafting substitution rules is time-consuming, and incomplete rule sets can lead to performance degradation. Here, I present a hardware-independent, template-free strategy for quantum circuit optimization. The algorithm leverages the often-overlooked insight that substitution rules are implicitly applied through a series of matrix multiplications. Gates are first grouped by the greedy modularity community detection algorithm, and then consolidated into a single unitary gate. This approach enables effective circuit cleanup as a preprocessing step, streamlining messy designs before further optimization.<br><br>
</div>

### (1) 1-qubit gate optimization
<div align="justify">
  Single-qubit gates can be effectively groupped by setting multi-qubit gates as barriers and appyling the community detection algorithm. In Fig.4 each community is colored differently. Afterwards, the effective unitary each community results in is calculated and transfored to basis single-qubit gate. During this process, gate cancellation between single-qubit gates are implicitly done!
</div><br>
  <p align='center'><img src="img/1q_opt.png" width="500"/></p>
  <p align='center'>Fig. 4. single-qubit gate optimization</p>
  <p align='center'><img src="img/after_1q_opt.png" width="500"/></p>
  <p align='center'>Fig. 5. After single-qubit gate optimization</p>

### (2) 2-qubit gate optimization
<div align="justify">
  Rather than applying the previous approach for single-qubit gates, I have prepared a different approach for 2-qubit gates: after other gates are removed, each connected component is examined. Once the pair of nodes which share two edges is detected, these are removed from the original gate. Note that this graph algorithm hands the list of gates to remove over to the original circuit!
</div><br>
  <p align='center'><img src="img/2q_opt.png" width="500"/></p>
  <p align='center'>Fig. 6.two-qubit gate optimization</p>
  <p align='center'><img src="img/after_1q_opt.png" width="500"/></p>
  <p align='center'>Fig. 7. After two-qubit gate optimization</p>

### (3) single and double- qubit gate joint optimization

### (4) Results
<p align='center'><img src="img/penny.png" width="500"/></p>
<p align='center'>Fig. 8. Quantum circuit preprocessed by PennyLane's compiler</p>
<p align='center'><img src="img/my.png" width="500"/></p>
<p align='center'>Fig. 7. Quantum circuit preprocessed by my strategy developed so far. The same as Penny's!</p>
  
## Further optimization methods will be coming! 😉
