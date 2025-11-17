import numpy as np

def is_identity(gate):
    return np.allclose(gate, np.eye(len(gate)))

def is_unitary(matrix, tol=1e-10):
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        return False
    identity = np.eye(matrix.shape[0])
    return np.allclose(matrix.conj().T @ matrix, identity, atol=tol)

two_q_gates = {
    'CNOT': np.array([
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  0,  1,  0]
    ], dtype=complex),
    
    'CNOTinv': np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  0,  1,  0],
        [ 0,  1,  0,  0]
    ], dtype=complex),
    
    'CZ': np.array([
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0, -1]
    ], dtype=complex),

    'SWAP': np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1]
    ], dtype=complex),

    # CNOT
    'CNOT_Hadamard': 0.71 * np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, -1], [0, 1, -1, 0]], dtype=complex),
    'CNOT_PauliX': np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=complex),
    'Hadamard_CNOT': 0.71 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [1, 0, -1, 0]], dtype=complex), 
    'PauliX_CNOT': np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=complex),
    'Hadamard_CNOT_PauliX': 0.71 * np.array([[0, 1, 0, -1], [1, 0, -1, 0], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=complex),
    'PauliX_CNOT_Hadamard': 0.71 * np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, -1, 1, 0], [-1, 0, 0, 1]], dtype=complex),
    'Hadamard_CNOT_Hadamard': 0.5* np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=complex),
    'PauliX_CNOT_PauliX': np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex),
    
    # CNOTinv
    'CNOTinv_Hadamard': 0.71 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0], [0, -1, 0, 1]], dtype=complex),
    'CNOTinv_PauliX': np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=complex),
    'Hadamard_CNOTinv': 0.71 * np.array([[1, 0, 1, 0], [0, 1, 0, -1], [1, 0, -1, 0], [0, 1, 0, 1]], dtype=complex), 
    'PauliX_CNOTinv': np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=complex),
    'Hadamard_CNOTinv_PauliX': 0.71 * np.array([[1, 0, -1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, -1]], dtype=complex),
    'PauliX_CNOTinv_Hadamard': 0.71 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, 1, 0], [0, 1, 0, -1]], dtype=complex),

    # CZ
    'CZ_Hadamard': 0.71 * np.array([[1, 0, 1, 0], [0, 1, 0, -1], [1, 0, -1, 0], [0, 1, 0, 1]], dtype=complex),
    'CZ_PauliX': np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=complex),
    'Hadamard_CZ': 0.71 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0], [0, -1, 0, 1]], dtype=complex), 
    'PauliX_CZ': np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]], dtype=complex),
    'Hadamard_CZ_PauliX': 0.71 * np.array([[1, 0, -1, 0], [0, -1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=complex),
    'PauliX_CZ_Hadamard': 0.71 * np.array([[1, 0, 1, 0], [0, -1, 0, 1], [-1, 0, 1, 0], [0, 1, 0, 1]], dtype=complex),
    'PauliX_CZ_PauliX': np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex),

    # --- Double Two-Qubit Gate Products ---
    'CZ_CNOT': np.array([
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0, -1],
        [ 0,  0,  1,  0] 
    ], dtype=complex),
    
    'CZ_CNOTinv': np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  0, -1],
        [ 0,  0,  1,  0],
        [ 0,  1,  0,  0] 
    ], dtype=complex),
    
    'CNOT_CZ': np.array([
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  0, -1,  0] 
    ], dtype=complex),
    
    'CNOTinv_CZ': np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  0,  1,  0],
        [ 0, -1,  0,  0] 
    ], dtype=complex),
    
    'CNOT_CNOTinv': np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1],
        [ 0,  1,  0,  0] 
    ], dtype=complex),
    
    'CNOTinv_CNOT': np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0] 
    ], dtype=complex),

    'CZ_SWAP': np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0, -1]
    ], dtype=complex),

    'CNOT_SWAP': np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0]
    ], dtype=complex),
}

two_q_gates_extended = {
    'CNOT': np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex),
    
    'CNOTinv': np.array(
        [[1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]]
        , dtype=complex),
    
    'CZ': np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=complex),

     'SWAP': np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex),

    'CZ_CNOT': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0, -1],
        [ 0,  0,  1,  0]]          
        , dtype=complex),
    
    'CZ_CNOTinv': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  0,  0,  -1],
        [ 0,  0,  1, 0],
        [ 0,  1,  0,  0]]          
        , dtype=complex),
    
    'CNOT_CZ': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0, 1],
        [ 0,  0,  -1,  0]]          
        , dtype=complex),
    
    'CNOTinv_CZ': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  0,  1, 0],
        [ 0,  -1,  0,  0]]          
        , dtype=complex),
    
    'CNOT_CNOTinv': np.array(
        [[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]      
        , dtype=complex),
    
    'CNOTinv_CNOT': np.array(
        [[1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0]]      
        , dtype=complex),
    
    'CZ_SWAP': np.array(
        [[ 1,  0,  0,  0],
       [ 0,  0,  1,  0],
       [ 0,  1,  0,  0],
       [ 0,  0,  0, -1]]
        , dtype=complex),

    'CNOT_SWAP': np.array(
        [[1, 0, 0, 0],
       [0, 0, 0, 1],
       [0, 1, 0, 0],
       [0, 0, 1, 0]]
        , dtype=complex),

    'CZ_CNOT_CZ': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0, -1],
        [ 0,  0,  -1,  0]]          
        , dtype=complex),
    
    'CZ_CNOTinv_CZ': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  0,  0,  -1],
        [ 0,  0,  1, 0],
        [ 0,  -1,  0,  0]]          
        , dtype=complex),
    
    'CNOT_CZ_CNOT': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  -1, 0],
        [ 0,  0,  0,  1]]          
        , dtype=complex),
    
    'CNOTinv_CZ_CNOT': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  -1,  0, 0],
        [ 0,  0,  1,  0]]          
        , dtype=complex),
    
    'CNOT_CZ_CNOTinv': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  0,  -1,  0],
        [ 0,  0,  0, 1],
        [ 0,  1,  0,  0]]          
        , dtype=complex),
    
    'CNOTinv_CZ_CNOTinv': np.array(
        [[ 1,  0,  0,  0],
        [ 0,  -1,  0,  0],
        [ 0,  0,  1, 0],
        [ 0,  0,  0,  1]]          
        , dtype=complex),
}
