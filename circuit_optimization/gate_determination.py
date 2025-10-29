import numpy as np

def is_matrix(gate, matrix):
    """to check whether the given gate matches with the certain matrix"""
    return np.allclose(gate, matrix)

# list of matrices
X_matrix = np.array([[0,1], [1,0]], dtype=complex)

H_matrix = np.array([[ 0.70710678+0.j,  0.70710678+0.j], 
[ 0.70710678+0.j, -0.70710678+0.j]], dtype=complex)

CNOT_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

CNOT_inverse_matrix = np.array(
    [[1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]]
    , dtype=complex)

CZ_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
    ], dtype=complex)

CZ_CNOT_matrix = np.array(
    [[ 1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0, -1],
    [ 0,  0,  1,  0]]        
    , dtype=complex)

CZ_CNOT_CZ_matrix = np.array(
    [[ 1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0, -1],
    [ 0,  0,  -1,  0]]        
    , dtype=complex)

CNOT_CZ_matrix = np.array(
    [[ 1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0, 1],
    [ 0,  0,  -1,  0]]        
    , dtype=complex)

CNOT_CZ_CNOT_matrix = np.array(
    [[ 1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  -1, 0],
    [ 0,  0,  0,  1]]        
    , dtype=complex)

def is_identity(gate):
    return np.allclose(gate, np.eye(len(gate)))

def is_unitary(matrix, tol=1e-10):
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        return False
    identity = np.eye(matrix.shape[0])
    return np.allclose(matrix.conj().T @ matrix, identity, atol=tol)