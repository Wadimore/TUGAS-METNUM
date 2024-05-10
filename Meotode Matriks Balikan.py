import numpy as np
import scipy.linalg

# Metode matriks balikan
def solve_using_inverse(matrix_A, vector_b):
    A_inv = np.linalg.inv(matrix_A)
    x = np.dot(A_inv, vector_b)
    return x

# Test Perhitungan
def test():
    matrix_A = np.array([[2, 3, -2], [4, 2, -3], [-2, 3, -1]])
    vector_b = np.array([5, 3, 1])
    
    # metode matriks balikan
    x_inverse = solve_using_inverse(matrix_A, vector_b)
    print("Solusi menggunakan metode matriks balikan:", x_inverse)

if __name__ == "__main__":
    test()
