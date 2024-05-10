import numpy as np
import scipy.linalg

# Metode dekomposisi Crout
def solve_using_crout(matrix_A, vector_b):
    LU, piv = scipy.linalg.lu_factor(matrix_A)
    L = np.tril(LU, k=-1) + np.eye(len(matrix_A))
    U = np.triu(LU)
    y = np.linalg.solve(L, vector_b)
    x = np.linalg.solve(U, y)
    return x

def test():
    matrix_A = np.array([[4, 3, -1], [2, 4, -3], [-2, 1, -1]])
    vector_b = np.array([5, 3, 1])

    #metode dekomposisi Crout
    x_crout = solve_using_crout(matrix_A, vector_b)
    print("Solusi menggunakan metode dekomposisi Crout:", x_crout)

if __name__ == "__main__":
    test()
