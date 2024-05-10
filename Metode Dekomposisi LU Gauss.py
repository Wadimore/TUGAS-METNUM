import numpy as np

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1

    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        L[k+1:, k] = (A[k+1:, k] - L[k+1:, :] @ U[:, k]) / U[k, k]

    return L, U

def solve_lu(A, b):
    L, U = lu_decomposition(A)
    n = len(A)
    y = np.zeros(n)
    x = np.zeros(n)

    # Solve Ly = b using forward substitution
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # Solve Ux = y using backward substitution
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

# Input the matrix A and vector b
A = np.array([[4, 3, -2], [6, 4, -3], [-8, 3, -1]])
b = np.array([4, 3, 1])

# Solve the system
x = solve_lu(A, b)
print("Solusi sistem persamaan linear:")
print(x)