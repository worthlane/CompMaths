import numpy as np

def multiplyMatrixVector(A, x):
    rows = len(A)
    cols = len(A[0])

    if len(x) != cols:
        raise ValueError("Матрицу и вектор нельзя перемножить")

    b = np.zeros(rows, dtype=float)

    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += A[i][j] * x[j]
        b[i] = s

    return b

def forwardPass(L, b):
    n = len(b)
    z = np.zeros(n, dtype=float)

    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * z[j]
        z[i] = (b[i] - s) / L[i, i]

    return z

def backwardPass(U, z):
    n = len(z)
    y = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * y[j]

        if abs(U[i, i]) < 1e-12:
            raise ValueError("Матрица вырождена")

        y[i] = (z[i] - s) / U[i, i]

    return y

def gaussMethod(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A не квалратная")

    n = A.shape[0]
    if len(b) != n:
        raise ValueError("Размеры вектора b и матрицы A не совпадают")

    U = A.copy()
    L = np.eye(n, dtype=float)
    Q = np.eye(n, dtype=float)

    for k in range(n):
        pivot_col = k
        max_value = abs(U[k, k])

        for j in range(k + 1, n):
            if abs(U[k, j]) > max_value:
                max_value = abs(U[k, j])
                pivot_col = j

        if abs(max_value) < 1e-12:
            raise ValueError(f"На шаге {k} не найден ненулевой главный элемент")

        if pivot_col != k:
            U[:, [k, pivot_col]] = U[:, [pivot_col, k]]
            Q[:, [k, pivot_col]] = Q[:, [pivot_col, k]]

            if k > 0:
                L[:k, [k, pivot_col]] = L[:k, [pivot_col, k]]

        for i in range(k + 1, n):
            if abs(U[k, k]) < 1e-12:
                raise ValueError(f"Нулевой pivot на шаге {k}")

            L[i, k] = U[i, k] / U[k, k]

            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]

    z = forwardPass(L, b)
    y = backwardPass(U, z)
    x = multiplyMatrixVector(Q, y)

    return L, U, Q, x

def getNormDiff(A, B):
    return np.linalg.norm(A - B)

def printVariable(name, V):
    print(f"{name} =")
    print(V)
    print()

def testOneCase(A, b, case_name):
    print(case_name)

    printVariable("A", A)
    printVariable("b", b)

    L, U, Q, x = gaussMethod(A, b)
    x_lib = np.linalg.solve(A, b)

    printVariable("L", L)
    printVariable("U", U)
    printVariable("Q", Q)

    print("x (метод Гаусса) =")
    print(x)
    print("x_lib (numpy.linalg.solve) =")
    print(x_lib)

    factorization_error = getNormDiff(A @ Q, L @ U) # в тестах же надеюсь можно юзать перемножение матриц чтобы сверить честно
    solution_error      = getNormDiff(x, x_lib)
    residual_error      = getNormDiff(A @ x, b)

    print(f"||AQ - LU||   = {factorization_error:.10e}")
    print(f"||x - x_lib|| = {solution_error:.10e}")
    print(f"||Ax - b||    = {residual_error:.10e}")
    print()


def main():
    A1 = np.array([
        [2, -4, 1],
        [4, -2, 1],
        [1, -4, 6]
    ], dtype=float)
    b1 = np.array([-3, 3, 11], dtype=float)

    A2 = np.array([
        [1, 7, 3],
        [2, 1, 8],
        [9, 6, 1]
    ], dtype=float)
    b2 = np.array([10, 11, 12], dtype=float)

    A3 = np.array([
        [0, 2, 9, 4],
        [3, 1, 5, 2],
        [1, 6, 2, 3],
        [2, 3, 4, 9]
    ], dtype=float)
    b3 = np.array([7, 8, 9, 10], dtype=float)

    testOneCase(A1, b1, "Тест 1")
    testOneCase(A2, b2, "Тест 2")
    testOneCase(A3, b3, "Тест 3")

if __name__ == "__main__":
    main()