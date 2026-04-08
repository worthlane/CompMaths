import numpy as np
import matplotlib.pyplot as plt
import time

from numba import jit

@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True)
def multiplyMatrices(A, B):
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])

    if colsA != rowsB:
        raise ValueError("Нельзя перемножить матрицы")

    C = np.zeros((rowsA, colsB), dtype=float)

    for i in range(rowsA):
        for j in range(colsB):
            s = 0.0
            for k in range(colsA):
                s += A[i][k] * B[k][j]
            C[i][j] = s

    return C

def transposeMatrix(A):
    rows = len(A)
    cols = len(A[0])
    B = np.zeros((cols, rows), dtype=float)

    for i in range(rows):
        for j in range(cols):
            B[j][i] = A[i][j]

    return B

@jit(nopython=True, cache=True)
def subtractVectors(a, b):
    n = len(a)
    result = np.zeros(n, dtype=float)

    for i in range(n):
        result[i] = a[i] - b[i]

    return result

@jit(nopython=True, cache=True)
def vectorNorm(v):
    s = 0.0
    for i in range(len(v)):
        s += v[i] * v[i]
    return np.sqrt(s)

@jit(nopython=True, cache=True)
def zeidelMethod(A, b, max_iter, tol):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)
    residuals = np.empty(max_iter, dtype=np.float64)
    iters = 0

    for _ in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sum1 = 0.0
            for j in range(i):
                sum1 += A[i, j] * x[j]

            sum2 = 0.0
            for j in range(i + 1, n):
                sum2 += A[i, j] * x_old[j]

            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        Ax = multiplyMatrixVector(A, x)
        r = subtractVectors(Ax, b)
        res_norm = vectorNorm(r)
        residuals[iters] = res_norm
        iters += 1

        if res_norm < tol:
            break

    return x, residuals[:iters]


def createSPDMatrix(n):
    R = np.random.randn(n, n)
    A = multiplyMatrices(transposeMatrix(R), R)

    for i in range(n):
        A[i, i] += n

    return A

def main():
    np.random.seed(67)

    n = 120
    max_iter = 10000
    tol = 1e-8

    A = createSPDMatrix(n)
    x_true = np.random.randn(n)
    b = multiplyMatrixVector(A, x_true)

    start_custom = time.perf_counter()
    x_zeidel, residuals = zeidelMethod(A, b, max_iter, tol)
    end_custom = time.perf_counter()

    start_original = time.perf_counter()
    x_original = np.linalg.solve(A, b)
    end_original = time.perf_counter()

    error = vectorNorm(subtractVectors(x_zeidel, x_original))
    iterations = len(residuals)

    print(f"Размеры сгенерированной матрицы: {n}")
    print(f"Число итераций: {iterations}")
    print(f"Точная ошибка: {error:.10e}")
    print(f"Время работы numpy.linalg.solve: {end_original - start_original:.10f} с")
    print(f"Время работы метода Зейделя: {end_custom - start_custom:.10f} с")

    plt.figure()
    plt.semilogy(range(1, len(residuals) + 1), residuals)
    plt.xlabel("Номер итерации")
    plt.ylabel("$\\log(\\|Ax - b\\|)$")
    plt.title("Зависимость логарифма невязки от номера итерации")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
