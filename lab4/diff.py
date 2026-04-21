
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def chebyshev_diff_matrix(a, b, n):
    if n < 2:
        raise ValueError("n must be at least 2")

    N = n - 1
    t = np.cos(np.pi * np.arange(n) / N)
    x = 0.5 * (a + b) + 0.5 * (b - a) * t

    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(n))

    X = np.tile(t, (n, 1))
    dX = X.T - X

    D = (np.outer(c, 1.0 / c)) / (dX + np.eye(n))
    D = D - np.diag(np.sum(D, axis=1))
    D = (2.0 / (b - a)) * D
    return x, D


def approximate_derivative(a, b, n, f):
    x, D = chebyshev_diff_matrix(a, b, n)
    values = f(x)
    derivative = np.dot(D, values)
    return x, derivative


def test_function(x):
    return np.exp(x)


def test_derivative(x):
    return np.exp(x)


def main():
    a = -1.0
    b = 1.0
    node_counts = list(range(4, 41))
    errors = []

    for n in node_counts:
        x, d_approx = approximate_derivative(a, b, n, test_function)
        d_exact = test_derivative(x)
        error = np.max(np.abs(d_approx - d_exact))
        errors.append(error)

    plt.figure(figsize=(8, 5))
    plt.semilogy(node_counts, errors, marker='o', markersize=3)
    plt.xlabel('Число узлов n')
    plt.ylabel(r"$\|f' - Df\|_\infty$")
    plt.title('Зависимость ошибки от числа узлов')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / 'cheb_lab_v2'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'error_plot.png', dpi=200)
    plt.close()

    for n, err in zip(node_counts, errors):
        print(n, err)


if __name__ == '__main__':
    main()
