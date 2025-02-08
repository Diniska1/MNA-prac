import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

def get_points(a, b, n, mode):
    if mode == 'uniform':
        points = np.linspace(a, b, n)
    elif mode == 'random':
        points = np.sort(np.random.rand(n) * (b - a) + a)
    elif mode == 'chebyshev':
        points = (np.polynomial.chebyshev.chebpts1(n) + 1) / 2 * (b - a) + a
    else:
        print(f'incorrect mode {mode}')
        points = 0
    return points

def plot(f1, f2, a, b, name):
    plot_points = get_points(a, b, int(100 * (b - a)), 'uniform')
    plt.plot(plot_points, f1(plot_points), label='interpolation')
    plt.plot(plot_points, f2(plot_points), label='function')
    plt.title(name)
    plt.legend()
    plt.grid()
    plt.show()

def f(x):
    return np.abs(x)

#get matrix divided_differences
def divided_differences(n, p, f):
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i][0] = f(p[i])
        for j in range(1, i + 1):
            matrix[i][j] = (matrix[i][j - 1] - matrix[i - 1][j - 1]) / (p[i] - p[i - j])
    return matrix

def newton_interpol(n, a, b, f, mode):
    n = n + 1
    points = get_points(a, b, n, mode)
    diff = divided_differences(n, points, f)
    poly = Polynomial([diff[0][0]])
    for i in range(1, n):
        poly += diff[i][i] * Polynomial.fromroots(points[:i])
    plot(poly, f, a, b, 'Newton ' + mode)

newton_interpol(10, -1, 1, f, 'uniform')
newton_interpol(10, -1, 1, f, 'chebyshev')