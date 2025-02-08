import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from math import factorial


def get_points(a, b, n, deg, mode):
    deg = deg + 1
    if mode == 'uniform':
        points = np.linspace(a, b, n)
    elif mode == 'random':
        points = np.sort(np.random.rand(n) * (b - a) + a)
    elif mode == 'chebyshev':
        points = (np.polynomial.chebyshev.chebpts1(n) + 1) / 2 * (b - a) + a
    else:
        print(f'incorrect mode {mode}')
        points = 0
    new_points = []

    for el in points:
        for _ in range(deg):
            new_points.append(el)
    return np.array(new_points)

def plot(f1, f2, a, b, deg, name):
    plot_points = get_points(a, b, int(100 * (b - a)), 0, 'uniform')
    plt.plot(plot_points, f1(plot_points), label='interpolation')
    plt.plot(plot_points, f2(plot_points, 0), label='function')
    plt.title(name)
    plt.legend()
    plt.grid()
    plt.show()

def sin(x, deg=0):
    if deg % 4 == 0:
        return np.sin(x)
    elif deg % 4 == 1:
        return np.cos(x)
    elif deg % 4 == 2:
        return -np.sin(x)
    elif deg % 4 == 3:
        return -np.cos(x)


# get matrix divided_differences with equal nodes
def divided_differences(n, p, f, deg):
    deg = deg + 1
    matrix = np.zeros((n * deg, n * deg))
    for i in range(n):
        for j in range(deg):
            for k in range(0, j + 1):
                matrix[i * deg + j][k] = f(p[i * deg + j], k) / factorial(k)
            for k in range(j + 1, j + 1 + i * deg):
                matrix[i * deg + j][k] = (matrix[i * deg + j][k - 1] - matrix[i * deg + j - 1][k - 1]) / (p[i * deg + j] - p[i * deg + j - k])
    return matrix

def hermit_interpol(n, a, b, f, deg, mode):
    n = n + 1
    points = np.array(get_points(a, b, n, deg, mode))
    diff = divided_differences(n, points, f, deg)
    poly = Polynomial([diff[0][0]])
    for i in range(1, n * (deg + 1)):
        poly += diff[i][i] * Polynomial.fromroots(points[:i])
    plot(poly, f, a, b, deg, 'Hermit ' + mode + f' {deg=}')


hermit_interpol(5, -5, 5, sin, 1, 'uniform')
hermit_interpol(5, -5, 5, sin, 1, 'chebyshev')