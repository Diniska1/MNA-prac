import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from math import factorial


def get_points(a, b, n, mode):
    if mode == 'uniform':
        points = np.linspace(a, b, n)
    elif mode == 'random':
        points = np.concatenate(([a], np.sort(np.random.rand(n - 2) * (b - a) + a), [b]))
    elif mode == 'chebyshev':
        points = (np.polynomial.chebyshev.chebpts1(n) + 1) / 2 * (b - a) + a
    else:
        print(f'incorrect mode {mode}')
        points = 0
    return np.array(points)


def plot(splines, f, a, b, spline_points, name):
    plot_points = get_points(a, b, int(1000 * (b - a)), 'uniform')
    plt.plot(plot_points, f(plot_points), color='b', label='function')
    for i, s in enumerate(splines):
        points = plot_points[(plot_points >= spline_points[i]) & (plot_points <= spline_points[i + 1])]
        points_args = (points - points.min()) / (points.max() - points.min())
        plt.plot(points, splines[i](points_args), color='r')
    plt.plot(spline_points, f(spline_points), 'ro')
    plt.title(name)
    plt.legend()
    plt.grid()
    plt.show()

def spline_interpol(n, a, b, f, mode):
    points = get_points(a, b, n + 1, mode)
    h = np.array([points[i] - points[i - 1] for i in range(1, n + 1)])
    fx = f(points)
    matrix = np.zeros((n - 1, n - 1))
    matrix[0][0] = 2 * (h[0] + h[1])
    matrix[0][1] = h[1]
    for i in range(1, n - 2):
        matrix[i][i - 1] = h[i]
        matrix[i][i] = 2 * (h[i] + h[i + 1])
        matrix[i][i + 1] = h[i + 1]
    matrix[n - 2][n - 2] = 2 * (h[n - 2] + h[n - 1])
    matrix[n - 2][n - 3] = h[n - 2]
    deltaf = np.zeros(n)
    for i in range(n):
        deltaf[i] = (fx[i + 1] - fx[i]) / h[i]
    ro = np.zeros((n - 1))
    for i in range(n - 1):
        ro[i] = 6 * (deltaf[i + 1] - deltaf[i])


    u = np.linalg.solve(matrix, ro)
    u = np.concatenate(([0], u, [0]))

    splines = []
    at = Polynomial([-1 / 6, 1 / 6]) - 1 / 6 * Polynomial.fromroots([1, 1, 1])  # Polynomial.fromroots([0, 1, 2]) / -6
    bt = Polynomial([0, -1 / 6, 0, 1 / 6])
    for k in range(n):
        spline = Polynomial([1, -1]) * fx[k] + Polynomial([0, 1]) * fx[k + 1] + u[k] * h[k] * h[k] * at + u[k + 1] * h[k] * h[k] * bt
        splines.append(spline)
    plot(splines, f, a, b, points, 'spline interpolation ' + mode)

np.random.seed(0)

spline_interpol(6, -1, 1, np.abs, 'random')
spline_interpol(6, -1, 1, np.abs, 'uniform')
spline_interpol(6, -1, 1, np.abs, 'chebyshev')