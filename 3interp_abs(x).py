import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

def f(x):
    return np.abs(x)

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

def vander_interpol(n, a, b, mode):
    points = get_points(a, b, n + 1, mode)
    matrix = np.vander(points, increasing=True)
    fx = f(points)
    coefs = np.linalg.solve(matrix, fx)
    poly = Polynomial(coefs)
    plot(poly, f, a, b, 'Vandermonde ' + mode)


# MAKE ORTH POLYNOMS SYSTEM
def deg_integral(coef, n, a, b):
    '''integral from a to b of coef * x**n'''
    return (b**(n + 1) - a**(n + 1)) / (n + 1) * coef

def scalar_prod(p, q, a, b):
    res = 0
    for i in range(len(p)):
        for j in range(len(q)):
            res += deg_integral(p[i] * q[j], i + j, a, b)
    return res

def gramm_matrix(n, a, b):
    g = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            g[i, j] = deg_integral(1, i + j, a, b)
    return g

def orthogonal_polynomials_cholesky(n, a, b):
    polys = []
    S = np.linalg.inv(np.linalg.cholesky(gramm_matrix(n, a, b)))
    for i in range(n):
        coeffs = S[i][:i+1]
        polys.append(Polynomial(coeffs))

    return polys


def orthogonal_interpol(n, a, b, mode):
    n = n + 1
    points = get_points(a, b, n, mode)
    polys = orthogonal_polynomials_cholesky(n, a, b)
    matrix = np.zeros((n, n))
    fx = f(points)
    for i in range(n):
        for j in range(n):
            matrix[i][j] = polys[i](points[j])
    matrix = matrix.T
    coefs = np.linalg.solve(matrix, fx)
    res = Polynomial([0])
    for i, c in enumerate(coefs):
        res += c * polys[i]
    plot(res, f, a, b, 'Orthogonal ' + mode)

def lagrange_interpol(n, a, b, mode):
    points = get_points(a, b, n + 1, mode)
    p = Polynomial.fromroots(points)
    w = p.deriv()
    l = Polynomial([0])
    for i, x in enumerate(points):
        new_points = list(points[:i]) + list(points[i+1:])
        l_i = Polynomial.fromroots(new_points) * f(x) / w(x)
        l += l_i
    plot(l, f, a, b, 'Lagrange ' + mode)

vander_interpol(10, -1, 1, 'uniform')
vander_interpol(10, -1, 1, 'chebyshev')

orthogonal_interpol(10, -1, 1, 'uniform')
orthogonal_interpol(10, -1, 1, 'chebyshev')

lagrange_interpol(10, -1, 1, 'uniform')
lagrange_interpol(10, -1, 1, 'chebyshev')