
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

def find_max(f, poly, a, b, n=1000):
    points = get_points(a, b, n, 'uniform')
    m = 0
    point = 0
    for x in points:
        r = np.abs(f(x) - poly(x))
        if r > m:
            m = r
            point = x
    return point

def remez(approx_degree, f, a, b, eps, mode, n):
    iters = 0
    err = np.inf
    points = get_points(a, b, approx_degree + 2, mode)
    matrix = np.column_stack((np.vander(points, approx_degree + 1, increasing=True), np.array([(-1)**(i+1) for i in range(approx_degree + 2)])))
    fx = np.array([f(x) for x in points])
    plot_points = get_points(a, b, int((b - a)*100), 'uniform')
    save = 0
    while err > eps:
        iters += 1
        res = np.linalg.solve(matrix, fx)   #solve system
        poly = Polynomial(res[:-1])
        d = res[-1]
        max_point = find_max(f, poly, a, b, n) #find max f - p
        max_value = np.abs(f(max_point) - poly(max_point))
        new_err = np.abs(max_value - np.abs(d))
        if iters > 20 or new_err > err:
            break
        save = poly
        err = new_err

        #GRAPH
        # plt.plot(plot_points, poly(plot_points), label='poly')
        # plt.plot(plot_points, f(plot_points), label='f')
        # plt.legend()
        # plt.grid()
        # plt.show()


        #replace
        index = np.argwhere(points > max_point)
        ind = int(index.min(initial=approx_degree + 1))
        matrix[ind, :-1] = np.array([max_point**i for i in range(approx_degree + 1)])
        fx[ind] = f(max_point)
        points[ind] = max_point


    #GRAPH
    max_point = find_max(f, save, a, b, n)
    max_value = np.abs(f(max_point) - poly(max_point))
    print(f'{iters} iterations to get {err} error with {eps=}')
    print(f'max difference = {max_value:.2f} at x = {max_point}')
    plt.plot(plot_points, save(plot_points), label='poly')
    plt.plot(plot_points, f(plot_points), label='f')
    plt.legend()
    plt.grid()
    plt.show()
    print("res polynom:\n",poly)

remez(3, np.sin, -5, 5, 1e-5, 'chebyshev', 1000)
remez(10, np.abs, -1, 1, 1e-5, 'uniform', 1000)
remez(10, np.abs, -1, 1, 1e-5, 'chebyshev', 1000)
