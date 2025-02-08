import numpy as np
from scipy.linalg import eigh_tridiagonal
from itertools import combinations
from functools import reduce
from operator import mul
from math import gamma, pi, factorial, sqrt

# sector [-1, 1]
# weight w(x) = 1

def integral_1(val):
    if val % 2:
        return 0
    else:
        return 2 / (val + 1)


def dot_product(first_mas, second_mas, corr, integral):
    val = 0
    for ind_i, x in enumerate(first_mas):
        for ind_j, k in enumerate(second_mas):
            val += x * k * integral(ind_i + ind_j + corr)
    return val


def test(L, n, integral, tolerance):
    flag = True

    #check L*L^T = I
    for i in range(0, n):
        val = dot_product(L[i, :(i + 1)], L[i, :(i + 1)], 0, integral)

        if (np.abs(np.abs(val) - 1) >= tolerance):
            print("BRUH")
            flag = False
            break

        for j in range(i + 1, n):
            val = dot_product(L[i, :(i + 1)], L[j, :(j + 1)], 0, integral)

            if (np.abs(val) >= tolerance):
                flag = False

    if flag:
        print("Tolerance achieved")
        print(L)
    else:
        print("Can't achive that tolerance")


eps = 10 ** (-2)
integral = integral_1


def gramm(n):
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = integral(i + j)
    L = np.linalg.inv(np.linalg.cholesky(G)) # lower triag
    test(L, n, integral, eps)

def recurrent(n):
    L = np.zeros((n, n))
    L[0, 0] = np.sqrt(0.5)
    beta = 0

    for i in range(1, n):
        alpha = dot_product(L[i - 1], L[i - 1], 1, integral) / dot_product(L[i - 1], L[i - 1], 0, integral)
        L[i, 1:(i + 1)] = L[(i - 1), :i] # b_n * L_n+1 = x * L_n
        L[i, :i] -= alpha * L[i - 1, :i] # b_n * L_n+1 -= a_n * L_n


        # b_n * L_n+1 -= b_n-1 * L_n-1
        if beta:
            L[i, :(i - 1)] -= beta * L[i - 2, :(i - 1)]
        beta = np.sqrt(dot_product(L[i], L[i], 0, integral)) # b_n
        L[i, :(i + 1)] /= beta # L_n+1 /= b_n

    test(L, n, integral, eps)


def eigenval(n):
    L = np.zeros((n, n))
    L[0, 0] = np.sqrt(0.5)
    mas_alpha = []
    mas_beta = []

    for i in range(1, n):
        alpha = dot_product(L[i - 1], L[i - 1], 1, integral)
        mas_alpha.append(alpha)
        roots = eigh_tridiagonal(np.array(mas_alpha), np.array(mas_beta))[0]

        for j in range(i):
            val = 0
            for x in [*combinations(roots, i - j)]:
                val += reduce(mul, x)
            L[i, j] = (-1) ** (i - j) * val
        L[i, i] = 1

        norm = np.sqrt(dot_product(L[i], L[i], 0, integral))
        L[i, :(i + 1)] /= norm
        beta = dot_product(L[i], L[i - 1], 1, integral)
        mas_beta.append(beta)

    test(L, n, integral, eps)

gramm(4)
recurrent(4)
eigenval(4)

