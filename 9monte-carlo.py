
import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc

def find_min_max(f, a, b):
    points = np.linspace(a, b, int((b - a) * 10))
    max = -np.inf
    min = np.inf
    for x in points:
        val = f(x)
        if val > max:
            max = val
        if val < min:
            min = val
    return min, max



def monte_carlo_std(f, a, b, eps):
    integration = quad(f, a, b)
    n = int( (3 * (b - a) / eps)**2 / 12) + 1
    points = a + np.random.rand(n) * (b - a)
    res = 0
    for i in range(n):
        res += f(points[i])
    res *= (b - a) / n
    print('monte carlo std res =', res)
    print('scipy integration result = ', integration[0])
    print('error =', np.abs(res - integration[0]))
    return res

def monte_carlo_points(f, a, b, eps):
    integration = quad(f, a, b)
    n = int( (3 * (b - a) / eps)**2 / 12) + 1
    min, max = find_min_max(f, a, b)
    points_x = a + np.random.rand(n) * (b - a)
    points_y = np.random.rand(n) * (max - min) + min
    res = 0
    for i in range(n):
        res += (f(points_x[i]) >= points_y[i])
    res *= (b - a) * (max - min) / n
    print('monte carlo_points res =', res)
    print('scipy integration result = ', integration[0])
    print('error =', np.abs(res - integration[0]))
    return res

def monte_carlo_sobol(f, a, b, eps):
    integration = quad(f, a, b)
    n = int( (3 * (b - a) / eps)**2 / 12) + 1
    min, max = find_min_max(f, a, b)

    sampler = qmc.Sobol(d=2, scramble=False)
    points = sampler.random(n)
    points_x = [a + x[0] * (b - a) for x in points]
    points_y = [x[1] * (max - min) + min for x in points]

    res = 0
    for i in range(n):
        res += (f(points_x[i]) >= points_y[i])
    res *= (b - a) * (max - min) / n
    print('monte carlo_sobol res =', res)
    print('scipy integration result = ', integration[0])
    print('error =', np.abs(res - integration[0]))
    return res



def monte_carlo_scrambled_sobol(f, a, b, eps):
    integration = quad(f, a, b)
    n = int( (3 * (b - a) / eps)**2 / 12) + 1
    min, max = find_min_max(f, a, b)

    sampler = qmc.Sobol(d=2, scramble=True)
    points = sampler.random(n)
    points_x = [a + x[0] * (b - a) for x in points]
    points_y = [x[1] * (max - min) + min for x in points]

    res = 0
    for i in range(n):
        res += (f(points_x[i]) >= points_y[i])
    res *= (b - a) * (max - min) / n
    print('monte carlo_scrambled_sobol res =', res)
    print('scipy integration result = ', integration[0])
    print('error =', np.abs(res - integration[0]))
    return res

eps = 0.01
monte_carlo_std(np.exp, -5, 5, eps)
monte_carlo_points(np.exp, -5, 5, eps)
monte_carlo_sobol(np.exp, -5, 5, eps)
monte_carlo_scrambled_sobol(np.exp, -5, 5, eps)


# =======
# OUTPUT
# =======

# monte carlo std res = 148.47645991775215
# scipy integration result =  148.40642115557753
# error = 0.0700387621746188
#
# monte carlo_points res = 148.8850806943986
# scipy integration result =  148.40642115557753
# error = 0.4786595388210628
#
# monte carlo_sobol res = 148.33894579272592
# scipy integration result =  148.40642115557753
# error = 0.06747536285161004
#
# monte carlo_scrambled_sobol res = 148.31124329771356
# scipy integration result =  148.40642115557753
# error = 0.09517785786397326