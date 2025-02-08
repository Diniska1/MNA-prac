import math
from scipy.integrate import quad
import scipy.special

# FUNCTION quad(func, a, b) return array [integr, err], where
# integr - value of integral of func on [a, b]
# err - error while integrating

# FUNCTION scipy.special.lpn(n, t) returns array [a, b], where:
# a - array [val_0, val_1, ... val_n] - where val_i = P_i(t)
# b - array [value_0, value_1, ... , value_n] - where value_i = p_i(t)
# where P_k(x) - Legendre polynomial degree of k,
# p_k(x) - derivative of P_k(x)


def x_func(t, a, b):
    return (a + b)/2 + (b - a) * t / 2

def f1(x):
    """x"""
    return x

def f2(x):
    """x^2"""
    return x * x

def f3(x):
    """1/(1 + x^2)"""
    return 1/(1 + x * x)


def mul(t, i):
    def f(x, t = t, i = i):
        res = 1
        for j in range(len(t)):
            if i != j:
                res *= (x - t[j]) / (t[i] - t[j])
        return res
    return f


func_to_integrate = f3              # Choose function to integrate

print("Choose sector [a, b]:")
a = float(input("Enter a:"))
b = float(input("Enter b:"))
# a, b = -2, 2
if a >= b:
    print("a >= b")
    exit()
n = (input("Enter amount of nodes (default=10):"))
# n = 10
if n == '':
    n = 10
n = int(n)


print(f"\nIntegrating function \n{func_to_integrate.__name__}(x) = {func_to_integrate.__doc__} on sector [{a}, {b}]")


# Newton-Cotes quad
# MNA paragraph 16.2

t = [i/n * 2 - 1 for i in range(n + 1)]         # grid on [-1, 1] with n + 1 nodes
x = [x_func(t[i], a, b) for i in range(len(t))] # grid on [a, b] from [-1, 1]

newton_cotes = 0
error = []
for i in range(n + 1):
    integr = quad(mul(t, i), -1, 1)
    error.append(integr[1])
    d = integr[0]                               # integral from mul on [-1, 1]
    newton_cotes += d * func_to_integrate(x[i])
newton_cotes *= (b - a) / 2
if (maxer := max(error)) > 1:
    print(f"Error while integrating is too big ({maxer}). Reduce amount nodes (n)")
    exit()

#print("MAXERROR:", max(error))
#print(f"Value of integral of func f3 on [{a}, {b}] on grid with {n} nodes is\n{newton_cotes}")



# GAUSS quad
# Info  https://courses.igankevich.com/numerical-methods/notes/numerical-integration/

roots = scipy.special.p_roots(n)[0]             # array of roots of Legendre polynom degree of n
gauss_quad = 0
for i in range(n):
    val = scipy.special.lpn(n, roots[i])[1][-1] # the value of the derivative of the Legendre polynomial degree of n
                                                # at the point roots[i]
    denumerator = (1 - roots[i] ** 2) * (val ** 2)
    coef = 2 / denumerator
    gauss_quad += coef * func_to_integrate(x_func(roots[i],a,b))
gauss_quad *= (b - a) / 2




# CLENSHAW-CURTIS quad
# Formulas get from https://arxiv.org/pdf/1401.0638

x_curt = [math.cos(j * math.pi / n) for j in range(n + 1)]
delta = [0.5 if i == 0 or i == n else 1 for i in range(n + 1)]
coef_curt = []
# forming array with curtis coefficients
for i in range(n + 1):
    d = 0
    for k in range(n // 2):
        d += delta[2 * k] * math.cos(2 * i * k * math.pi / n) / (1 - 4 * k * k)
    d *= 4 * delta[i] / n
    coef_curt.append(d)

clenshaw_curtis = 0
for i in range(n + 1):
    clenshaw_curtis += coef_curt[i] * func_to_integrate(x_func(x_curt[i],a,b))
clenshaw_curtis *= (b - a) / 2


true_val = quad(func_to_integrate,a,b)[0]

print(f"\nTrue value: {quad(func_to_integrate,a,b)}")

print("{:17}|{:19}|{:10}".format("Method name", "Value\t", "I - I_approx"))
print("{:17}|{:19}|{:10}".format("NEWTON-COTES",    newton_cotes,    abs(true_val - newton_cotes)))
print("{:17}|{:19}|{:10}".format("GAUSS QUAD",      gauss_quad,      abs(true_val - gauss_quad)))
print("{:17}|{:19}|{:10}".format("CLENSHAW-CURTIS", clenshaw_curtis, abs(true_val - clenshaw_curtis)))

# ========
# OUTPUT
# =======

# Integrating function
# f3(x) = 1/(1 + x^2) on sector [-1.0, 1.0]
#
# True value: (1.5707963267948968, 1.7439342485646153e-14)
# Method name      |Value	             |I - I_approx
# NEWTON-COTES     | 1.5709553615851541|0.0001590347902573619
# GAUSS QUAD       | 1.5707962702232696|5.6571627160550975e-08
# CLENSHAW-CURTIS  | 1.5707917818094437|4.544985453103223e-06