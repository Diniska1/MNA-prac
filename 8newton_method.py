import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.misc import derivative

def newton(f, start, tol):
    x = start
    n = 0
    while np.abs(f(x) / derivative(f, x, dx=1e-6)) >= tol:
        x = x - f(x) / derivative(f, x, dx=1e-6)
        n += 1
    print(f'{n=} \n{f(x)=} \n{x=}\n')

newton(np.sin, start=0.5, tol=1e-10)
newton(np.cos, start=1.0, tol=1e-10)
newton(Polynomial.fromroots([5, 10, 15]), start=161234.34623, tol=1e-10)
newton(np.exp, start=12.321412, tol=1e-10)

# ======
# OUTPUT
# ======

# n=3
# f(x)=np.float64(-1.2113614690327512e-14)
# x=np.float64(-1.2113614690327512e-14)
#
# n=3
# f(x)=np.float64(-6.012355577970274e-13)
# x=np.float64(1.5707963267954979)
#
# n=30
# f(x)=np.float64(2.3081838662619703e-09)
# x=np.float64(15.000000000046164)
#
# n=745
# f(x)=np.float64(0.0)
# x=np.float64(-inf)