import numpy as np
from math import factorial
import random

BOHR_RADIUS = 5.292 * 10 ** -11


def legendre(l,x):
    if l == 0:
        return 1
    elif l == 1:
        return x
    elif l == 2:
        return 1/2*(3*x**2 - 1)
    elif l == 3:
        return 1/2*(5*x**3 - 3*x)

def associated_legendre(m,l,x):
    if np.abs(m) > l:
        return
    if m < 0:
        m_dash = np.abs(m)
        return (-1)**m_dash * factorial(l - m_dash)/factorial(l + m_dash) * associated_legendre(m_dash, legendre, x)
    else:
        return (-1)**m *(1-x**2)**(m/2)*derivative(m, legendre, x)


def derivative(m, f: function, x):
    h = 0.0000000001
    if m == 0:
        return function(x)
    elif m == 1:
        return (f(x+h)-f(h))/h
    elif m == 2:
        return (f(x+h) - 2*f(h) + f(x-h))/(h**2)
    elif m == 3:
        return (-f(x-2*h) + 2*f(x-h) - 2*f(x+h) + f(x+2*h))/(2*h**3)
    

def spherical_harmonics(l, m ,theta):
    T = (-1)**m * np.sqrt(((2*l + 1) *factorial(l-m))/(4*np.pi*factorial(l+m))) * associated_legendre(m, l, np.cos(theta))
    return T

def radial_harmonics(n, l, r):
    rho = r/(BOHR_RADIUS*n)
    return rho**l * np.exp(-rho) * legendre(2*rho)


# def monte_carlo_integrate(f: function, a, b, h):
#     n = 