import numpy as np
from math import factorial
import random
from scipy.integrate import quad

BOHR_RADIUS = 5.292 * 10**-11


def legendre(l, x):
    if l == 0:
        return 1
    elif l == 1:
        return x
    elif l == 2:
        return 1 / 2 * (3 * x**2 - 1)
    elif l == 3:
        return 1 / 2 * (5 * x**3 - 3 * x)
    else:
        raise KeyError("l must be within 0 to 3")


def legendre_derivative(l, m, x):
    h = 1e-6
    if m == 0:
        return legendre(l, x)
    elif m == 1:
        return (legendre(l, x + h) - legendre(l, x)) / h
    elif m == 2:
        return (legendre(l, x + h) - 2 * legendre(l, x) + legendre(l, x - h)) / (h**2)
    elif m == 3:
        return (
            -legendre(l, x - 2 * h)
            + 2 * legendre(l, x - h)
            - 2 * legendre(l, x + h)
            + legendre(l, x + 2 * h)
        ) / (2 * h**3)


def associated_legendre(l, m, x):
    if m < 0:
        m_dash = np.abs(m)
        return (
            (-1) ** m_dash
            * factorial(l - m_dash)
            / factorial(l + m_dash)
            * associated_legendre(l, m_dash, x)
        )
    else:
        return (-1) ** m * (1 - x**2) ** (m / 2) * legendre_derivative(l, m, x)


def associated_laguerre(p, q, x):
    s = 0
    for i in range(p + 1):
        numerator = (-1) ** i * factorial(p + q) * x**i
        denominator = factorial(p - i) * factorial(q + i) * factorial(i)
        s += numerator / denominator
    return s


def spherical_harmonics(l, m, theta):
    if np.abs(m) > l:
        raise ValueError("Magnitude of m must be larger than l")
    T = (
        (-1) ** m
        * np.sqrt(((2 * l + 1) * factorial(l - m)) / (4 * np.pi * factorial(l + m)))
        * associated_legendre(l, m, np.cos(theta))
    )
    return T


def radial_harmonics(n, l, r):
    if l < 0:
        raise KeyError("l must be at least 0 and at must n")
    if n < l or n == 0:
        raise KeyError("n > 0 and n >= l >= 0")
    rho = r / (BOHR_RADIUS * n)
    return rho**l * np.exp(-rho) * associated_laguerre(n - l - 1, 2 * l + 1, 2 * rho)


def radial_harmonics_natural(n, l, r):
    # r = r'/a to normalise the Bohr Radius to prevent underflow
    if l < 0:
        raise KeyError("l must be at least 0 and at must n")
    if n < l or n == 0:
        raise KeyError("n > 0 and n >= l >= 0")
    rho = r / n
    return rho**l * np.exp(-rho) * associated_laguerre(n - l - 1, 2 * l + 1, 2 * rho)


def spherical_normalisation(a, b, N, l, m):
    if N % 2 != 0:
        raise KeyError("N must be even")
    theta = np.linspace(a, b, num=int(N))
    p = np.abs(spherical_harmonics(l, m, theta)) ** 2 * np.sin(theta)
    h = (b - a) / N
    area = p[0] + p[-1]
    for i in range(1, len(p) - 1):
        if i % 2 == 0:
            area += 2 * p[i]
        else:
            area += 4 * p[i]
    return h / 3 * area


def radial_normalisation(n, l):
    """Integrate in units of a0 (FIX: raw-metre magnitudes underflow quad's
    default tolerance and silently return 0). Result is in units of a0^3."""

    def objective(x):
        return (radial_harmonics_natural(n, l, x) * x) ** 2

    return quad(objective, 0, np.inf, epsabs=1e-12, epsrel=1e-10)[0]

### hello
