"""Rejection sampler for the hydrogen atom using scipy.special.

Self-contained alternative to visualise.py: the spherical harmonics and the radial
Laguerre polynomials come from scipy.special instead of equations.py, so there is no
restriction to l <= 3 and no finite-difference derivatives.

The density factorises into independent parts (lengths in units of the Bohr radius),

    p(r)     = r^2 R_nl(r)^2
    p(theta) = 2 pi |Y_lm(theta, 0)|^2 sin(theta)
    p(phi)   = 1 / (2 pi),

and both p(r) and p(theta) are normalised analytically, so no numerical
normalisation is needed.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, gammaln, sph_harm_y

R_MIN = 0.0

THETA_MIN = 0.0
THETA_MAX = np.pi

PHI_MIN = 0.0
PHI_MAX = 2 * np.pi

GRID_POINTS = 200_000  # grid used to locate the maximum of each density
ENVELOPE_MARGIN = 1.01  # safety factor so the envelope always lies above the pdf


def cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def radial_pdf(n, l):
    """Normalised p(r) = r^2 R_nl(r)^2, with r in units of a0.

    R_nl = sqrt[(2/n)^3 (n-l-1)! / (2n (n+l)!)] e^{-rho} (2 rho)^l L^{2l+1}_{n-l-1}(2 rho),
    rho = r / n. The prefactor is built from log-gamma functions to avoid overflow.
    """
    log_prefactor = 0.5 * (
        3 * np.log(2 / n) + gammaln(n - l) - np.log(2 * n) - gammaln(n + l + 1)
    )

    def pdf(r):
        rho = r / n
        radial = (
            np.exp(log_prefactor - rho)
            * (2 * rho) ** l
            * eval_genlaguerre(n - l - 1, 2 * l + 1, 2 * rho)
        )
        return (r * radial) ** 2

    return pdf


def polar_pdf(l, m):
    """Normalised p(theta) = 2 pi |Y_lm(theta, 0)|^2 sin(theta).

    |Y_lm| does not depend on phi, so phi = 0 is used.
    """

    def pdf(theta):
        return 2 * np.pi * np.abs(sph_harm_y(l, m, theta, 0.0)) ** 2 * np.sin(theta)

    return pdf


def outer_turning_point(n, l):
    """Outer classical turning point in units of a0."""
    return n**2 * (1 + np.sqrt(1 - l * (l + 1) / n**2))


def radial_cutoff(pdf, r_plus, tail=1e-10):
    """Smallest r_max (grown from r_plus) with a radial tail mass below `tail`."""
    r_max = r_plus
    while quad(pdf, r_max, np.inf, epsabs=1e-14, epsrel=1e-8)[0] > tail:
        r_max *= 1.25
    return r_max


def rejection_sample(pdf, lo, hi, peak, size, rng):
    """Draw `size` independent samples from a normalised 1D pdf on [lo, hi].

    A candidate x is uniform on [lo, hi] and is accepted if u <= pdf(x), where u is
    uniform on [0, peak] and peak >= max(pdf). Candidates are generated in batches.
    """
    acceptance = 1.0 / ((hi - lo) * peak)  # expected acceptance for a normalised pdf
    chunks = []
    have = 0
    while have < size:
        batch = int(min(max(1.2 * (size - have) / acceptance, 1_000), 5_000_000))
        x = rng.uniform(lo, hi, batch)
        u = rng.uniform(0.0, peak, batch)
        accepted = x[u <= pdf(x)]
        chunks.append(accepted)
        have += accepted.size
    return np.concatenate(chunks)[:size]


def generate(npts: int, n, l, m, seed=None, tail=1e-10):
    """Sample npts points (r, theta, phi) from |psi_nlm|^2.

    Returns (coords, density): coords has shape (npts, 3) with r in units of a0, and
    density is p(r) * p(theta) at each point, used to colour the plot.
    """
    if n < 1 or not 0 <= l < n:
        raise ValueError("require n >= 1 and 0 <= l < n")
    if abs(m) > l:
        raise ValueError("require |m| <= l")

    rng = np.random.default_rng(seed)

    f_r = radial_pdf(n, l)
    f_theta = polar_pdf(l, m)

    r_max = radial_cutoff(f_r, outer_turning_point(n, l), tail)

    r_grid = np.linspace(R_MIN, r_max, GRID_POINTS)
    theta_grid = np.linspace(THETA_MIN, THETA_MAX, GRID_POINTS)
    peak_r = ENVELOPE_MARGIN * np.max(f_r(r_grid))
    peak_theta = ENVELOPE_MARGIN * np.max(f_theta(theta_grid))

    # r, theta and phi are independent, so each is sampled separately.
    r = rejection_sample(f_r, R_MIN, r_max, peak_r, npts, rng)
    theta = rejection_sample(f_theta, THETA_MIN, THETA_MAX, peak_theta, npts, rng)
    phi = rng.uniform(PHI_MIN, PHI_MAX, npts)  # p(phi) is uniform: no rejection needed

    density = f_r(r) * f_theta(theta)
    return np.column_stack([r, theta, phi]), density


if __name__ == "__main__":
    coord, psi_2 = generate(30000, 4, 3, 1)
    x, y, z = cartesian(coord[:, 0], coord[:, 1], coord[:, 2])
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=2,
                    color=psi_2,
                    colorscale="Blackbody",
                    opacity=0.6,
                ),
            )
        ]
    )

    fig.show()
