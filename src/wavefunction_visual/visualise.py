# generate r according to radial_harmonics_natural
# generate theta according to spherical_harmonics_natural
# generate phi uniformly from 0 to 2pi

import numpy as np
import wavefunction_visual.equations as eq
import random
import plotly.graph_objects as go

R_MIN = 0

THETA_MIN = 0
THETA_MAX = np.pi

PHI_MIN = 0
PHI_MAX = 2 * np.pi


def cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.cos(theta)
    return x, y, z


# can implement logic determining the skin depth of radial probability decay
# for now choose 500 * classical region
def generate(npts: int, n, l, m):
    r_norm = eq.radial_normalisation(n, l)
    s_norm = eq.spherical_normalisation(THETA_MIN, THETA_MAX, 10e6, l, m)
    sample = []

    classical_region = n**2 * (1 + np.sqrt(1 - l * (l + 1) / (n**2)))
    r_max = 500 * classical_region

    def f_r(x):
        return (x * eq.radial_harmonics_natural(n, l, x)) ** 2 / r_norm

    def f_theta(x):
        return eq.spherical_harmonics(l, m, x) ** 2 * np.sin(x) / s_norm

    def f_phi(x):
        return 1 / PHI_MAX

    r_xs = np.linspace(R_MIN, r_max, num=int(10e6))
    s_xs = np.linspace(THETA_MIN, THETA_MAX, num=int(10e6))

    r_fs = f_r(r_xs)
    s_fs = f_theta(s_xs)

    peak_r = max(r_fs)
    peak_s = max(s_fs)
    peak_p = 1 / PHI_MAX

    while len(sample) < npts:
        const = random.random()
        random_r = r_max * random.random()
        random_theta = THETA_MAX * random.random()
        random_phi = PHI_MAX * random.random()

        if (
            const * peak_r <= f_r(random_r)
            and const * peak_s <= f_theta(random_theta)
            and const * peak_p <= f_phi(random_phi)
        ):
            sample.append([random_r, random_theta, random_phi])
    return np.array(sample)


coord = generate(10000, 3, 1, 0)
x, y, z = cartesian(coord[:, 0], coord[:, 1], coord[:, 2])
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=3,
            ),
        )
    ]
)

fig.show()
