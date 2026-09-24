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

    r_xs = np.linspace(R_MIN, r_max, 10e6)
    s_xs = np.linspace(THETA_MIN, THETA_MAX, 10e6)
    p_xs = np.linspace(PHI_MIN, PHI_MAX, 10e6)

    r_fs = f_r(r_xs)
    s_fs = f_theta(s_xs)
    p_fs = f_phi(p_xs)

    peak_r = max(r_fs)
    peak_s = max(s_fs)
    peak_p = max(p_fs)

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
