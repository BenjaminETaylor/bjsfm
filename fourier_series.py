import numpy as np


def fourier_series_coefficients(f, T, N, return_complex=True):
    """
    Ref: https://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy

    Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    complex coefficients {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N : the function will return the first N + 1 Fourier coeff.

    Returns
    -------
    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N

    t = np.linspace(-T/2, T/2, f_sample, endpoint=False)

    y = np.fft.rfft(f(t)) / t.size

    # multiply odd terms by -1 to match SageMath
    y[1::2] *= -1

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:].real, -y[1:].imag


def x_dir_alpha_coefficients(N):
    """
    Calculates alpha coefficients for x-direction only loaded holes [used in Eq. 37.6, Ref. 2]
    :param N: <int> number of fourier series coefficients
    :return: fourier series coefficients
    """
    def brg_load_x_component(thetas):
        """
        Cosine load distribution [Fig. 10, Ref. 4]
        :param thetas: <np.array> angles
        """
        new_array = np.zeros(len(thetas))
        for i, theta in enumerate(thetas):
            # assume bearing load is only applied in x-dir
            if -np.pi / 2 <= theta <= np.pi / 2:
                new_array[i] = np.cos(theta) ** 2
        return new_array

    return fourier_series_coefficients(brg_load_x_component, 2 * np.pi, N, return_complex=False)[1]


def x_dir_beta_coefficients(N):
    """
    Calculates beta coefficients for x-direction only loaded holes [used in Eq. 37.6, Ref. 2]
    :param N: <int> number of fourier series coefficients
    :return: fourier series coefficients
    """
    def brg_load_y_component(thetas):
        """
        Cosine load distribution [Fig. 10, Ref. 4]
        :param thetas: <np.array> angles
        """
        new_array = np.zeros(len(thetas))
        for i, theta in enumerate(thetas):
            # assume bearing load is only applied in x-dir
            if -np.pi / 2 <= theta <= np.pi / 2:
                new_array[i] = np.cos(theta) * np.sin(theta)
        return new_array

    m = np.arange(1, N + 1)
    coefficients = fourier_series_coefficients(brg_load_y_component, 2 * np.pi, N, return_complex=False)[2]

    return coefficients/m**2


# alpha coefficients for N=45 [as chosen in Ref. 3]
x_dir_alphas = np.array([
    4.24413071e-01,  2.50000000e-01,  8.48829687e-02,  2.09853354e-17,
    -1.21266487e-02, -3.46703761e-18,  4.04281929e-03, -2.04095036e-17,
    -1.83831501e-03,  1.79351133e-17,  9.90587139e-04,  9.54820713e-18,
    -5.95128317e-04, -2.50426608e-17,  3.85909348e-04,  5.83059835e-18,
    -2.64921920e-04, -1.25285584e-17,  1.90165049e-04, -1.67449002e-17,
    -1.41554093e-04,  2.85843879e-17,  1.08648095e-04,  1.79447506e-17,
    -8.56499436e-05, -1.52270172e-17,  6.91652094e-05,  1.83687941e-17,
    -5.71178970e-05,  1.23358114e-17,  4.81921138e-05,  3.35379872e-18,
    -4.15292564e-05,  3.39234813e-18,  3.65564968e-05, -4.62592927e-18,
    -3.28860868e-05, -5.55111512e-18,  3.02545481e-05,  2.46716228e-18,
    -2.84852662e-05,  1.48029737e-17,  2.74654242e-05,  0.00000000e+00,
    -2.71322009e-05
])

# beta coefficients for N=45 [as chosen in Ref. 3]
x_dir_betas = np.array([
    2.12336003e-01,  6.25000000e-02,  1.41326777e-02, -1.89133046e-19,
    -1.20737928e-03,  1.31978190e-19,  2.86020692e-04, -2.40933816e-20,
    -1.00418267e-04, -1.27020308e-19,  4.38427890e-05,  1.78558728e-19,
    -2.20074675e-05, -2.83220159e-20,  1.21698860e-05, -2.55992180e-20,
    -7.22281343e-06,  5.35408480e-20,  4.52162358e-06,  2.48643698e-20,
    -2.94917680e-06, -7.80545917e-21,  1.98560450e-06,  2.59673113e-20,
    -1.36973879e-06, -1.84763447e-20,  9.61963771e-07, -8.85062998e-21,
    -6.83696174e-07,  4.62592927e-21,  4.88739304e-07,  3.61400724e-21,
    -3.48904592e-07,  9.87078333e-21,  2.46425590e-07,  9.51837298e-21,
    -1.69779959e-07,  4.27140283e-21,  1.11300545e-07,  9.25185854e-21,
    -6.57634238e-08, -5.59447228e-21,  2.95281910e-08,  1.01948854e-20,
    -0.00000000e+00
])


