import numpy as np


def fourier_series_coefficients(f, T, N, return_complex=True, sample_rate=1000):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    complex coefficients {c0,c1,c2,...}
    such that:

    .. math:: f(t) ~= \sum_{k=-N}^N c_k \cdot e^{i2 \pi kt/T}

    where we define :math: `c_{-n}=\overline{c_n}`

    Refer to `wikipedia <http://en.wikipedia.org/wiki/Fourier_series>`_ for the relation between the real-valued and
    complex valued coeffs.

    Notes
    -----
    This function was copied from
    `stackoverflow <https://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy>`_.

    Parameters
    ----------
    f : callable
        the periodic function, a callable like f(t)
    T : float
        the period of the function f, so that f(0)==f(T)
    N : int
        the function will return the first N+1 Fourier coeff.

    Returns
    -------
    ndarray
        numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theorem we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal. (Minimum 1000*N required for sufficient accuracy.)
    f_sample = sample_rate * N

    t = np.linspace(-T/2, T/2, f_sample, endpoint=False)

    y = np.fft.rfft(f(t)) / t.size

    # multiply odd terms by -1 to match SageMath
    y[1::2] *= -1

    # only take the number of coefficients requested
    y = y[:N + 1]

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:].real, -y[1:].imag


def x_dir_alpha_coefficients(N):
    """ Calculates alpha coefficients for x-direction loaded holes

    This function calculates the fourier series coefficients used in Eq. 37.6 [2]_.

    Parameters
    __________
    N : <positive int>
        number of fourier series coefficients

    Returns
    -------
    ndarray
        fourier series coefficients
    """
    def brg_load_x_component(thetas):
        """ Cosine load distribution Fig. 10 [4]_

        Parameters
        ----------
        thetas : ndarray
            angles
        """
        new_array = np.zeros(len(thetas))
        for i, theta in enumerate(thetas):
            # assume bearing load is only applied in x-dir
            if -np.pi / 2 <= theta <= np.pi / 2:
                # x-direction component of cosine load distribution
                # (in Ref. 2 Eq. 37.2, alpha is associated with the Y-direction. Can someone explain?)
                new_array[i] = np.cos(theta)**2
        return new_array

    # return all coefficients except the first one (Ao)
    return fourier_series_coefficients(brg_load_x_component, 2 * np.pi, N, sample_rate=100000)[1:]


def x_dir_beta_coefficients(N):
    """ Calculates beta coefficients for x-direction loaded holes

    This function calculates the fourier series coefficients using in Eq. 37.6 [2]_.

    Parameters
    ----------
    N : <positive int>
        number of fourier series coefficients

    Returns
    -------
    ndarray
        fourier series coefficients
    """
    def brg_load_y_component(thetas):
        """ Cosine load distribution Fig. 10 [4]_

        Parameters
        ----------
        thetas : ndarray
            angles
        """
        new_array = np.zeros(len(thetas))
        for i, theta in enumerate(thetas):
            # assume bearing load is only applied in x-dir
            if -np.pi / 2 <= theta <= np.pi / 2:
                # y-direction component of cosine load distribution
                # (in Ref. 2 Eq. 37.2, beta is associated with the X-direction. Can someone explain?)
                new_array[i] = np.cos(theta) * np.sin(theta)
        return new_array

    # return all coefficients except the first one (Ao)
    return fourier_series_coefficients(brg_load_y_component, 2 * np.pi, N, sample_rate=100000)[1:]


########################################################################
# The below hard-coded values were developed using the functions above #
########################################################################


# alpha coefficients for N=45 as chosen in [3]_ at sample_rate=100000
# x_dir_alphas = x_dir_alpha_coefficients(45)
x_dir_alphas = np.array([
    2.12206591e-01-4.77083644e-17j,  1.25000000e-01-5.89573465e-17j,
    4.24413182e-02-1.91840853e-17j, -8.90314393e-18-1.79348322e-19j,
    -6.06304545e-03+6.55633890e-18j,  5.48463980e-18+4.37501201e-18j,
    2.02101515e-03-3.66997376e-18j, -2.47147905e-18-3.77237815e-19j,
    -9.18643250e-04+6.67550845e-19j,  1.15294597e-18+4.32409913e-20j,
    4.94654058e-04-5.26048781e-18j, -1.92490138e-18-3.55274303e-18j,
    -2.96792435e-04+4.00276461e-18j,  3.49945789e-18+2.84432075e-18j,
    1.92042164e-04-7.15349518e-19j, -2.10847715e-18+5.86429928e-19j,
    -1.31397270e-04+5.42357122e-19j,  5.26279974e-19+5.07907945e-19j,
    9.38551927e-05-1.60287068e-18j, -2.62667554e-19-2.81642867e-20j,
    -6.93712294e-05+4.72318710e-19j, -1.55309233e-19-6.73163746e-19j,
    5.27221344e-05+3.74419334e-19j,  1.10507308e-18-3.45051024e-18j,
    -4.10061045e-05+1.56923065e-19j,  9.40356979e-19-2.19017030e-18j,
    3.25220829e-05-3.91078386e-19j,  1.36872347e-19-4.27353360e-19j,
    -2.62274862e-05+2.86611820e-19j,  9.78311008e-20-7.89061684e-20j,
    2.14588523e-05-8.91027872e-19j, -1.30904740e-19+1.91919825e-19j,
    -1.77801919e-05+1.97944104e-19j,  8.14254172e-19+2.81801032e-19j,
    1.48969176e-05-1.66624951e-19j, -1.34123974e-18+1.17525380e-18j,
    -1.26050841e-05+1.21462369e-18j,  5.21951371e-19-1.06955735e-18j,
    1.07604376e-05-1.17456794e-18j, -8.16624019e-20+5.13214752e-20j,
    -9.25898123e-06-1.65297614e-19j,  3.30062278e-19-2.46250926e-20j,
    8.02445040e-06-2.73275116e-19j, -2.39245061e-19+5.01995076e-19j,
    -7.00005248e-06+1.01720924e-19j
])


# beta coefficients for N=45 as chosen in [3]_ at sample_rate=100000
# x_dir_betas = x_dir_beta_coefficients(45)
x_dir_betas = np.array([
    -1.94319243e-17-1.06103295e-01j, -5.45839291e-17-1.25000000e-01j,
    -3.62876318e-17-6.36619772e-02j,  1.30591839e-18+1.52792630e-17j,
    1.58336660e-17+1.51576136e-02j,  1.61007420e-18-1.20107231e-17j,
    -9.15844587e-18-7.07355303e-03j, -4.65834606e-19+4.69348027e-18j,
    7.82631893e-18+4.13389463e-03j, -2.07168349e-19-5.48019331e-18j,
    -7.79806861e-18-2.72059732e-03j, -8.28820898e-19+3.72983658e-18j,
    5.67464898e-18+1.92915083e-03j, -9.41779078e-19-2.96224847e-18j,
    -4.81136247e-18-1.44031623e-03j, -4.18882423e-20+3.92096760e-18j,
    3.53379639e-18+1.11687679e-03j,  1.18208219e-18-3.45316542e-18j,
    -3.35800239e-18-8.91624331e-04j, -3.88844853e-19+2.81568924e-18j,
    3.55287198e-18+7.28397909e-04j, -7.24302864e-22-3.24725934e-18j,
    -2.86484044e-18-6.06304545e-04j,  1.85812997e-18+2.72227446e-18j,
    2.71489222e-18+5.12576306e-04j, -1.22325211e-18-2.62305288e-18j,
    -3.25375118e-18-4.39048119e-04j,  5.06148684e-20+1.30612327e-18j,
    2.02547194e-18+3.80298550e-04j, -1.10424267e-19-1.61508137e-18j,
    -2.30407373e-18-3.32612211e-04j, -4.65115570e-19+1.28879601e-18j,
    2.22873521e-18+2.93373167e-04j,  8.28830477e-20-1.39232809e-18j,
    -1.82653809e-18-2.60696058e-04j,  3.63246046e-19+1.92275788e-18j,
    1.97581297e-18+2.33194056e-04j,  2.19814138e-20-1.77673402e-18j,
    -1.35481930e-18-2.09828534e-04j,  9.33755027e-20+1.34376519e-18j,
    1.71339592e-18+1.89809115e-04j,  1.30928047e-19-1.79294538e-18j,
    -1.94173495e-18-1.72525684e-04j, -1.07013407e-19+9.92738558e-19j,
    1.57354012e-18+1.57501181e-04j
])


