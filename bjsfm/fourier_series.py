import numpy as np


def fourier_series_coefficients(f, T, N, return_complex=True, sample_rate=1000):
    r"""Calculates the first 2*N+1 Fourier series coefficients of a periodic function.

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
    return_complex : bool, optional
        defaults to True
    sample_rate : int, optional
        used to tune fast Fourier transform (FFT) algorithm accuracy

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




