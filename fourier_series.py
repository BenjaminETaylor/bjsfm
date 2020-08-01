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
    :param N: <positive int> number of fourier series coefficients
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
                # x-direction component of cosine load distribution
                # (in Ref. 2 Eq. 37.2, alpha is associated with the Y-direction. Can someone explain?)
                new_array[i] = np.cos(theta) ** 2
        return new_array

    coefficents = fourier_series_coefficients(brg_load_x_component, 2 * np.pi, N, return_complex=False)
    return 1/2*(coefficents[1] - 1j*coefficents[2])


def x_dir_beta_coefficients(N):
    """
    Calculates beta coefficients for x-direction only loaded holes [used in Eq. 37.6, Ref. 2]
    :param N: <positive int> number of fourier series coefficients
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
                # y-direction component of cosine load distribution
                # (in Ref. 2 Eq. 37.2, beta is associated with the X-direction. Can someone explain?)
                new_array[i] = np.cos(theta) * np.sin(theta)
        return new_array

    coefficents = fourier_series_coefficients(brg_load_y_component, 2 * np.pi, N, return_complex=False)
    return 1/2*(coefficents[1] - 1j*coefficents[2])


########################################################################
# The below hard-coded values were developed using the functions above #
########################################################################


# alpha coefficients for N=45 [as chosen in Ref. 3]
x_dir_alphas = np.array([
    2.12206536e-01+9.25185854e-18j,  1.25000000e-01+1.35693925e-17j,
    4.24414843e-02+1.99025668e-17j,  1.04926677e-17+5.08640204e-18j,
    -6.06332434e-03-7.14310928e-18j, -1.73351881e-18-1.22546160e-18j,
    2.02140964e-03-3.99806781e-18j, -1.02047518e-17+7.31476042e-18j,
    -9.19157506e-04+1.53462045e-17j,  8.96755664e-18+2.17179596e-17j,
    4.95293570e-04+9.56505919e-18j,  4.77410357e-18-1.05905351e-17j,
    -2.97564159e-04-1.73347303e-17j, -1.25213304e-17-2.62858793e-18j,
    1.92954674e-04+6.40402083e-18j,  2.91529917e-18+1.60630575e-17j,
    -1.32460960e-04+7.66456243e-18j, -6.26427922e-18+7.69542284e-20j,
    9.50825246e-05-6.80938133e-18j, -8.37245011e-18-1.23337711e-17j,
    -7.07770466e-05-6.22035980e-18j,  1.42921940e-17+8.73238269e-19j,
    5.43240477e-05+4.88346498e-19j,  8.97237531e-18-7.08772154e-18j,
    -4.28249718e-05-7.81867135e-18j, -7.61350859e-18-5.47460798e-18j,
    3.45826047e-05+5.84119911e-19j,  9.18439707e-18+6.56590423e-18j,
    -2.85589485e-05-1.87687443e-18j,  6.16790569e-18-6.40402083e-18j,
    2.40960569e-05-1.15575952e-17j,  1.67689936e-18-5.44223716e-18j,
    -2.07646282e-05-2.89029031e-18j,  1.69617407e-18-4.19238151e-19j,
    1.82782484e-05+8.58345259e-18j, -2.31296463e-18-7.01245368e-18j,
    -1.64430434e-05-1.16142048e-17j, -2.77555756e-18-1.15537523e-17j,
    1.51272740e-05-5.87909826e-18j,  1.23358114e-18+7.96825967e-18j,
    -1.42426331e-05+1.02255203e-17j,  7.40148683e-18+2.63243081e-18j,
    1.37327121e-05-6.16790569e-18j,  0.00000000e+00-3.08395285e-18j,
    -1.35661005e-05+0.00000000e+00j
])

# beta coefficients for N=45 [as chosen in Ref. 3]
x_dir_betas = np.array([
    1.04854397e-17-1.06168002e-01j,  2.03540888e-17-1.25000000e-01j,
    1.25837142e-17-6.35970496e-02j, -5.93907888e-18+1.51306437e-18j,
    -2.23586581e-18+1.50922409e-02j,  1.69858944e-17-2.37560743e-18j,
    9.85204277e-18-7.00750696e-03j, -8.02012964e-18+7.70988212e-19j,
    -1.04675222e-17+4.06693981e-03j,  5.64872348e-18+6.35101539e-18j,
    1.37217956e-17-2.65248874e-03j, -7.55588319e-18-1.28562284e-17j,
    -1.19060337e-17+1.85963101e-03j,  2.52211931e-18+2.77555756e-18j,
    -1.85037171e-18-1.36911217e-03j, -1.13499343e-17+3.27669990e-18j,
    -8.44805508e-18+1.04369654e-03j, -6.80807287e-18-8.67361738e-18j,
    -6.73941295e-18-8.16153056e-04j, -1.43127035e-17-4.97287396e-18j,
    -1.39247380e-17+6.50293485e-04j, -1.90686164e-18+1.88892112e-18j,
    4.58604568e-18-5.25192390e-04j,  7.58622114e-20-7.47858565e-18j,
    -1.80095012e-18+4.28043371e-04j,  5.71507253e-19+6.24500451e-18j,
    -2.65580742e-18-3.50635794e-04j, -4.04195395e-18+3.46944695e-18j,
    1.55838405e-18+2.87494241e-04j,  5.85951041e-18-2.08166817e-18j,
    6.88393687e-18-2.34839236e-04j,  9.59306907e-18-1.85037171e-18j,
    -1.19501186e-17+1.89978551e-04j, -2.00277475e-18-5.70531277e-18j,
    7.47735083e-18-1.50935674e-04j,  9.54233636e-18-6.16790569e-18j,
    8.08759111e-18+1.16214382e-04j,  6.33396908e-19-3.08395285e-18j,
    4.14400132e-18-8.46440648e-05j,  2.85265638e-18-7.40148683e-18j,
    2.89367545e-18+5.52741577e-05j, -2.47902802e-19+4.93432455e-18j,
    1.85037171e-18-2.72988126e-05j, -1.85037171e-18-9.86864911e-18j,
    -5.55111512e-18+0.00000000e+00j
])


