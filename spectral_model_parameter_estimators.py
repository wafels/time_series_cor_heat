import numpy as np

# Range of power law indices permitted.
power_law_index_bounds = (0, 4)


# Assume a power law in power spectrum - Use a Bayesian marginal distribution
# to calculate the probability that the power spectrum has a power law index
# 'n'.
def bayeslogprob(f, I, n, m):
    """
    Return the log of the marginalized Bayesian posterior of an observed
    Fourier power spectra fit with a model spectrum Af^{-n}, where A is a
    normalization constant, f is a normalized frequency, and n is the power law
    index at which the probability is calculated. The marginal probability is
    calculated using a prior p(A) ~ A^{m}.
    The function returns log[p(n)] give.  The most likely value of 'n' is the
    maximum value of p(n).
    f : normalized frequencies
    I : Fourier power spectrum
    n : power law index of the power spectrum
    m : power law index of the prior p(A) ~ A^{m}
    """
    N = len(f)
    term1 = n * np.sum(np.log(f))
    term2 = (N - m - 1) * np.log(np.sum(I * f ** n))
    return term1 - term2


# Find the most likely power law index given a prior on the amplitude of the
# power spectrum.
def most_probable_power_law_index(f, I, m, n):
    blp = np.zeros_like(n)
    for inn, nn in enumerate(n):
        blp[inn] = bayeslogprob(f, I, nn, m)
    return n[np.argmax(blp)]


def _frequency_range_indices(f, frequency_limits):
    return [np.argmin(np.abs(f-frequency_limits[0])),
            np.argmin(np.abs(f-frequency_limits[1]))]


def range_index_estimate(f, frequency_limits=None):
    nf = len(f)
    if frequency_limits is None:
        ii = [int(0.025*nf), int(0.975*nf)]
    else:
        ii = _frequency_range_indices(f, frequency_limits)
    return ii


def range_amplitude_estimate(f, frequency_limits=None):
    nf = len(f)
    if frequency_limits is None:
        ii = [0, int(0.025*nf)]
    else:
        ii = _frequency_range_indices(f, frequency_limits)
    return ii


def range_background_estimate(f, frequency_limits=None):
    nf = len(f)
    if frequency_limits is None:
        ii = [int(0.975 * nf), nf]
    else:
        ii = _frequency_range_indices(f, frequency_limits)
    return ii


class InitialParameterEstimatePlC(object):
    def __init__(self, f, p, ir=None, ar=None, br=None,
                 bayes_search=(0, np.linspace(power_law_index_bounds[0], power_law_index_bounds[1], 50))):
        """
        Estimate of three parameters of the power law + constant observation model - the amplitude,
        the power law index, and the background value.

        Parameters
        ----------
        f :
            Positive frequencies of the power law spectrum

        p :
            Power at the frequencies

        ir :


        ar :


        br :


        bayes_search : ~tuple


        """
        self.f = f
        self.p = p
        self.ir = ir
        self.ar = ar
        self.br = br
        self.bayes_search = bayes_search

        self._ir = range_index_estimate(self.f, frequency_limits=self.ir)
        self._ar = range_index_estimate(self.f, frequency_limits=self.ar)
        self._br = range_background_estimate(self.f, frequency_limits=self.br)

        # The bit in-between the low and high end of the spectrum to estimate the power law index.
        self._index = most_probable_power_law_index(self.f[self._ir[0]:self._ir[1]],
                                                    self.p[self._ir[0]:self._ir[1]],
                                                    bayes_search[0], bayes_search[1])

        # Use the low-frequency end to estimate the amplitude, normalizing for the first frequency
        self._amplitude = np.mean(self.p[self._ar[0]:self._ar[1]]) * (self.f[0] ** self._index)

        # Use the high frequency part of the spectrum to estimate the constant value.
        self._background = np.exp(np.mean(np.log(self.p[self._br[0]:self._br[1]])))

    @property
    def index_range(self):
        return self._ir

    @property
    def amplitude_range(self):
        return self._ar

    @property
    def background_range(self):
        return self._br

    @property
    def index(self):
        return self._index

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def background(self):
        return self._background
