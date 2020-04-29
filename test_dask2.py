import os
import numpy as np
from matplotlib import pyplot as plt

from stingray import Powerspectrum
from stingray.modeling import PSDLogLikelihood
from stingray.modeling import PSDParEst

import astropy.units as u
from astropy.time import Time
from astropy.modeling import models
from astropy.modeling.fitting import _fitter_to_model_params
from spectral_models import LogLorentz1D

import distributed

# Output
directory = os.path.expanduser('~/Data/ts/project_data/test_dask2_output')
if not os.path.exists(directory):
    os.makedirs(directory)

# Power law component
power_law = models.PowerLaw1D()

# fix x_0 of power law component
power_law.x_0.fixed = True

# Constant component
constant = models.Const1D()

# Lorentz component
log_lorentz = LogLorentz1D()
log_lorentz.name = 'log_lorentz'

# All the models
observation_models = list()

observation_models.append(power_law + constant)
observation_models[0].name = 'pl_c'

observation_models.append(power_law + constant + log_lorentz)
observation_models[1].name = 'pl_c_ll'

#observation_models.append(power_law + constant + log_normal)
#observation_models[2].name = 'pl_c_ln'


#############################
# Load in some power spectra


#############################
# Make some simulated data
freq = np.linspace(0.01, 10.0, int(10.0/0.01))

# Size of the array
nx = 10
ny = 11

# Power law
alpha = 2.0
amplitude = 5.0

# Constant
white_noise = 2.0

# Log Lorentz
ll_amplitude = 1000.0
ll_log_x_0 = np.log(1.0)
ll_fwhm = np.log(1.01)

# True parameters
true_parameters = [amplitude, alpha, white_noise]


# Create a list of simulated Fourier powers
# TODO: add the option to create a known range of power law indices, the most important parameter for this study.
def create_simulated_data(nx, ny, observation_model, true_parameters, freq):
    powers = []
    for i in range(0, nx):
        for j in range(0, ny):
            # This section will be replaced with a section that reads observed power spectra
            # Set the model parameters
            _fitter_to_model_params(observation_model, true_parameters)

            # Create the true data
            psd_shape = observation_model(freq)

            # Now randomize the true data and store it in an iterable
            powers.append(psd_shape * np.random.chisquare(2, size=psd_shape.shape[0]) / 2.0)
    return powers


powers = create_simulated_data(nx, ny, observation_models[0], true_parameters, freq)


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


def initial_parameter_estimate_pl_c(f, p):
    """
    Estimate of the power law + constant observation model

    Parameters
    ----------
    f :


    p :

    Return
    ------

    """
    nf = len(f)

    # Use the low frequency part of the spectrum to estimate the amplitude of the power spectrum.
    ar = [0, int(0.025*nf)]
    amplitude_estimate = np.exp(np.mean(np.log(p[ar[0]:ar[1]])))

    # Use the high frequency part of the spectrum to estimate the constant value.
    br = [int(0.975 * nf), nf - 1]
    background_estimate = np.exp(np.mean(np.log(p[br[0]:br[1]])))

    # The bit in-between the low and high end of the spectrum to estimate the power law index.
    ir = [ar[1], br[0]]
    index_estimate = most_probable_power_law_index(f[ir[0]:ir[1]],
                                                   p[ir[0]:ir[1]],
                                                   0.0, np.arange(0.0, 4.0, 0.01))

    return [amplitude_estimate, index_estimate, background_estimate]


def dask_fit_fourier_pl_c(powers):
    """
    Fits the power law + constant observation model

    Parameters
    ----------
    powers :

    Return
    ------

    """

    # Make the random data into a Powerspectrum object
    ps = Powerspectrum()
    ps.freq = freq
    ps.power = powers
    ps.df = ps.freq[1] - ps.freq[0]
    ps.m = 1

    # Define the log-likelihood of the data given the model
    loglike = PSDLogLikelihood(ps.freq, ps.power, observation_models[0], m=ps.m)

    # Parameter estimation object
    parameter_estimate = PSDParEst(ps, fitmethod="L-BFGS-B", max_post=False)

    # Estimate the starting parameters - will be replaced with a function
    # starting_parameters = starting_parameter_selector(model_name)
    initial_parameter_estimate = initial_parameter_estimate_pl_c(ps.freq, ps.power)

    return parameter_estimate.fit(loglike, initial_parameter_estimate)


def dask_fit_fourier(powers):
    """
    Fits a model to the spectrum

    Parameters
    ----------
    powers:


    Returns
    -------

    """

    # Make the random data into a Powerspectrum object
    ps = Powerspectrum()
    ps.freq = freq
    ps.power = powers
    ps.df = ps.freq[1] - ps.freq[0]
    ps.m = 1

    # Define the log-likelihood of the data given the model
    loglike = PSDLogLikelihood(ps.freq, ps.power, observation_model, m=ps.m)

    # Parameter estimation object
    parameter_estimate = PSDParEst(ps, fitmethod="L-BFGS-B", max_post=False)

    # Estimate the starting parameters - will be replaced with a function
    # starting_parameters = starting_parameter_selector(model_name)
    starting_pars = [amplitude, alpha, white_noise, ll_amplitude, ll_log_x_0, ll_fwhm]

    return parameter_estimate.fit(loglike, starting_pars)


# TODO: Create plots of the output
# TODO: 1. Histograms of values
# TODO: 2. Spatial maps of values
# TODO: 3. Scatter plots of one value versus another
if __name__ == '__main__':

    # Use Dask to to fit the spectra
    client = distributed.Client()
    print('Dask processing of {:n} spectra'.format(nx*ny))

    #
    observation_model_name = observation_models[0].name

    # Get the start time
    t_start = Time.now()

    # Do the model fits
    results = client.map(dask_fit_fourier_pl_c, powers)
    z = client.gather(results)

    # Get the end time and calculate the time taken
    t_end = Time.now()
    print('Time taken to ', (t_end-t_start).to(u.s))

    # Now go through all the results and save out the results
    # Total number of outputs = 2*n_parameters + 3
    # For short hand call n_parameters 'n' instead
    # 0 : n-1 : parameter values
    # n : 2n-1 : error in parameter values
    # 2n + 0 : AIC
    # 2n + 1 : BIC
    # 2n + 2 : result
    n_parameters = len(z[0].p_opt)
    n_outputs = 2 * n_parameters + 3
    outputs = np.zeros((nx, ny, n_outputs))

    # Turn the results into an easier to use array
    for i in range(0, nx):
        for j in range(0, ny):
            index = j + i*ny
            r = z[index]
            outputs[i, j, 0:n_parameters] = r.p_opt[:]
            outputs[i, j, n_parameters:2*n_parameters] = r.err[:]
            outputs[i, j, 2 * n_parameters + 0] = r.aic
            outputs[i, j, 2 * n_parameters + 1] = r.bic
            outputs[i, j, 2 * n_parameters + 2] = r.result

    filename = '{:s}.outputs.npz'.format(observation_model_name)
    filepath = os.path.join(directory, filename)
    print('Saving ' + filepath)
    np.savez(filepath, outputs)

    # Create a list the names of the output in the same order that they appear in the outputs
    output_names = list()
    param_names = observation_models[0].param_names
    fixed = observation_models[0].fixed
    for name in param_names:
        if not fixed[name]:
            output_names.append(name)
    for name in param_names:
        if not fixed[name]:
            output_names.append('err_{:s}'.format(name))
    output_names.append('aic')
    output_names.append('bic')
    output_names.append('result')

    filename = '{:s}.output_names.txt'.format(observation_model_name)
    filepath = os.path.join(directory, filename)
    print('Saving ' + filepath)
    with open(filepath, 'w') as f:
        for output_name in output_names:
            f.write(f"{output_name}\n")
