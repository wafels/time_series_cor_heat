import os
import numpy as np

import distributed

from stingray import Powerspectrum
from stingray.modeling import PSDLogLikelihood
from stingray.modeling import PSDParEst

import astropy.units as u
from astropy.time import Time
from astropy.modeling import models
from astropy.modeling.fitting import _fitter_to_model_params
from spectral_models import LogLorentz1D
from spectral_model_parameter_estimators import InitialParameterEstimatePlC

# Output
directory = os.path.expanduser('~/Data/ts/project_data/test_dask2_output')
if not os.path.exists(directory):
    os.makedirs(directory)

# Power law component
power_law = models.PowerLaw1D()
power_law.amplitude.min = 0.0
power_law.amplitude.max = None
power_law.alpha.min = 0.0
power_law.alpha.max = 4.0

# fix x_0 of power law component
power_law.x_0.fixed = True

# Constant component
constant = models.Const1D()
constant.amplitude.min = 0.0
constant.amplitude.max = None

# Lorentz component
#log_lorentz = LogLorentz1D()
#log_lorentz.name = 'log_lorentz'

# Create the model
observation_model = power_law + constant
observation_model.name = 'pl_c'

#observation_models.append(power_law + constant + log_lorentz)
#observation_models[1].name = 'pl_c_ll'

#observation_models.append(power_law + constant + log_normal)
#observation_models[2].name = 'pl_c_ln'


# Get some properties of the model
param_names = observation_model.param_names
fixed = observation_model.fixed
bounds = []
for param_name in param_names:
    if not fixed[param_name]:
        bounds.append((observation_model.bounds[param_name]))
scipy_optimize_options = {"bounds": bounds}

#############################
# Load in some power spectra
#############################
# Make some simulated data
# Based on the Bradshaw & Viall data
df = 0.000196078431372549 - 9.80392156862745e-05
freq = 9.80392156862745e-05 + df * np.arange(424)

# Size of the array
nx = 30
ny = 35

# Power law
alpha = np.linspace(2, 2, nx)
amplitude = 2923968935.189489
amplitude = 1.0

# Constant
white_noise = 200 / 2923968935.189489

# True parameters
true_parameters = [amplitude, alpha, white_noise]


# Create an array of simulated Fourier powers
def create_simulated_power_spectra(nx, ny, observation_model, model_parameters, frequencies):
    d = np.zeros([nx, ny, len(frequencies)])
    for i in range(0, nx):
        this_alpha = model_parameters[1][i]
        for j in range(0, ny):
            # This section will be replaced with a section that reads observed power spectra
            # Set the model parameters
            _fitter_to_model_params(observation_model,
                                    [model_parameters[0], this_alpha, model_parameters[2]])
            # Create the true data
            psd_shape = observation_model(frequencies)

            # Now randomize the true data and store it in an iterable
            d[i, j, :] = psd_shape * np.random.chisquare(2, size=psd_shape.shape[0]) / 2.0
    return d


# Create an array of data similar to that from loading in from an array
data = create_simulated_power_spectra(nx, ny, observation_model, true_parameters, freq)


# Create a list of powers as accepted by the dask processor
powers = []
for i in range(0, nx):
    for j in range(0, ny):
        powers.append((freq, data[i, j, :]))


def dask_fit_fourier_pl_c(power_spectrum):
    """
    Fits the power law + constant observation model

    Parameters
    ----------
    power_spectrum :

    Return
    ------

    """

    # Make the random data into a Powerspectrum object
    ps = Powerspectrum()
    ps.freq = power_spectrum[0]
    ps.power = power_spectrum[1]
    ps.df = ps.freq[1] - ps.freq[0]
    ps.m = 1

    # Define the log-likelihood of the data given the model
    loglike = PSDLogLikelihood(ps.freq, ps.power, observation_model, m=ps.m)
    # Parameter estimation object
    parameter_estimate = PSDParEst(ps, fitmethod="L-BFGS-B", max_post=False)

    # Estimate the starting parameters
    ipe = InitialParameterEstimatePlC(ps.freq, ps.power)
    return parameter_estimate.fit(loglike, [ipe.amplitude, ipe.index, ipe.background],
                                  scipy_optimize_options=scipy_optimize_options)


# TODO: Create plots of the output
# TODO: 3. Scatter plots of one value versus another
if __name__ == '__main__':

    # Use Dask to to fit the spectra
    client = distributed.Client()
    print('Dask processing of {:n} spectra'.format(nx*ny))

    # Name of the model we are considering
    observation_model_name = observation_model.name

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
    mfits = np.zeros_like(data)

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

            mfits[i, j, :] = r.mfit[:]

    filename = '{:s}.outputs.npz'.format(observation_model_name)
    filepath = os.path.join(directory, filename)
    print('Saving ' + filepath)
    np.savez(filepath, outputs)

    # Create a list the names of the output in the same order that they appear in the outputs
    output_names = list()
    for name in param_names:
        if not fixed[name]:
            output_names.append(name)
    for name in param_names:
        if not fixed[name]:
            output_names.append('err_{:s}'.format(name))
    output_names.append('aic')
    output_names.append('bic')
    output_names.append('result')

    filename = '{:s}.names.txt'.format(observation_model_name)
    filepath = os.path.join(directory, filename)
    print('Saving ' + filepath)
    with open(filepath, 'w') as file_out:
        for output_name in output_names:
            file_out.write(f"{output_name}\n")

    filename = '{:s}.mfits.npz'.format(observation_model_name)
    filepath = os.path.join(directory, filename)
    print('Saving ' + filepath)
    np.savez(filepath, freq, mfits)

    filename = '{:s}.data.npz'.format(observation_model_name)
    filepath = os.path.join(directory, filename)
    print('Saving ' + filepath)
    np.savez(filepath, freq, data)


""""
def dask_fit_fourier(powers):
    #Fits a model to the spectrum

    #    Parameters
    #----------
    #powers:


    #Returns
    #-------

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
"""