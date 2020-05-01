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
ny = 100

# Power law
alpha = np.linspace(0, 5, nx)
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
def create_simulated_data(nx, ny, observation_model, model_parameters, frequencies):
    p = []
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
            p.append(psd_shape * np.random.chisquare(2, size=psd_shape.shape[0]) / 2.0)
    return p


powers = create_simulated_data(nx, ny, observation_models[0], true_parameters, freq)


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
    ipe = InitialParameterEstimatePlC(ps.freq, ps.power)

    return parameter_estimate.fit(loglike, [ipe.amplitude, ipe.index, ipe.background])


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
    with open(filepath, 'w') as file_out:
        for output_name in output_names:
            file_out.write(f"{output_name}\n")


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