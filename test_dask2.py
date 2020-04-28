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
nx = 50
ny = 51

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


# TODO: complete this initial parameter estimator
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
    return [amplitude, alpha, white_noise]


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
    :param spectrum:
    :return:
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
    # 2n : AIC
    # 2n + 1 : BIC
    # 2n + 2 : result
    # TODO: put all the output into the same three dimensional array
    n_parameters = len(z[0].p_opt)
    p_opt = np.zeros((nx, ny, n_parameters))
    err = np.zeros_like(p_opt)
    aic = np.zeros((nx, ny))
    bic = np.zeros_like(aic)
    result = np.zeros_like(aic)

    # Turn the results in to easier to use arrays
    for i in range(0, nx):
        for j in range(0, ny):
            index = j + i*ny
            r = z[index]
            p_opt[i, j, :] = r.p_opt[:]
            err[i, j, :] = r.err[:]
            aic[i, j] = r.aic
            bic[i, j] = r.bic
            result[i, j] = r.result

    # TODO: write out all the output names in the same order as they are stored in the array
    # If they are referred via a dictionary it is easier to automate their saving
    output = {'p_opt': p_opt, 'err': err, 'aic': aic, 'bic': bic, 'result': result}

    # Save the arrays - to be analyzed by other programs
    for output_type in list(output.keys()):
        filename = '{:s}.{:s}.npz'.format(observation_models[0].name, output_type)
        filepath = os.path.join(directory, filename)
        print('Saving ' + filepath)
        np.savez(filepath, output[output_type])


"""

class SaveParameters:
    #Object for the storage of the fitting results
    def __init__(self, nx, ny, observation_model, parameter_names=None):
        self._nx = nx
        self._ny = ny
        self._observation_model = observation_model
        self._model_name = self._observation_model.name
        if parameter_names is None:
            self._parameter_names = self._observation_model.param_names
        else:
            self._parameter_names = parameter_names
        self._np = len(self._parameter_names)
        self.p_opt = np.zeros((self._nx, self._ny, self._np))
        self.err = np.zeros_like(self.p_opt)
        self.aic = np.zeros((self._nx, self._ny))
        self.bic = np.zeros_like(self.aic)
        self.result = np.zeros_like(self.aic)

    def store(self, i, j, res):
        self.p_opt[i, j, :] = res.p_opt[:]
        self.err[i, j, :] = res.err[:]
        self.aic[i, j] = res.aic
        self.bic[i, j] = res.bic
        self.result[i, j] = res.result

    @property
    def observation_model(self):
        return self._observation_model

    @property
    def model_name(self):
        return self._model_name

    @property
    def parameter_names(self):
        return self._parameter_names

    def _filename(self):
        return self.model_name

    def _save_basic_information(self, file_path):
        f = os.path.join(file_path, self._filename)

    def save(self, file_path, replace=False):
        self._save_basic_information(self, file_path)

model_name = 'pl_c_ll'
sp = SaveParameters(nx, ny, observation_model, parameter_names=['amplitude', 'alpha', 'white_noise', 'll_amplitude', 'll_log_x_0', 'll_fwhm'])
for i in range(0, nx):
    for j in range(0, ny):
        # This section will be replaced with a section that reads observed power spectra
        # Set the model parameters
        _fitter_to_model_params(observation_model, true_parameters)

        # Create the true data
        psd_shape = observation_model(freq)

        # Now randomize the true data
        powers = psd_shape * np.random.chisquare(2, size=psd_shape.shape[0]) / 2.0

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

        # Do the fit and store the results
        sp.store(i, j, parameter_estimate.fit(loglike, starting_pars))

# Save the results
#sp.save('/Users/Desktop/')
"""
