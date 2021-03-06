import numpy as np
from matplotlib import pyplot as plt

from stingray import Powerspectrum
from stingray.modeling import PSDLogLikelihood
from stingray.modeling import PSDParEst

from astropy.modeling import models
from astropy.modeling.fitting import _fitter_to_model_params
from spectral_models import LogLorentz1D

# Power law component
power_law = models.PowerLaw1D()

# fix x_0 of power law component
power_law.x_0.fixed = True

# Constant component
constant = models.Const1D()

# Gaussian component
log_lorentz = LogLorentz1D()

# Power law plus constant plus Gaussian
observation_model = log_lorentz


# parameters for fake data.

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
true_parameters = [amplitude, alpha, white_noise, ll_amplitude, ll_log_x_0, ll_fwhm]
true_parameters = [ll_amplitude, ll_log_x_0, ll_fwhm]



# Frequency range
freq = np.linspace(0.01, 10.0, int(10.0/0.01))

# Set the model parameters and create some fake true data
_fitter_to_model_params(observation_model, true_parameters)
psd_shape = observation_model(freq)

# Now randomize the true data
powers = psd_shape*np.random.chisquare(2, size=psd_shape.shape[0])/2.0

# Make the random data into a Powerspectrum object

ps = Powerspectrum()
ps.freq = freq
ps.power = powers
ps.df = ps.freq[1] - ps.freq[0]
ps.m = 1

# Define the log-likelihood of the data given the model
loglike = PSDLogLikelihood(ps.freq, ps.power, observation_model, m=ps.m)
loglike(true_parameters)

res.print_summary(loglike)

plt.figure(figsize=(12, 8))
plt.loglog(ps.freq, psd_shape, label="true power spectrum", lw=3)
plt.loglog(ps.freq, psd_shape_start, label="initial estimate power spectrum", lw=3)
plt.loglog(ps.freq, ps.power, label="simulated data")
plt.loglog(ps.freq, res.mfit, label="best fit", lw=3)
plt.legend()
plt.show()




class SaveParameters:
    """
    Object for the storage of the fitting results
    """
    def __init__(self, nx, ny, model_name, parameter_names):
        self._nx = nx
        self._ny = ny
        self._model_name = model_name
        self._np = len(parameter_names)
        self._parameter_names = parameter_names
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

    def _filename(self):
        return self._model_name

    def _save_basic_information(self, file_path):
        f = os.path.join(file_path, self._filename)

    def save(self, file_path, replace=False):
        self._save_basic_information(self, file_path)


nx = 10
ny = 11
model_name = 'pl_c_ll'
sp = SaveParameters(nx, ny, model_name, ['amplitude', 'alpha', 'white_noise', 'll_amplitude', 'll_log_x_0', 'll_fwh'])
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
        starting_pars = [ll_amplitude/2.0, -1.0, ll_fwhm/5.0]

        # Do the fit and store the results
        sp.store(i, j, parameter_estimate.fit(loglike, starting_pars))

# Save the results
sp.save('/Users/Desktop/')

