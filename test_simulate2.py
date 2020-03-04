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

# Parameter estimation
parest = PSDParEst(ps, fitmethod="L-BFGS-B", max_post=False)
starting_pars = true_parameters
res = parest.fit(loglike, starting_pars)

print(true_parameters)
print(res.p_opt)
print(res.err)
print("AIC: " + str(res.aic))
print("BIC: " + str(res.bic))

res.print_summary(loglike)

plt.figure(figsize=(12, 8))
plt.loglog(ps.freq, psd_shape, label="true power spectrum", lw=3)
plt.loglog(ps.freq, ps.power, label="simulated data")
plt.loglog(ps.freq, res.mfit, label="best fit", lw=3)
plt.legend()
plt.show()
