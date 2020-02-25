import numpy as np
from matplotlib import pyplot as plt

from stingray import Lightcurve
from stingray import Powerspectrum
from stingray.utils import create_window

from astropy.modeling import models


# Amount of time the CCD is exposed in seconds
exposure_time = 2.9008230

# Cadence of the exposures in seconds
cadence = 12.

# Number of exposures
n_exposure = 234

# Times at which the exposures are made
times = cadence*np.arange(0, n_exposure)  # seconds

# Counts per second
signal = 300 * np.sin(2.*np.pi*times/0.5) + 1000  # counts/s

# Noisy version of the true signal
noisy = np.random.poisson(signal*exposure_time)  # counts


uniform = create_window(len(signal))
hanning = create_window(len(signal), window_type='hanning')

lc = []
for window_type in [uniform, hanning]:
    data = noisy*window_type
    lc.append(Lightcurve(times, data))

ps1 = Powerspectrum(lc[0])
ps2 = Powerspectrum(lc[1])


fig, ax1 = plt.subplots(1, 1, figsize=(9, 6), sharex=True)
ax1.plot(ps1.freq, ps1.power, lw=1, color='blue', label='no window')
ax1.plot(ps2.freq, ps2.power, lw=1, color='red', label='Hanning')
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Power (raw)")
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax1.tick_params(which='major', width=1.5, length=7)
ax1.tick_params(which='minor', width=1.5, length=4)

plt.show()


# Power law component
power_law = models.PowerLaw1D()

# fix x_0 of power law component
power_law.x_0.fixed = True

# Constant component
constant = models.Const1D()

# Power law plus constant
plc = power_law + constant

# Gaussian component
g = models.Gaussian1D()

# Power law plus constant plus Gaussian
plcg = plc + g

# parameters for fake data.
alpha = 2.0
amplitude = 5.0
white_noise = 2.0

freq = np.linspace(0.01, 10.0, 10.0/0.01)

from astropy.modeling.fitting import _fitter_to_model_params

_fitter_to_model_params(plc, [amplitude, alpha, white_noise])

psd_shape = plc(freq)

powers = psd_shape*np.random.chisquare(2, size=psd_shape.shape[0])/2.0

plt.figure(figsize=(12,7))
plt.loglog(freq, psd_shape, label="power spectrum")
plt.loglog(freq, powers, linestyle="steps-mid", label="periodogram realization")
plt.legend()
