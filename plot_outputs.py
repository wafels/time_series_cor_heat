import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
rc('text', usetex=True)  # Use LaTeX

# Which model to look at
observation_model_name = 'pl_c'

# Number of equally spaced bins in the histogram
bins = 50

# Colour for bad fits in the spatial distribution
bad_color = 'red'

# Location of the data we are analyzing
directory = os.path.expanduser('~/Data/ts/project_data/test_dask2_output')

# Load in some information about how to treat and plot the outputs, for example
# output_name,lower_bound,upper_bound,variable_name
# "amplitude_0",None,None,"A_{0}"
# "alpha_0",0,4,"n"
filename = '{:s}.outputs_information.csv'.format(observation_model_name)
filepath = os.path.join(os.path.expanduser('~/time_series_cor_heat-git/time_series_cor_heat'), filename)
df = pd.read_csv(filepath, index_col=0)
df = df.replace({"None": None})

# Load in the data and the output names 
filename = '{:s}.outputs.npz'.format(observation_model_name)
filepath = os.path.join(directory, filename)
outputs = np.load(filepath)['arr_0']

filename = '{:s}.output_names.txt'.format(observation_model_name)
filepath = os.path.join(directory, filename)
with open(filepath) as f:
    output_names = [line.rstrip() for line in f]


class SummaryStatistics:
    """Some simple descriptive statistics of a 1-D input array"""
    def __init__(self, d, ci=(0.16, 0.84), **histogram_kwargs):
        # Only work with unmasked data.
        if np.ma.is_masked(d):
            raise ValueError('Unmasked data only')
        self.data = d
        self.n = np.size(self.data)
        self.n_finite = np.sum(np.isfinite(self.data))
        self.n_not_finite = self.n - self.n_finite
        self.max = np.nanmax(self.data)
        self.min = np.nanmin(self.data)
        self.mean = np.nanmean(self.data)
        self.median = np.nanmedian(self.data)
        self.hist, self.xhist = np.histogram(self.data, **histogram_kwargs)
        self.x = 0.5 * (self.xhist[0:-2] + self.xhist[1:-1])
        self.mode = self.xhist[np.argmax(self.hist)]
        self.std = np.nanstd(self.data)
        self.cred = []
        for cilevel in ci:
            self.cred.append(np.nanquantile(self.data, cilevel))

    @property
    def is_all_finite(self):
        return self.n == self.n_finite


# Calculate a mask.  The mask eliminates results that we do not wish to consider,
# for example, bad fits.  The mask is calculated using all the variable
# output.  True values will be masked out
mask = np.zeros_like(outputs[:, :, 0], dtype=bool)
for i, output_name in enumerate(output_names):
    data = outputs[:, :, i]

    # Finiteness
    is_not_finite = ~np.isfinite(data)
    mask = np.logical_or(mask, is_not_finite)

    # Data that exceeds the lower bound is masked out
    lower_bound = df['lower_bound'][output_name]
    if lower_bound is None:
        lb_mask = np.zeros_like(mask)
    else:
        lb_mask = data < float(lower_bound)
    mask = np.logical_or(mask, lb_mask)

    # Data that exceeds the upper bound is masked out
    upper_bound = df['upper_bound'][output_name]
    if upper_bound is None:
        ub_mask = np.zeros_like(mask)
    else:
        ub_mask = data > float(upper_bound)
    mask = np.logical_or(mask, ub_mask)

# Make the plots
for i, output_name in enumerate(output_names):
    data = np.ma.array(outputs[:, :, i], mask=mask)

    # Total number of fits, including bad ones
    n_samples = data.size

    # Compressed
    compressed = data.flatten().compressed()
    n_good = compressed.size
    n_bad = n_samples - n_good

    # Summary statistics
    ss = SummaryStatistics(compressed, bins=bins, ci=(0.16, 0.84, 0.025, 0.975))

    # The variable name is used in the plot instead of the output_name
    # because we use LaTeX in the plots to match with the variables
    # used in the paper.
    variable_name = df['variable_name'][output_name]

    # Percentage that are bad fits
    percent_bad_string = "{:.1f}$\%$".format(100*n_bad/n_samples)

    # Information that goes in to every title
    title_information = f"{variable_name}\n{n_samples} fits, {n_bad} bad, {n_good} good, {percent_bad_string} bad"

    # Histograms
    plt.close('all')
    fig, ax = plt.subplots()
    h = ax.hist(compressed, bins=bins)
    plt.xlabel(variable_name)
    plt.ylabel('number')
    plt.title(f'histogram of {title_information}')
    plt.grid(linestyle=":")
    ax.axvline(ss.mean, label='mean ({:.2f})'.format(ss.mean), color='r')
    ax.axvline(ss.mode, label='mode ({:.2f})'.format(ss.mode), color='k')
    ax.axvline(ss.median, label='median ({:.2f})'.format(ss.median), color='y')
    ax.axvline(ss.cred[0], color='r', linestyle=':')
    ax.axvline(ss.cred[1], label='$\pm1\sigma$ equiv. ({:.2f}$\\rightarrow${:.2f})'.format(ss.cred[0], ss.cred[1]),
               color='r', linestyle=':')
    ax.axvline(ss.cred[2], color='k', linestyle=':')
    ax.axvline(ss.cred[3], label='$\pm2\sigma$ equiv. ({:.2f}$\\rightarrow${:.2f})'.format(ss.cred[2], ss.cred[3]),
               color='k', linestyle=':')
    ax.legend()
    filename = 'histogram.{:s}.{:s}.png'.format(observation_model_name, output_name)
    filename = os.path.join(directory, filename)
    plt.savefig(filename)

    # Spatial distribution
    title_information = f"{variable_name}\n{n_samples} fits, {n_bad} bad (in {bad_color}), {n_good} good, {percent_bad_string} bad"
    plt.close('all')
    fig, ax = plt.subplots()
    im = ax.imshow(data, origin='lower')
    im.cmap.set_bad(bad_color)
    ax.set_xlabel('solar X')
    ax.set_ylabel('solar Y')
    ax.set_title(f'spatial distribution of {title_information}')
    ax.grid(linestyle=":")
    fig.colorbar(im, ax=ax, label=variable_name)
    filename = 'spatial.{:s}.{:s}.png'.format(observation_model_name, output_name)
    filename = os.path.join(directory, filename)
    plt.savefig(filename)
