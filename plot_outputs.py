import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

observation_model_name = 'pl_c'

directory = os.path.expanduser('~/Data/ts/project_data/test_dask2_output')

# Load in some information about how to treat and plot the outputs, for example
# output_name,lower_bound,upper_bound,variable_name
# "amplitude_0",None,None,"A_{0}"
# "alpha_0",0,4,"n"
filename = '{:s}.outputs_information.csv'.format(observation_model_name)
filepath = os.path.join(directory, filename)
df = pd.read_csv(filepath)
df = df.replace({"None": None})

# Load in the data and the output names 
filename = '{:s}.outputs.npz'.format(observation_model_name)
filepath = os.path.join(directory, filename)
outputs = np.load(filepath)['arr_0']

filename = '{:s}.output_names.txt'.format(observation_model_name)
filepath = os.path.join(directory, filename)
with open(filepath) as f:
    output_names = [line.rstrip() for line in f]

# Calculate a mask.  The mask eliminates results that we do not wish to consider,
# for example, bad fits.  The mask is calculated using all the variable
# output.  True values will be masked out
mask = np.zeros_like(outputs[:, :, i], dtype=bool)
for i, output_name in enumerate(output_names):
    data = outputs[:, :, i]

    # Finiteness
    is_not_finite = ~np.isfinite(data)

    # Update the mask
    mask = np.logical_or(mask, is_not_finite)

    # Data that exceeds the lower bound is masked out
    lower_bound = float(df['lower_bound'][output_name])
    if lower_bound is None:
        lb_mask = np.zeros_like(mask)
    else:
        lb_mask = data < lower_bound

    # Update the mask
    mask = np.logical_or(is_not_finite, lb_mask)

    # Data that exceeds the upper bound is masked out
    upper_bound = float(df['upper_bound'][output_name])
    if upper_bound is None:
        ub_mask= np.zeros_like(mask)
    else:
        ub_mask= data > upper_bound

    # Update the mask
    mask = np.logical_or(mask, ub_mask)


# Make the plots
for i, output_name in enumerate(output_names):
    data = np.ma.array(outputs[:, :, i], mask=mask)

    # Total number of fits, including bad ones
    nsamples = data.size

    # Total number of good fits
    ngood = nsamples - np.sum(mask)

    # The variable name is used in the plot instead of the output_name
    # because we use LaTeX in the plots to match with the variables
    # used in the paper.
    variable_name = df['variable_name'][output_name]

    # Histograms
    plt.close('all')
    n, bins, patches = plt.hist(data.flatten().compressed(), nbins=nbins)
    plt.xlabel(variable_name)
    plt.ylabel('Number')
    plt.title(f'Histogram of {variable_name}\n{nsamples} samples, {ngood} good')
    plt.grid(linestyle=":")
    filename = 'histogram.{:s}.{:s}.png'.format(observation_model_name, output_name)
    filename = os.path.join(directory, filename)
    plt.savefig(filename)

    # Spatial distribution
    plt.close('all')
    fig, ax = plt.subplots()
    im = ax.imshow(data, origin='lower')
    ax.set_xlabel('solar X')
    ax.set_ylabel('solar Y')
    ax.set_title(f'Spatial distribution {variable_name}\n{nsamples} samples, {ngood} good')
    ax.grid(linestyle=":")
    fig.colorbar(im, ax=ax)
    filename = 'spatial.{:s}.{:s}.png'.format(observation_model_name, output_name)
    filename = os.path.join(directory, filename)
    plt.savefig(filename)
