import os
import numpy as np
import matplotlib.pyplot as plt

observation_model_name = 'pl_c'

directory = os.path.expanduser('~/Data/ts/project_data/test_dask2_output')
filename = '{:s}.outputs.npz'.format(observation_model_name)
filepath = os.path.join(directory, filename)

outputs = np.load(filepath)['arr_0']

filename = '{:s}.output_names.txt'.format(observation_model_name)
filepath = os.path.join(directory, filename)
print('Reading ' + filepath)
with open(filepath) as f:
    output_names = [line.rstrip() for line in f]

for i, output_name in enumerate(output_names):
    data = outputs[:, :, i]
    nsamples = data.size

    # Histograms
    plt.close('all')
    n, bins, patches = plt.hist(data.flatten(), 200)
    plt.xlabel(output_name)
    plt.ylabel('Number')
    plt.title(f'Histogram of {output_name}\n{nsamples} samples')
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
    ax.set_title(f'Spatial distribution {output_name}\n{nsamples} samples')
    ax.grid(linestyle=":")
    fig.colorbar(im, ax=ax)
    filename = 'spatial.{:s}.{:s}.png'.format(observation_model_name, output_name)
    filename = os.path.join(directory, filename)
    plt.savefig(filename)
