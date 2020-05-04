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


class SummaryStatistics:
    """Some simple descriptive statistics of a 1-D input array"""
    def __init__(self, data, ci=[0.68, 0.95], **kwargs):
        self.n = np.size(data)
        self.n_finite = np.sum(np.isfinit(data))
        self.n_not_finite = self.n - self.n_finite
        self.max = np.nanmax(data)
        self.min = np.nanmin(data)
        self.mean = np.nanmean(data)
        self.median = np.nanmedian(data)
        self.hist, self.xhist = np.histogram(data, **kwargs)
        self.x = 0.5 * (self.xhist[0:-2] + self.xhist[1:-1])
        self.mode = self.xhist[np.argmax(self.hist)]
        self.std = np.nanstd(data)
        self.cred = {}
        for cilevel in ci:
            lo = 0.5 * (1.0 - cilevel)
            hi = 1.0 - lo
            sorted_data = np.sort(data)
            self.cred[cilevel] = [sorted_data[np.rint(lo * self.n)],
                                  sorted_data[np.rint(hi * self.n)]]

    @property
    def is_all_finite(self):
        return self.n == self.n_finite


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
