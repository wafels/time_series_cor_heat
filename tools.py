import numpy as np


class SummaryStatistics:
    def __init__(self, d, ci=(0.16, 0.84), **histogram_kwargs):
        """
        Return some simple descriptive statistics of an input array

        Parameters
        ----------
        d : `~numpy.array`
            An n-dimensional array of data.  The data must not have a mask.

        ci : ~tuple
            The value of the input data at these quantiles are calculated.

        histogram_kwargs :
            Keywords accepted by `~numpy.histogram`.

        """
        # Only work with unmasked data.
        if np.ma.is_masked(d):
            raise ValueError('Unmasked data only')
        self.data = d.flatten()
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
        self.ci = ci
        self.cred = []
        for cilevel in self.ci:
            self.cred.append(np.nanquantile(self.data, cilevel))

    @property
    def is_all_finite(self):
        """Returns True if all elements are finite."""
        return np.all(np.isfinite(self.data))
