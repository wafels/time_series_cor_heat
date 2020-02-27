import numpy as np
from astropy.modeling import Fittable1DModel, Parameter


class LogGaussian1D(Fittable1DModel):
    """
    One dimensional log-Gaussian model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the log-Gaussian.
    mean : float
        Mean of the log-Gaussian.
    stddev : float
        Standard deviation of the log-Gaussian.

    Notes
    -----

    Model formula:

        .. math:: f(x) = A e^{- \\frac{\\left(\ln(x) - \ln(x_{0})\\right)^{2}}{2 \\sigma^{2}}}
        .. math:: x>0

    Examples
    --------
    >>> from astropy.modeling import models
    >>> def tie_center(model):
    ...         mean = 50 * model.stddev
    ...         return mean
    >>> tied_parameters = {'mean': tie_center}

    Specify that 'mean' is a tied parameter in one of two ways:

    >>> g1 = models.LogGaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                             tied=tied_parameters)

    or

    >>> g1 = models.LogGaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.mean.tied
    False
    >>> g1.mean.tied = tie_center
    >>> g1.mean.tied
    <function tie_center at 0x...>

    Fixed parameters:

    >>> g1 = models.LogGaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                        fixed={'stddev': True})
    >>> g1.stddev.fixed
    True

    or

    >>> g1 = models.LogGaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.stddev.fixed
    False
    >>> g1.stddev.fixed = True
    >>> g1.stddev.fixed
    True

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import LogGaussian1D

        plt.figure()
        s1 = LogGaussian1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

        plt.axis([-5, 5, -1, 4])
        plt.show()

    See Also
    --------
    Gaussian1D
    """

    amplitude = Parameter(default=1)
    mean = Parameter(default=0)

    # Ensure stddev makes sense if its bounds are not explicitly set.
    # stddev must be non-zero and positive.
    stddev = Parameter(default=1, bounds=(FLOAT_EPSILON, None))

    def bounding_box(self, factor=5.5):
        """
        Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``

        Parameters
        ----------
        factor : float
            The multiple of `stddev` used to define the limits.
            The default is 5.5, corresponding to a relative error < 1e-7.

        Examples
        --------
        >>> from astropy.modeling.models import LogGaussian1D
        >>> model = LogGaussian1D(mean=0, stddev=2)
        >>> model.bounding_box
        (-11.0, 11.0)

        This range can be set directly (see: `Model.bounding_box
        <astropy.modeling.Model.bounding_box>`) or by using a different factor,
        like:

        >>> model.bounding_box = model.bounding_box(factor=2)
        >>> model.bounding_box
        (-4.0, 4.0)
        """

        x0 = self.mean
        dx = factor * self.stddev

        return (x0 - dx, x0 + dx)

    @property
    def fwhm(self):
        """Gaussian full width at half maximum."""
        return self.stddev * GAUSSIAN_SIGMA_TO_FWHM

    @staticmethod
    def evaluate(x, amplitude, mean, stddev):
        """
        LogGaussian1D model function.
        """
        return amplitude * np.exp(- 0.5 * (np.log(x) - mean) ** 2 / stddev ** 2)

    @staticmethod
    def fit_deriv(x, amplitude, mean, stddev):
        """
        LogGaussian1D model function derivatives.
        """
        logx = np.log(x)
        d_amplitude = np.exp(-0.5 / stddev ** 2 * (logx - mean) ** 2)
        d_mean = amplitude * d_amplitude * (logx - mean) / stddev ** 2
        d_stddev = amplitude * d_amplitude * (logx - mean) ** 2 / stddev ** 3
        return [d_amplitude, d_mean, d_stddev]

    @property
    def input_units(self):
        if self.mean.unit is None:
            return None
        else:
            return {'x': self.mean.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {'mean': inputs_unit['x'],
                'stddev': inputs_unit['x'],
                'amplitude': outputs_unit['y']}
