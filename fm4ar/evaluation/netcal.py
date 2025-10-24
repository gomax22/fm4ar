# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import warnings
import numpy as np
import matplotlib.pyplot as plt

from typing import Iterable, Tuple, Union, List
from scipy.stats import norm
from scipy.interpolate import interp1d, griddata

from netcal import accepts, meanvar, cumulative_moments
from netcal import is_in_quantile
from netcal.metrics.Miscalibration import _Miscalibration
from netcal.metrics.regression import QCE

# Adapted version of ReliabilityQCE from netcal package
# Tikzplotlib parts have been commented out to avoid dependency issues
class ReliabilityQCE(object):
    """
    Visualizes the Conditional Quantile Calibration Error (C-QCE) in the scope of regression calibration as a bar chart
    for probabilistic regression models.
    See :class:`netcal.metrics.regression.QCE` for a detailed documentation of the C-QCE [1]_.
    This method is able to visualize the C-QCE in terms of multiple univariate distributions if the input is given
    as multiple independent Gaussians.
    This method is also able to visualize the multivariate C-QCE for a multivariate Gaussian if the input is given
    with covariance matrices.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the C-QCE binning.
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).

    References
    ----------
    .. [1] Küppers, Fabian, Schneider, Jonas, and Haselhoff, Anselm:
       "Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection."
       European Conference on Computer Vision (ECCV) Workshops, 2022.
       `Get source online <https://arxiv.org/pdf/2207.01242.pdf>`__
    """

    eps = np.finfo(np.float32).eps

    def __init__(self, bins: Union[int, Iterable[float], np.ndarray] = 10):
        """ Constructor. For detailed parameter documentation view classdocs. """

        self.qce = QCE(bins=bins, marginal=False)

    def plot(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            q: Union[float, Iterable[float], np.ndarray],
            *,
            kind: str = 'meanstd',
            range_: List[Tuple[float, float]] = None,
            filename: str = None,
            tikz: bool = False,
            title_suffix: str = None,
            fig: plt.Figure = None,
            **save_args
    ) -> Union[plt.Figure, str]:
        """
        Visualizes the C-QCE as a bar chart either for multiple univariate data (if standard deviations are given as
        input) or for a joint multivariate distribution (if covariance matrices are given as input).
        See parameter "kind" for a detailed description of the input format.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        q : np.ndarray of shape (q,)
            Quantile scores in [0, 1] of size q to compute the x-valued quantile boundaries for.
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) and (n, [d, [d]]), this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If the second NumPy array has shape (n, d, d), this method asserts covariance matrices as input
                       for each sample. In this case, the NLL is calculated for multivariate distributions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions).
        range_ : list of length d with tuples (lower_bound: float, upper_bound: float)
            List of tuples that define the binning range of the standard deviation for each dimension separately.
            For example, if input data is given with only a few samples having high standard deviations,
            this might distort the calculations as the binning scheme commonly takes the (min, max) as the range
            for the binning, yielding a high amount of empty bins.
        filename : str, optional, default: None
            Optional filename to save the plotted figure.
        tikz : bool, optional, default: False
            If True, use 'tikzplotlib' package to return tikz-code for Latex rather than a Matplotlib figure.
        title_suffix : str, optional, default: None
            Suffix for plot title.
        fig: plt.Figure, optional, default: None
            If given, the figure instance is used to draw the reliability diagram.
            If fig is None, a new one will be created.
        **save_args : args
            Additional arguments passed to 'matplotlib.pyplot.Figure.savefig' function if 'tikz' is False.
            If 'tikz' is True, the argument are passed to 'tikzplotlib.get_tikz_code' function.

        Returns
        -------
        matplotlib.pyplot.Figure if 'tikz' is False else str with tikz code.
            Visualization of the C-QCE either as Matplotlib figure or as string with tikz code.
        """

        # measure QCE and return a miscalibration map
        _, qce_map, num_samples_hist = self.qce.measure(
            X=X,
            y=y,
            q=q,
            kind=kind,
            reduction="none",
            range_=range_,
            return_map=True,
            return_num_samples=True
        )  # (q, b)

        # get number of dimensions
        ndims = len(qce_map)

        # catch if bin_edges is None
        assert len(self.qce._bin_edges) != 0, "Fatal error: could not compute bin_edges for ReliabilityQCE."

        # initialize plot and create an own chart for each dimension
        if fig is None:
            fig, axes = plt.subplots(nrows=2, ncols=ndims, figsize=(7 * ndims, 6), squeeze=False)
        else:

            axes = [
                [fig.add_subplot(2, ndims, idx) for idx in range(1, ndims + 1)],
                [fig.add_subplot(2, ndims, idx) for idx in range(ndims + 1, 2 * ndims + 1)],
            ]

        for dim in range(ndims):

            # convert absolute number of samples to relative amount
            n_samples_hist = np.divide(num_samples_hist[dim], np.sum(num_samples_hist[dim]))

            # compute mean over all quantiles as well as mean over all bins separately
            mean_over_quantiles = np.mean(qce_map[dim], axis=0)  # (b,)

            # get binning boundaries for actual dimension
            bounds = self.qce._bin_edges[dim]  # (b+1)

            for ax, metric, title, ylabel in zip(
                [axes[0][dim], axes[1][dim]],
                [n_samples_hist, mean_over_quantiles],
                ["Sample Histogram", "QCE mean over quantiles"],
                ["% of Samples", "Quantile Calibration Error (QCE)"],
            ):

                # draw bar chart with given edges and metrics
                ax.bar(bounds[:-1], height=metric, width=np.diff(bounds), align='edge', edgecolor='black')

                # set axes edges
                ax.set_xlim((bounds[0], bounds[-1]))
                ax.set_ylim((0., 1.))

                # labels and grid
                if self.qce._is_cov:
                    title = title + ' - multivariate'
                    ax.set_xlabel("sqrt( Standardized Generalized Variance (SGV) )")
                else:
                    title = title + ' - dim %02d' % dim
                    ax.set_xlabel("Standard Deviation")

                ax.set_ylabel(ylabel)
                ax.grid(True)

                # set axis title
                if title_suffix is not None:
                    title = title + ' - ' + title_suffix

                ax.set_title(title)

        fig.tight_layout()

        # if tikz is true, create tikz code from matplotlib figure
        # if tikz:

            # get tikz code for our specific figure and also pass filename to store possible bitmaps
            # tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=filename, **save_args)

            # close matplotlib figure when tikz figure is requested to save memory
            # plt.close(fig)
            # fig = tikz_fig

        # save figure either as matplotlib PNG or as tikz output file
        if filename is not None:
            if tikz:
                with open(filename, "w") as open_file:
                    open_file.write(fig)
            else:
                fig.savefig(filename, **save_args)

        return fig


# Adapted version of ReliabilityQCE from netcal package
# Tikzplotlib parts have been commented out to avoid dependency issues
class ReliabilityRegression(object):
    """
    Reliability diagram in the scope of regression calibration for probabilistic regression models.
    This diagram visualizes the quantile coverage frequency for several quantile levels and plots these observed
    coverage scores above the desired quantile levels.
    In this way, it is possible to compare the predicted and the observed quantile levels with each other.

    This method is able to visualize the quantile coverage in terms of multiple univariate distributions if the input
    is given as multiple independent Gaussians.
    This method is also able to visualize the multivariate quantile coverage for a joint multivariate Gaussian if the
    input is given with covariance matrices.

    Parameters
    ----------
    quantiles : int or iterable, default: 11
        Quantile levels that are used for the visualization of the regression reliability diagram.
        If int, use NumPy's linspace method to get the quantile levels.
        If iterable, use the specified quantiles for visualization.
    """

    eps = np.finfo(np.float32).eps

    def __init__(self, quantiles: Union[int, Iterable[float], np.ndarray] = 11):
        """ Constructor. For detailed parameter documentation view classdocs. """

        # init list of quantiles if input type is int
        if isinstance(quantiles, int):
            self.quantiles = np.clip(np.linspace(0., 1., quantiles), self.eps, 1.-self.eps)

        # use input list or array as quantile list
        elif isinstance(quantiles, (list, np.ndarray)):

            # at this point, allow for 0 and 1 quantile to be aligned on the miscalibration curve
            assert (quantiles >= 0).all(), "Found quantiles <= 0."
            assert (quantiles <= 1).all(), "Found quantiles >= 1."
            self.quantiles = np.clip(np.array(quantiles), self.eps, 1.-self.eps)

        else:
            raise AttributeError("Unknown type \'%s\' for param \'quantiles\'." % type(quantiles))

    def plot(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            *,
            kind: str = 'meanstd',
            filename: str = None,
            tikz: bool = False,
            title_suffix: str = None,
            feature_names: List[str] = None,
            fig: plt.Figure = None,
            **save_args
    ) -> Union[plt.Figure, str]:
        """
        Reliability diagram for regression calibration to visualize the predicted quantile levels vs. the actually
        observed quantile coverage probability.
        This method is able to visualize the reliability diagram in terms of multiple univariate distributions if the
        input is given as multiple independent Gaussians.
        This method is also able to visualize the joint multivariate quantile calibration for a multivariate Gaussian
        if the input is given with covariance matrices (see parameter "kind" for a detailed description of the input
        format).

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) and (n, [d, [d]]), this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If the second NumPy array has shape (n, d, d), this method asserts covariance matrices as input
                       for each sample. In this case, the NLL is calculated for multivariate distributions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions).
        filename : str, optional, default: None
            Optional filename to save the plotted figure.
        tikz : bool, optional, default: False
            If True, use 'tikzplotlib' package to return tikz-code for Latex rather than a Matplotlib figure.
        title_suffix : str, optional, default: None
            Suffix for plot title.
        feature_names : list, optional, default: None
            Names of the additional features that are attached to the axes of a reliability diagram.
        fig: plt.Figure, optional, default: None
            If given, the figure instance is used to draw the reliability diagram.
            If fig is None, a new one will be created.
        **save_args : args
            Additional arguments passed to 'matplotlib.pyplot.Figure.savefig' function if 'tikz' is False.
            If 'tikz' is True, the argument are passed to 'tikzplotlib.get_tikz_code' function.

        Returns
        -------
        matplotlib.pyplot.Figure if 'tikz' is False else str with tikz code.
            Visualization of the quantile calibration either as Matplotlib figure or as string with tikz code.
        """

        assert kind in ['meanstd', 'cauchy', 'cumulative'], 'Parameter \'kind\' must be either \'meanstd\', or \'cumulative\'.'

        # get quantile coverage of input
        in_quantile, _, _, _, _ = is_in_quantile(X, y, self.quantiles, kind)  # (q, n, [d]), (q, n, d), (n, d), (n, d, [d])

        # get the frequency of which y is within the quantile bounds
        frequency = np.mean(in_quantile, axis=1)  # (q, [d])

        # make frequency array at least 2d
        if frequency.ndim == 1:
            frequency = np.expand_dims(frequency, axis=1)  # (q, d) or (q, 1)

        n_dims = frequency.shape[-1]

        # check feature names parameter
        if feature_names is not None:
            assert isinstance(feature_names, (list, tuple)), "Parameter \'feature_names\' must be tuple or list."
            assert len(feature_names) == n_dims, "Length of parameter \'feature_names\' must be equal to the amount " \
                                                 "of dimensions. Input with full covariance matrices is interpreted " \
                                                 "as n_features=1."

        # initialize plot and create an own chart for each dimension
        if fig is None:
            fig, axes = plt.subplots(nrows=n_dims, figsize=(7, 3 * n_dims), squeeze=False)
        else:
            axes = [fig.add_subplot(n_dims, 1, idx) for idx in range(1, n_dims + 1)]

        for dim, ax in enumerate(axes):

            # ax object also has an extra dim for columns
            ax = ax[0]

            ax.plot(self.quantiles, frequency[:, dim], "o-")

            # draw diagonal as perfect calibration line
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.0, 1.0))

            # labels and legend of second plot
            ax.set_xlabel('Expected quantile')
            ax.set_ylabel('Observed frequency')
            ax.legend(['Output', 'Perfect Calibration'])
            ax.grid(True)

            # set axis title
            title = 'Reliability Regression Diagram'
            if title_suffix is not None:
                title = title + ' - ' + title_suffix
            if feature_names is not None:
                title = title + ' - ' + feature_names[dim]
            else:
                title = title + ' - dim %02d' % dim

            ax.set_title(title)

        fig.tight_layout()

        # if tikz is true, create tikz code from matplotlib figure
        # if tikz:

            # get tikz code for our specific figure and also pass filename to store possible bitmaps
            # tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=filename, **save_args)

            # close matplotlib figure when tikz figure is requested to save memory
            # plt.close(fig)
            # fig = tikz_fig

        # save figure either as matplotlib PNG or as tikz output file
        if filename is not None:
            if tikz:
                with open(filename, "w") as open_file:
                    open_file.write(fig)
            else:
                fig.savefig(filename, **save_args)

        return fig


# Adapted version of ReliabilityQCE from netcal package
# Tikzplotlib parts have been commented out to avoid dependency issues
class ReliabilityDiagram(object):
    """
    Plot Confidence Histogram and Reliability Diagram to visualize miscalibration in the context of
    confidence calibration.
    On classification, plot the gaps between average confidence and observed accuracy bin-wise over the confidence
    space [1]_, [2]_.
    On detection, plot the miscalibration w.r.t. the additional regression information provided (1-D or 2-D) [3]_.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the ACE/ECE/MCE.
        On detection mode: if int, use same amount of bins for each dimension (nx1 = nx2 = ... = bins).
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
    equal_intervals : bool, optional, default: True
        If True, the bins have the same width. If False, the bins are splitted to equalize
        the number of samples in each bin.
    detection : bool, default: False
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    fmin : float, optional, default: None
        Minimum value for scale color.
    fmax : float, optional, default: None
        Maximum value for scale color.
    metric : str, default: 'ECE'
        Metric to measure miscalibration. Might be either 'ECE', 'ACE' or 'MCE'.

    References
    ----------
    .. [1] Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger:
       "On Calibration of Modern Neural Networks."
       Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
       `Get source online <https://arxiv.org/abs/1706.04599>`__

    .. [2] A. Niculescu-Mizil and R. Caruana:
       “Predicting good probabilities with supervised learning.”
       Proceedings of the 22nd International Conference on Machine Learning, 2005, pp. 625–632.
       `Get source online <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>`__

    .. [3] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020.
       `Get source online <https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf>`__
    """

    def __init__(
            self,
            bins: Union[int, Iterable[int]] = 10,
            *,
            equal_intervals: bool = True,
            detection: bool = False,
            sample_threshold: int = 1,
            fmin: float = None,
            fmax: float = None,
            metric: str = 'ECE',
            **kwargs
    ):
        """ Constructor. For detailed parameter documentation view classdocs. """

        self.bins = bins
        self.detection = detection
        self.sample_threshold = sample_threshold
        self.fmin = fmin
        self.fmax = fmax
        self.metric = metric

        if 'feature_names' in kwargs:
            self.feature_names = kwargs['feature_names']

        if 'title_suffix' in kwargs:
            self.title_suffix = kwargs['title_suffix']

        self._miscalibration = _Miscalibration(
            bins=bins, equal_intervals=equal_intervals,
            detection=detection, sample_threshold=sample_threshold
        )

    def plot(
            self,
            X: Union[Iterable[np.ndarray], np.ndarray],
            y: Union[Iterable[np.ndarray], np.ndarray],
            *,
            batched: bool = False,
            uncertainty: str = None,
            filename: str = None,
            tikz: bool = False,
            title_suffix: str = None,
            feature_names: List[str] = None,
            fig: plt.Figure = None,
            **save_args
    ) -> Union[plt.Figure, str]:
        """
        Reliability diagram to visualize miscalibration. This could be either in classical way for confidences only
        or w.r.t. additional properties (like x/y-coordinates of detection boxes, width, height, etc.). The additional
        properties get binned. Afterwards, the miscalibration will be calculated for each bin. This is
        visualized as a 2-D plots.

        Parameters
        ----------
        X : iterable of np.ndarray, or np.ndarray of shape=([n_bayes], n_samples, [n_classes/n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            If 3-D, interpret first dimension as samples from an Bayesian estimator with mulitple data points
            for a single sample (e.g. variational inference or MC dropout samples).
            If this is an iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : iterable of np.ndarray with same length as X or np.ndarray of shape=([n_bayes], n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
            If 3-D, interpret first dimension as samples from an Bayesian estimator with mulitple data points
            for a single sample (e.g. variational inference or MC dropout samples).
            If iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
        batched : bool, optional, default: False
            Multiple predictions can be evaluated at once (e.g. cross-validation examinations) using batched-mode.
            All predictions given by X and y are separately evaluated and their results are averaged afterwards
            for visualization.
        uncertainty : str, optional, default: False
            Define uncertainty handling if input X has been sampled e.g. by Monte-Carlo dropout or similar methods
            that output an ensemble of predictions per sample. Choose one of the following options:
            - flatten:  treat everything as a separate prediction - this option will yield into a slightly better
                        calibration performance but without the visualization of a prediction interval.
            - mean:     compute Monte-Carlo integration to obtain a simple confidence estimate for a sample
                        (mean) with a standard deviation that is visualized.
        filename : str, optional, default: None
            Optional filename to save the plotted figure.
        tikz : bool, optional, default: False
            If True, use 'tikzplotlib' package to return tikz-code for Latex rather than a Matplotlib figure.
        title_suffix : str, optional, default: None
            Suffix for plot title.
        feature_names : list, optional, default: None
            Names of the additional features that are attached to the axes of a reliability diagram.
        fig: plt.Figure, optional, default: None
            If given, the figure instance is used to draw the reliability diagram.
            If fig is None, a new one will be created.
        **save_args : args
            Additional arguments passed to 'matplotlib.pyplot.Figure.savefig' function if 'tikz' is False.
            If 'tikz' is True, the argument are passed to 'tikzplotlib.get_tikz_code' function.

        Returns
        -------
        matplotlib.pyplot.Figure if 'tikz' is False else str with tikz code.
            Visualization of the reliability diagrams either as Matplotlib figure or as string with tikz code.

        Raises
        ------
        AttributeError
            - If parameter metric is not string or string is not 'ACE', 'ECE' or 'MCE'
            - If parameter 'feature_names' is set but length does not fit to second dim of X
            - If no ground truth samples are provided
            - If length of bins parameter does not match the number of features given by X
            - If more than 3 feature dimensions (including confidence) are provided
        """

        # assign deprecated constructor parameter to title_suffix and feature_names
        if hasattr(self, 'title_suffix') and title_suffix is None:
            title_suffix = self.title_suffix

        if hasattr(self, 'feature_names') and feature_names is None:
            feature_names = self.feature_names

        # check if metric is correct
        if not isinstance(self.metric, str):
            raise AttributeError('Parameter \'metric\' must be string with either \'ece\', \'ace\' or \'mce\'.')

        # check metrics parameter
        if self.metric.lower() not in ['ece', 'ace', 'mce']:
            raise AttributeError('Parameter \'metric\' must be string with either \'ece\', \'ace\' or \'mce\'.')
        else:
            self.metric = self.metric.lower()

        # perform checks and prepare input data
        X, matched, sample_uncertainty, bin_bounds, num_features = self._miscalibration.prepare(X, y, batched, uncertainty)
        if num_features > 3:
            raise AttributeError("Diagram is not defined for more than 2 additional feature dimensions.")

        histograms = []
        num_samples_hist = []
        for batch_X, batch_matched, batch_uncertainty, bounds in zip(X, matched, sample_uncertainty, bin_bounds):

            batch_histograms, batch_num_samples, _, _ = self._miscalibration.binning(
                bounds,
                batch_X,
                batch_matched,
                batch_X[:, 0],
                batch_uncertainty[:, 0]
            )

            histograms.append(batch_histograms)
            num_samples_hist.append(batch_num_samples)

        # no additional dimensions? compute standard reliability diagram
        if num_features == 1:
            fig = self.__plot_confidence_histogram(X, matched, histograms, num_samples_hist, bin_bounds, title_suffix, fig)

        # one additional feature? compute 1D-plot
        elif num_features == 2:
            fig = self.__plot_1d(histograms, num_samples_hist, bin_bounds, title_suffix, feature_names, fig)

        # two additional features? compute 2D plot
        elif num_features == 3:
            fig = self.__plot_2d(histograms, num_samples_hist, bin_bounds, title_suffix, feature_names, fig)

        # number of dimensions exceeds 3? quit
        else:
            raise AttributeError("Diagram is not defined for more than 2 additional feature dimensions.")

        # if tikz is true, create tikz code from matplotlib figure
        # if tikz:

            # get tikz code for our specific figure and also pass filename to store possible bitmaps
            # tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=filename, **save_args)

            # close matplotlib figure when tikz figure is requested to save memory
            # plt.close(fig)
            # fig = tikz_fig

        # save figure either as matplotlib PNG or as tikz output file
        if filename is not None:
            if tikz:
                with open(filename, "w") as open_file:
                    open_file.write(fig)
            else:
                fig.savefig(filename, **save_args)

        return fig

    @classmethod
    def __interpolate_grid(cls, metric_map: np.ndarray) -> np.ndarray:
        """ Interpolate missing values in a 2D-grid using the mean of the data. The interpolation is done inplace. """

        # get all NaNs
        nans = np.isnan(metric_map)
        x = lambda z: z.nonzero()

        # get mean of the remaining values and interpolate missing by the mean
        mean = float(np.mean(metric_map[~nans]))
        metric_map[nans] = griddata(x(~nans), metric_map[~nans], x(nans), method='cubic', fill_value=mean)
        return metric_map

    def __plot_confidence_histogram(
            self,
            X: List[np.ndarray],
            matched: List[np.ndarray],
            histograms: List[np.ndarray],
            num_samples_hist: List[np.ndarray],
            bin_bounds: List,
            title_suffix: str = None,
            fig: plt.Figure = None,
    ) -> plt.Figure:
        """ Plot confidence histogram and reliability diagram to visualize miscalibration for condidences only. """

        # get number of bins (self.bins has not been processed yet)
        n_bins = len(bin_bounds[0][0])-1

        median_confidence = [(bounds[0][1:] + bounds[0][:-1]) * 0.5 for bounds in bin_bounds]
        mean_acc, mean_conf = [], []
        for batch_X, batch_matched, batch_hist, batch_num_samples, batch_median in zip(
                X, matched, histograms, num_samples_hist, median_confidence
        ):

            acc_hist, conf_hist, _ = batch_hist
            empty_bins, = np.nonzero(batch_num_samples == 0)

            # calculate overall mean accuracy and confidence
            mean_acc.append(np.mean(batch_matched))
            mean_conf.append(np.mean(batch_X))

            # set empty bins to median bin value
            acc_hist[empty_bins] = batch_median[empty_bins]
            conf_hist[empty_bins] = batch_median[empty_bins]

            # convert num_samples to relative afterwards (inplace denoted by [:])
            batch_num_samples[:] = batch_num_samples / np.sum(batch_num_samples)

        # get mean histograms and values over all batches
        acc = np.mean([hist[0] for hist in histograms], axis=0)
        conf = np.mean([hist[1] for hist in histograms], axis=0)
        uncertainty = np.sqrt(np.mean([hist[2] for hist in histograms], axis=0))
        num_samples = np.mean([x for x in num_samples_hist], axis=0)
        mean_acc = np.mean(mean_acc)
        mean_conf = np.mean(mean_conf)
        median_confidence = np.mean(median_confidence, axis=0)
        bar_width = np.mean([np.diff(bounds[0]) for bounds in bin_bounds], axis=0)

        # compute credible interval of uncertainty
        p = 0.05
        z_score = norm.ppf(1. - (p / 2))
        uncertainty = z_score * uncertainty

        # if no uncertainty is given, set variable uncertainty to None in order to prevent drawing error bars
        if np.count_nonzero(uncertainty) == 0:
            uncertainty = None

        # calculate deviation
        deviation = conf - acc

        # -----------------------------------------
        # plot data distribution histogram first
        if fig is None:
            fig, axes = plt.subplots(2, squeeze=True, figsize=(7, 6))

        # use an existing figure object if given
        else:
            axes = [
                fig.add_subplot(2, 1, 1),
                fig.add_subplot(2, 1, 2),
            ]

        ax = axes[0]

        # set title suffix is given
        if title_suffix is not None:
            ax.set_title('Confidence Histogram - ' + title_suffix)
        else:
            ax.set_title('Confidence Histogram')

        # create bar chart with relative amount of samples in each bin
        # as well as average confidence and accuracy
        ax.bar(median_confidence, height=num_samples, width=bar_width, align='center', edgecolor='black')
        ax.plot([mean_acc, mean_acc], [0.0, 1.0], color='black', linestyle='--')
        ax.plot([mean_conf, mean_conf], [0.0, 1.0], color='gray', linestyle='--')
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))

        # labels and legend
        ax.set_xlabel('Confidence')
        ax.set_ylabel('% of Samples')
        ax.legend(['Avg. Accuracy', 'Avg. Confidence', 'Relative Amount of Samples'])

        # second plot: reliability histogram
        ax = axes[1]

        # set title suffix if given
        if title_suffix is not None:
            ax.set_title('Reliability Diagram' + " - " + title_suffix)
        else:
            ax.set_title('Reliability Diagram')

        # create two overlaying bar charts with bin accuracy and the gap of each bin to the perfect calibration
        ax.bar(median_confidence, height=acc, width=bar_width, align='center',
               edgecolor='black', yerr=uncertainty, capsize=4)
        ax.bar(median_confidence, height=deviation, bottom=acc, width=bar_width, align='center',
               edgecolor='black', color='red', alpha=0.6)

        # draw diagonal as perfect calibration line
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))

        # labels and legend of second plot
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.legend(['Perfect Calibration', 'Output', 'Gap'])

        fig.tight_layout()
        return fig

    def __plot_1d(
            self,
            histograms: List[np.ndarray],
            num_samples_hist: List[np.ndarray],
            bin_bounds: List,
            title_suffix: str = None,
            feature_names: List[str] = None,
            fig: plt.Figure = None,
    ) -> plt.Figure:
        """ Plot 1-D miscalibration w.r.t. one additional feature. """

        # z score for credible interval (if uncertainty is given)
        p = 0.05
        z_score = norm.ppf(1. - (p / 2))

        results = []
        for batch_hist, batch_num_samples, bounds in zip(histograms, num_samples_hist, bin_bounds):

            result = self._miscalibration.process(self.metric, *batch_hist, num_samples_hist=batch_num_samples)
            bin_median = (bounds[-1][:-1] + bounds[-1][1:]) * 0.5

            # interpolate missing values
            x = np.linspace(0.0, 1.0, 1000)
            miscalibration = interp1d(bin_median, result[1], kind='cubic', fill_value='extrapolate')(x)
            acc = interp1d(bin_median, result[2], kind='cubic', fill_value='extrapolate')(x)
            conf = interp1d(bin_median, result[3], kind='cubic', fill_value='extrapolate')(x)
            uncertainty = interp1d(bin_median, result[4], kind='cubic', fill_value='extrapolate')(x)

            results.append((miscalibration, acc, conf, uncertainty))

        # get mean over all batches and convert mean variance to a std deviation afterwards
        miscalibration = np.mean([result[0] for result in results], axis=0)
        acc = np.mean([result[1] for result in results], axis=0)
        conf = np.mean([result[2] for result in results], axis=0)
        uncertainty = np.sqrt(np.mean([result[3] for result in results], axis=0))

        # draw routines
        if fig is None:
            fig, ax1 = plt.subplots()

        # use an existing figure object if given
        else:
            ax1 = fig.add_subplot(1, 1, 1)

        conf_color = 'tab:blue'

        # set name of the additional feature
        if feature_names is not None:
            ax1.set_xlabel(feature_names[0])

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.0])
        ax1.set_ylabel('accuracy/confidence', color=conf_color)

        # draw confidence and accuracy on the same (left) axis
        x = np.linspace(0.0, 1.0, 1000)
        line1, = ax1.plot(x, acc, '-.', color='black')
        line2, = ax1.plot(x, conf, '--', color=conf_color)
        ax1.tick_params('y', labelcolor=conf_color)

        # if uncertainty is given, compute average of variances over all bins and get std deviation by sqrt
        # compute credible interval afterwards
        # define lower and upper bound
        uncertainty = z_score * uncertainty
        lb = conf - uncertainty
        ub = conf + uncertainty

        # create second axis for miscalibration
        ax11 = ax1.twinx()
        miscal_color = 'tab:red'
        line3, = ax11.plot(x, miscalibration, '-', color=miscal_color)

        if self.metric == 'ace':
            ax11.set_ylabel('Average Calibration Error (ACE)', color=miscal_color)
        elif self.metric == 'ece':
            ax11.set_ylabel('Expected Calibration Error (ECE)', color=miscal_color)
        elif self.metric == 'mce':
            ax11.set_ylabel('Maximum Calibration Error (MCE)', color=miscal_color)

        ax11.tick_params('y', labelcolor=miscal_color)

        # set miscalibration limits if given
        if self.fmin is not None and self.fmax is not None:
            ax11.set_ylim([self.fmin, self.fmax])

        ax1.legend((line1, line2, line3),
                   ('accuracy', 'confidence', '%s' % self.metric.upper()),
                   loc='best')

        if title_suffix is not None:
            ax1.set_title('Accuracy, confidence and %s\n- %s -' % (self.metric.upper(), title_suffix))
        else:
            ax1.set_title('Accuracy, confidence and %s' % self.metric.upper())

        ax1.grid(True)

        fig.tight_layout()
        return fig

    def __plot_2d(
            self,
            histograms: List[np.ndarray],
            num_samples_hist: List[np.ndarray],
            bin_bounds: List[np.ndarray],
            title_suffix: str = None,
            feature_names: List[str] = None,
            fig: plt.Figure = None,
    ) -> plt.Figure:
        """ Plot 2D miscalibration reliability diagram heatmap. """

        results = []
        for batch_hist, batch_num_samples in zip(histograms, num_samples_hist):

            result = self._miscalibration.process(self.metric, *batch_hist, num_samples_hist=batch_num_samples)

            # interpolate 2D data inplace to avoid "empty" bins
            batch_samples = result[-1]
            for map in result[1:-1]:
                map[batch_samples == 0.0] = 0.0
                # TODO: check what to do here
                # map[batch_samples == 0.0] = np.nan
                # self.__interpolate_grid(map)

            # on interpolation, it is sometimes possible that empty bins have negative values
            # however, this is invalid for variance
            result[4][result[4] < 0] = 0.0
            results.append(result)

        # calculate mean over all batches and transpose
        # transpose is necessary. Miscalibration is calculated in the order given by the features
        # however, imshow expects arrays in format [rows, columns] or [height, width]
        # e.g., miscalibration with additional x/y (in this order) will be drawn [y, x] otherwise
        miscalibration = np.mean([result[1] for result in results], axis=0).T
        acc = np.mean([result[2] for result in results], axis=0).T
        conf = np.mean([result[3] for result in results], axis=0).T
        mean = np.mean([result[4] for result in results], axis=0).T
        uncertainty = np.sqrt(mean)

        # -----------------------------------------------------------------------------------------
        # draw routines

        def set_axis(ax, map, vmin=None, vmax=None):
            """ Generic function to set all subplots equally """
            # TODO: set proper fmin, fmax values
            img = ax.imshow(map, origin='lower', interpolation="gaussian", cmap='jet', aspect=1, vmin=vmin, vmax=vmax)

            # set correct x- and y-ticks
            ax.set_xticks(np.linspace(0., len(bin_bounds[0][1])-2, 5))
            ax.set_xticklabels(np.linspace(0., 1., 5))
            ax.set_yticks(np.linspace(0., len(bin_bounds[0][2])-2, 5))
            ax.set_yticklabels(np.linspace(0., 1., 5))
            ax.set_xlim([0.0, len(bin_bounds[0][1])-2])
            ax.set_ylim([0.0, len(bin_bounds[0][2])-2])

            # draw feature names on axes if given
            if feature_names is not None:
                ax.set_xlabel(feature_names[0])
                ax.set_ylabel(feature_names[1])

            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

            return ax, img

        # -----------------------------------

        # create only two subplots if no additional uncertainty is given
        if np.count_nonzero(uncertainty) == 0:
            if fig is None:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            else:
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)

        # process additional uncertainty if given
        else:
            if fig is None:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, squeeze=True, figsize=(10, 10))
            else:
                ax1 = fig.add_subplot(2, 2, 1)
                ax2 = fig.add_subplot(2, 2, 2)
                ax3 = fig.add_subplot(2, 2, 3)
                ax4 = fig.add_subplot(2, 2, 4)

            ax4, img4 = set_axis(ax4, uncertainty)

            if title_suffix is not None:
                ax4.set_title("Confidence std deviation\n- %s -" % title_suffix)
            else:
                ax4.set_title("Confidence std deviation")

        ax1, img1 = set_axis(ax1, acc, vmin=0, vmax=1)
        ax2, img2 = set_axis(ax2, conf, vmin=0, vmax=1)
        ax3, img3 = set_axis(ax3, miscalibration, vmin=self.fmin, vmax=self.fmax)

        # draw title if given
        if title_suffix is not None:
            ax1.set_title("Average accuracy\n- %s -" % title_suffix)
            ax2.set_title("Average confidence\n- %s -" % title_suffix)
            ax3.set_title("%s\n- %s -" % (self.metric.upper(), title_suffix))
        else:
            ax1.set_title("Average accuracy")
            ax2.set_title("Average confidence")
            ax3.set_title("%s" % self.metric.upper())

        # -----------------------------------------------------------------------------------------

        return fig


class ENCE(_Miscalibration):
    """
    Expected Normalized Calibration Error (ENCE) for a regression calibration evaluation to test for
    *variance calibration*. A probabilistic regression model takes :math:`X` as input and outputs a
    mean :math:`\\mu_Y(X)` and a standard deviation :math:`\\sigma_Y(X)` targeting the ground-truth :math:`y`.
    Similar to the :class:`netcal.metrics.confidence.ECE`, the ENCE applies a binning scheme with :math:`B` bins
    over the predicted standard deviation :math:`\\sigma_Y(X)` and measures the absolute (normalized) difference
    between root mean squared error (RMSE) and root mean variance (RMV) [1]_.
    Thus, the ENCE [1]_ is defined by

    .. math::
        \\text{ENCE} := \\frac{1}{B} \\sum^B_{b=1} \\frac{|RMSE(b) - RMV(b)|}{RMV(b)} ,

    where :math:`RMSE(b)` and :math:`RMV(b)` are the root mean squared error and the root mean variance within
    bin :math:`b`, respectively.

    If multiple dimensions are given, the ENCE is measured for each dimension separately.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the ENCE binning.
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
    sample_threshold : int, optional, default: 1
        Bins with an amount of samples below this threshold are not included into the miscalibration metrics.

    References
    ----------
    .. [1] Levi, Dan, et al.:
       "Evaluating and calibrating uncertainty prediction in regression tasks."
       arXiv preprint arXiv:1905.11659 (2019).
       `Get source online <https://arxiv.org/pdf/1905.11659.pdf>`__
    """

    @accepts((int, tuple, list), int)
    def __init__(
            self,
            bins: Union[int, Iterable[int]] = 10,
            sample_threshold: int = 1
    ):
        """ Constructor. For detailed parameter description, see class docs. """

        super().__init__(bins=bins, equal_intervals=True, detection=False, sample_threshold=sample_threshold)

    def measure(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            *,
            kind: str = 'meanstd',
            range_: List[Tuple[float, float]] = None,
    ):
        """
        Measure the ENCE for given input data either as tuple consisting of mean and stddev estimates or as
        NumPy array consisting of a sample distribution.
        If multiple dimensions are given, the ENCE is measured for each dimension separately.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions).
        range_ : list of length d with tuples (lower_bound: float, upper_bound: float)
            List of tuples that define the binning range of the standard deviation for each dimension separately.
            For example, if input data is given with only a few samples having high standard deviations,
            this might distort the calculations as the binning scheme commonly takes the (min, max) as the range
            for the binning, yielding a high amount of empty bins.

        Returns
        -------
        NumPy array of shape (d,)
            NumPy array with the ENCE for each input dimension separately.
        """

        assert kind in ['meanstd', 'cauchy', 'cumulative'], 'Parameter \'kind\' must be either \'meanstd\', or \'cumulative\'.'
        if kind == "meanstd":
            (mean, var), y, cov = meanvar(X, y)

            # check if correlated input is given
            if cov:
                raise RuntimeError("UCE is not defined for multivariate data with correlation.")

            mean = np.expand_dims(mean, axis=1) if mean.ndim == 1 else mean  # (n, d)
            var = np.expand_dims(var, axis=1) if var.ndim == 1 else var  # (n, d)

        # Cauchy distribution has no variance - ENCE is not applicable
        elif kind == "cauchy":

            n_dims = y.shape[1] if y.ndim == 2 else 1

            warnings.warn("ENCE is not applicable for Cauchy distributions.")
            return np.full(shape=(n_dims,), fill_value=float('nan'))

        else:

            # extract sampling points t and cumulative
            # get differences of cumulative and intermediate points of t
            t, cdf = X
            mean, var = cumulative_moments(t, cdf)  # (n, d) and (n, d)

        y = np.expand_dims(y, axis=1) if y.ndim == 1 else y
        std = np.sqrt(var)
        n_samples, n_dims = y.shape

        # squared error
        error = np.square(mean - y)  # (n, d)

        # prepare binning boundaries for regression
        bin_bounds = self._prepare_bins_regression(std, n_dims=n_dims, range_=range_)

        ence_means = []
        ence_stds = []
        for dim in range(n_dims):

            # perform binning over 1D stddev
            (mv_hist, mse_hist), n_samples, _, _ = self.binning(
                [bin_bounds[dim]],
                std[:, dim],
                var[:, dim],
                error[:, dim]
            )

            rmv_hist = np.sqrt(mv_hist)  # (b,)
            rmse_hist = np.sqrt(mse_hist)  # (b,)

            # ENCE for current dim is equally weighted (but normalized) mean over all bins
            ence_means.append(
                np.nanmean(
                    np.divide(
                        np.abs(rmv_hist - rmse_hist), rmv_hist,
                        out=np.full_like(rmv_hist, fill_value=float('nan')),
                        where=rmv_hist != 0
                    )
                )
            )
            ence_stds.append(
                np.nanstd(
                    np.divide(
                        np.abs(rmv_hist - rmse_hist), rmv_hist,
                        out=np.full_like(rmv_hist, fill_value=float('nan')),
                        where=rmv_hist != 0
                    )
                )
            )

        ence_means = np.array(ence_means).squeeze()
        ence_stds = np.array(ence_stds).squeeze()
        return ence_means, ence_stds


class UCE(_Miscalibration):
    """
    Uncertainty Calibration Error (UCE) for a regression calibration evaluation to test for
    *variance calibration*. A probabilistic regression model takes :math:`X` as input and outputs a
    mean :math:`\\mu_Y(X)` and a variance :math:`\\sigma_Y^2(X)` targeting the ground-truth :math:`y`.
    Similar to the :class:`netcal.metrics.confidence.ECE`, the UCE applies a binning scheme with :math:`B` bins
    over the predicted variance :math:`\\sigma_Y^2(X)` and measures the absolute difference
    between mean squared error (MSE) and mean variance (RMV) [1]_.
    Thus, the UCE [1]_ is defined by

    .. math::
        \\text{UCE} := \\sum^B_{b=1} \\frac{N_b}{N} |MSE(b) - MV(b)| ,

    where :math:`MSE(b)` and :math:`MV(b)` are the mean squared error and the mean variance within
    bin :math:`b`, respectively, and :math:`N_b` is the number of samples within bin :math:`b`.

    If multiple dimensions are given, the UCE is measured for each dimension separately.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the UCE binning.
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
    sample_threshold : int, optional, default: 1
        Bins with an amount of samples below this threshold are not included into the miscalibration metrics.

    References
    ----------
    .. [1] Laves, Max-Heinrich, et al.:
       "Well-calibrated regression uncertainty in medical imaging with deep learning."
       Medical Imaging with Deep Learning. PMLR, 2020.
       `Get source online <http://proceedings.mlr.press/v121/laves20a/laves20a.pdf>`__
    """

    @accepts((int, tuple, list), int)
    def __init__(
            self,
            bins: Union[int, Iterable[int]] = 10,
            sample_threshold: int = 1
    ):
        """ Constructor. For detailed parameter description, see class docs. """

        super().__init__(bins=bins, equal_intervals=True, detection=False, sample_threshold=sample_threshold)

    def measure(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            *,
            kind: str = 'meanstd',
            range_: List[Tuple[float, float]] = None
    ):
        """
        Measure quantile loss for given input data either as tuple consisting of mean and stddev estimates or as
        NumPy array consisting of a sample distribution. The loss is computed for several quantiles
        given by parameter q.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions).
        range_ : list of length d with tuples (lower_bound: float, upper_bound: float)
            List of tuples that define the binning range of the variance for each dimension separately.
            For example, if input data is given with only a few samples having high variance,
            this might distort the calculations as the binning scheme commonly takes the (min, max) as the range
            for the binning, yielding a high amount of empty bins.

        Returns
        -------
        NumPy array of shape (d,)
            NumPy array with the UCE for each input dimension separately.
        """

        assert kind in ['meanstd', 'cauchy', 'cumulative'], 'Parameter \'kind\' must be either \'meanstd\', or \'cumulative\'.'
        if kind == "meanstd":
            (mean, var), y, cov = meanvar(X, y)

            # check if correlated input is given
            if cov:
                raise RuntimeError("UCE is not defined for multivariate data with correlation.")

            mean = np.expand_dims(mean, axis=1) if mean.ndim == 1 else mean  # (n, d)
            var = np.expand_dims(var, axis=1) if var.ndim == 1 else var  # (n, d)

        # Cauchy distribution has no variance - ENCE is not applicable
        elif kind == "cauchy":

            n_dims = y.shape[1] if y.ndim == 2 else 1

            warnings.warn("UCE is not applicable for Cauchy distributions.")
            return np.full(shape=(n_dims,), fill_value=float('nan'))

        else:

            # extract sampling points t and cumulative
            # get differences of cumulative and intermediate points of t
            t, cdf = X
            mean, var = cumulative_moments(t, cdf)  # (n, d) and (n, d)

        y = np.expand_dims(y, axis=1) if y.ndim == 1 else y
        n_samples, n_dims = y.shape

        # squared error
        error = np.square(mean - y)  # (n, d)

        # prepare binning boundaries for regression
        bin_bounds = self._prepare_bins_regression(var, n_dims=n_dims, range_=range_)

        uce_means = []
        uce_stds = []
        for dim in range(n_dims):

            # perform binning over 1D variance
            (mv_hist, mse_hist), n_samples_hist, bounds, _ = self.binning(
                [bin_bounds[dim]],
                var[:, dim],
                var[:, dim],
                error[:, dim]
            )
            n_samples_hist = n_samples_hist / n_samples

            # UCE for current dim is weighted sum over all bins
            uce_means.append(
                np.sum(np.abs(mse_hist-mv_hist) * n_samples_hist) # np.mean
            )

            uce_stds.append(
                np.std(np.abs(mse_hist-mv_hist))
            )

        uce_means = np.array(uce_means).squeeze()
        uce_stds = np.array(uce_stds).squeeze()
        return uce_means, uce_stds


