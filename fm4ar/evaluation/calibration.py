import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

from netcal.metrics.regression import NLL, PinballLoss, PICP, QCE
from netcal.presentation import ReliabilityRegression, ReliabilityQCE
from fm4ar.evaluation.netcal import ENCE, UCE, ReliabilityRegression, ReliabilityQCE

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from netcal import is_in_quantile
import tikzplotlib
mpl.rcParams['text.usetex'] = False 

def compute_calibration_metrics(
    posterior_samples: np.ndarray,
    thetas: np.ndarray,
    bins: int = 20,
    quantiles: np.ndarray = np.linspace(0.05, 0.95, 19)
) -> dict:
    """
    Evaluate regression calibration metrics using netcal.
    Args:
        posterior_samples: NumPy n-D array holding posterior samples of shape (n, m, d) with n samples, m posterior samples and d dimensions
        thetas: NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions
        bins: Number of bins to use for binning-based metrics (default: 20)
        quantiles: NumPy 1-D array holding the quantile levels to evaluate (default: np.linspace(0.05, 0.95, 19))
    Returns:
        calibration_metrics: A dictionary containing the computed calibration metrics.

    """

    # net:cal regression calibration metrics
    # mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
    # stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
    # ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions
    
    posterior_mean = np.mean(posterior_samples, axis=1)
    posterior_std = np.std(posterior_samples, axis=1)
    posterior_cov = np.array([np.cov(realizations, rowvar=False) for realizations in posterior_samples])

    univariate = (posterior_mean, posterior_std)
    multivariate = (posterior_mean, posterior_cov)


    # print(f"Mean (posterior): {posterior_mean.shape}")
    # print(f"Stddev (posterior): {posterior_std.shape}")
    # print(f"Cov (posterior): {posterior_cov.shape}")
    # print(f"Ground truth: {thetas.shape}")

    # regression 
    # Metrics for Regression Uncertainty Calibration
    # Methods for measuring miscalibration in the context of regression uncertainty calibration for probabilistic
    # regression models.

    # The common methods for regression uncertainty evaluation are *netcal.metrics.regression.PinballLoss* (Pinball
    # loss), the *netcal.metrics.regression.NLL* (NLL), and the *netcal.metrics.regression.QCE* (M-QCE and
    # C-QCE). The Pinball loss as well as the Marginal/Conditional Quantile Calibration Error (M-QCE and C-QCE) evaluate
    # the quality of the estimated quantiles compared to the observed ground-truth quantile coverage. The NLL is a proper
    # scoring rule to measure the overall quality of the predicted probability distributions.

    # Further metrics are the *netcal.metrics.regression.UCE* (UCE) and the *netcal.metrics.regression.ENCE*
    # (ENCE) which both apply a binning scheme over the predicted standard deviation/variance and test for *variance
    # calibration*.

    # For a detailed description of the available metrics within regression calibration, see the module doc of
    # *netcal.regression*.


    nll_loss = NLL()
    # Negative log likelihood (NLL) for probabilistic regression models.
    # If a probabilistic forecaster outputs a probability density function (PDF) :math:`f_Y(y)` targeting the ground-truth
    # :math:`y`, the negative log likelihood is defined by

    # .. math::
    #     \\text{NLL} = -\\sum^N_{n=1} \\log\\big(f_Y(y)\\big) ,

    # with :math:`N` as the number of samples within the examined data set.

    # **Note:** the input field for the standard deviation might also be given as quadratic NumPy arrays of shape
    # (n, d, d) with d dimensions. In this case, this method asserts covariance matrices as input
    # for each sample and the NLL is calculated for multivariate distributions.


    pinball_loss = PinballLoss()
    # Pinball aka quantile loss within regression calibration to test for *quantile calibration* of a probabilistic
    # regression model. The Pinball loss is an asymmetric loss that measures the quality of the predicted
    # quantiles. Given a probabilistic regression model that outputs a probability density function (PDF) :math:`f_Y(y)`
    # targeting the ground-truth :math:`y`, we further denote the cumulative as :math:`F_Y(y)` and the (inverse)
    # percent point function (PPF) as :math:`F_Y^{-1}(\\tau)` for a certain quantile level :math:`\\tau \\in [0, 1]`.

    # The Pinball loss is given by

    # .. math::
    #    L_{\\text{Pin}}(\\tau) =
    #    \\begin{cases}
    #         \\big( y-F_Y^{-1}(\\tau) \\big)\\tau \\quad &\\text{if } y \\geq F_Y^{-1}(\\tau)\\\\
    #         \\big( F_Y^{-1}(\\tau)-y \\big)(1-\\tau) \\quad &\\text{if } y < F_Y^{-1}(\\tau)
    #    \\end{cases} .
    # """



    qce_loss = QCE(bins=bins, marginal=True)  # if "marginal=False", we can also measure the QCE by means of the predicted variance levels (realized by binning the variance space)
    # Marginal Quantile Calibration Error (M-QCE) and Conditional Quantile Calibration Error (C-QCE) which both measure
    # the gap between predicted quantiles and observed quantile coverage also for multivariate distributions.
    # The M-QCE and C-QCE have originally been proposed by [1]_.
    # The derivation of both metrics are based on
    # the Normalized Estimation Error Squared (NEES) known from object tracking [2]_.
    # The derivation of both metrics is shown in the following.

    # **Definition of standard NEES:**
    # Given mean prediction :math:`\\hat{\\boldsymbol{y}} \\in \\mathbb{R}^M`, ground-truth
    # :math:`\\boldsymbol{y} \\in \\mathbb{R}^M`, and
    # estimated covariance matrix :math:`\\hat{\\boldsymbol{\\Sigma}} \\in \\mathbb{R}^{M \\times M}` using
    # :math:`M` dimensions, the NEES is defined as

    # .. math::
    #     \\epsilon = (\\boldsymbol{y} - \\hat{\\boldsymbol{y}})^\\top \\hat{\\boldsymbol{\\Sigma}}^{-1}
    #     (\\boldsymbol{y} - \\hat{\\boldsymbol{y}}) .

    # The average NEES is defined as the mean error over :math:`N` trials in a Monte-Carlo simulation for
    # Kalman-Filter testing, so that

    # .. math::
    #     \\bar{\\epsilon} = \\frac{1}{N} \\sum^N_{i=1} \\epsilon_i .

    # Under the condition, that :math:`\\mathbb{E}[\\boldsymbol{y} - \\hat{\\boldsymbol{y}}] = \\boldsymbol{0}` (zero mean),
    # a :math:`\\chi^2`-test is performed to evaluate the estimated uncertainty. This test is accepted, if

    # .. math::
    #     \\bar{\\epsilon} \\leq \\chi^2_M(\\tau),

    # where :math:`\\chi^2_M(\\tau)` is the PPF score obtained by a :math:`\\chi^2` distribution
    # with :math:`M` degrees of freedom, for a certain quantile level :math:`\\tau \\in [0, 1]`.

    # **Marginal Quantile Calibration Error (M-QCE):**
    # In the case of regression calibration testing, we are interested in the gap between predicted quantile levels and
    # observed quantile coverage probability for a certain set of quantile levels. We assert :math:`N` observations of our
    # test set that are used to estimate the NEES, so that we can compute the expected deviation between predicted
    # quantile level and observed quantile coverage by

    # .. math::
    #     \\text{M-QCE}(\\tau) := \\mathbb{E} \\Big[ \\big| \\mathbb{P} \\big( \\epsilon \\leq \\chi^2_M(\\tau) \\big) - \\tau \\big| \\Big] ,

    # which is the definition of the Marginal Quantile Calibration Error (M-QCE) [1]_.
    # The M-QCE is calculated by

    # .. math::
    #     \\text{M-QCE}(\\tau) = \\Bigg| \\frac{1}{N} \\sum^N_{n=1} \\mathbb{1} \\big( \\epsilon_n \\leq \\chi^2_M(\\tau) \\big) - \\tau \\Bigg|

    # **Conditional Quantile Calibration Error (C-QCE):**
    # The M-QCE measures the marginal calibration error which is more suitable to test for *quantile calibration*.
    # However, similar to :class:`netcal.metrics.regression.UCE` and :class:`netcal.metrics.regression.ENCE`,
    # we want to induce a dependency on the estimated covariance, since we require
    # that

    # .. math::
    #     &\\mathbb{E}[(\\boldsymbol{y} - \\hat{\\boldsymbol{y}})(\\boldsymbol{y} - \\hat{\\boldsymbol{y}})^\\top |
    #     \\hat{\\boldsymbol{\\Sigma}} = \\boldsymbol{\\Sigma}] = \\boldsymbol{\\Sigma},

    #     &\\forall \\boldsymbol{\\Sigma} \\in \\mathbb{R}^{M \\times M}, \\boldsymbol{\\Sigma} \\succcurlyeq 0,
    #     \\boldsymbol{\\Sigma}^\\top = \\boldsymbol{\\Sigma} .

    # To estimate the a *covariance* dependent QCE, we apply a binning scheme (similar to the
    # :class:`netcal.metrics.confidence.ECE`) over the square root of the *standardized generalized variance* (SGV) [3]_,
    # that is defined as

    # .. math::
    #     \\sigma_G = \\sqrt{\\text{det}(\\hat{\\boldsymbol{\\Sigma}})^{\\frac{1}{M}}} .

    # Using the generalized standard deviation, it is possible to get a summarized statistic across different
    # combinations of correlations to denote the distribution's dispersion. Thus, the Conditional Quantile Calibration
    # Error (C-QCE) [1]_ is defined by

    # .. math::
    #     \\text{C-QCE}(\\tau) := \\mathbb{E}_{\\sigma_G, X}\\Big[\\Big|\\mathbb{P}\\big(\\epsilon \\leq \\chi^2_M(\\tau) | \\sigma_G\\big) - \\tau \\Big|\\Big] ,

    # To approximate the expectation over the generalized standard deviation, we use a binning scheme with :math:`B` bins
    # (similar to the ECE) and :math:`N_b` samples per bin to compute the weighted sum across all bins, so that

    # .. math::
    #     \\text{C-QCE}(\\tau) \\approx \\sum^B_{b=1} \\frac{N_b}{N} | \\text{freq}(b) - \\tau |

    # where :math:`\\text{freq}(b)` is the coverage frequency within bin :math:`b` and given by

    # .. math::
    #     \\text{freq}(b) = \\frac{1}{N_b} \\sum_{n \\in \\mathcal{M}_b} \\mathbb{1}\\big(\\epsilon_i \\leq \\chi^2_M(\\tau)\\big) ,

    # with :math:`\\mathcal{M}_b` as the set of indices within bin :math:`b`.
    
    picp_loss = PICP(bins=bins)
    # Compute Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).
    # These metrics have been proposed by [1]_, [2]_.
    # This metric is used for Bayesian models to determine the quality of the uncertainty estimates.
    # In Bayesian mode, an uncertainty estimate is attached to each sample. The PICP measures the probability, that
    # the true (observed) accuracy falls into the p% prediction interval. The uncertainty is well-calibrated, if
    # the PICP is equal to p%. Simultaneously, the MPIW measures the mean width of all prediction intervals to evaluate
    # the sharpness of the uncertainty estimates.
    
    ence_loss = ENCE(bins=bins)
    # Expected Normalized Calibration Error (ENCE) for a regression calibration evaluation to test for
    # *variance calibration*. A probabilistic regression model takes :math:`X` as input and outputs a
    # mean :math:`\\mu_Y(X)` and a standard deviation :math:`\\sigma_Y(X)` targeting the ground-truth :math:`y`.
    # Similar to the :class:`netcal.metrics.confidence.ECE`, the ENCE applies a binning scheme with :math:`B` bins
    # over the predicted standard deviation :math:`\\sigma_Y(X)` and measures the absolute (normalized) difference
    # between root mean squared error (RMSE) and root mean variance (RMV) [1]_.
    # Thus, the ENCE [1]_ is defined by

    # .. math::
    #     \\text{ENCE} := \\frac{1}{B} \\sum^B_{b=1} \\frac{|RMSE(b) - RMV(b)|}{RMV(b)} ,

    # where :math:`RMSE(b)` and :math:`RMV(b)` are the root mean squared error and the root mean variance within
    # bin :math:`b`, respectively.

    # If multiple dimensions are given, the ENCE is measured for each dimension separately.

    uce_loss = UCE(bins=bins)
    # Uncertainty Calibration Error (UCE) for a regression calibration evaluation to test for
    # *variance calibration*. A probabilistic regression model takes :math:`X` as input and outputs a
    # mean :math:`\\mu_Y(X)` and a variance :math:`\\sigma_Y^2(X)` targeting the ground-truth :math:`y`.
    # Similar to the :class:`netcal.metrics.confidence.ECE`, the UCE applies a binning scheme with :math:`B` bins
    # over the predicted variance :math:`\\sigma_Y^2(X)` and measures the absolute difference
    # between mean squared error (MSE) and mean variance (RMV) [1]_.
    # Thus, the UCE [1]_ is defined by

    # .. math::
    #     \\text{UCE} := \\sum^B_{b=1} \\frac{N_b}{N} |MSE(b) - MV(b)| ,

    # where :math:`MSE(b)` and :math:`MV(b)` are the mean squared error and the mean variance within
    # bin :math:`b`, respectively, and :math:`N_b` is the number of samples within bin :math:`b`.

    # If multiple dimensions are given, the UCE is measured for each dimension separately.


    # univariate
    pinball_losses = pinball_loss.measure(univariate, thetas, q=quantiles, reduction="none")
    pinball_loss_mean = np.mean(
        pinball_losses,  # (q, n, d)
        axis=(0, 1)
    )  # (d,)
    pinball_loss_std = np.std(
        pinball_losses,  # (q, n, d)
        axis=(0, 1)
    )  # (d,)
    nll_losses = nll_loss.measure(univariate, thetas, reduction="none")
    nll_losses_mean = np.mean(nll_losses, axis=0)  # (d,)
    nll_losses_std = np.std(nll_losses, axis=0)  # (d,)


    qce_losses = qce_loss.measure(univariate, thetas, q=quantiles, reduction="none")
    qce_losses_mean = np.mean(qce_losses, axis=0)  # (q, d)
    qce_losses_std = np.std(qce_losses, axis=0)  # (q, d)


    picp_losses = picp_loss.measure(univariate, thetas, q=quantiles, reduction="none")
    picp_losses_mean = np.mean(picp_losses, axis=0)  # (2, d)
    picp_losses_std = np.std(picp_losses, axis=0)  # (2, d)

    # only univariate

    ence_loss_mean, ence_loss_stds = ence_loss.measure(univariate, thetas)
    uce_loss_mean, uce_loss_stds = uce_loss.measure(univariate, thetas)

    # multivariate
    nll_losses_joint = nll_loss.measure(multivariate, thetas, reduction="none")  # (n, d) or (d,)
    nll_losses_joint_mean = np.mean(
        nll_losses_joint,  # (n, d) or (d,)
        axis=0
    )

    nll_losses_joint_std = np.std(
        nll_losses_joint,  # (n, d) or (d,)
        axis=0
    )

    qce_losses_joint = qce_loss.measure(multivariate, thetas, q=quantiles, reduction="none")  # (q, n, d)
    qce_losses_joint_mean = np.mean(qce_losses_joint, axis=0)  # (q, d)
    qce_losses_joint_std = np.std(qce_losses_joint, axis=0)  # (q, d)


    # if nll_losses_joint_mean.size > 1:
    #     nll_losses_joint_mean = np.sum(nll_losses_joint_mean)
    #     nll_losses_joint_std = np.sum(nll_losses_joint_std)

    # if isinstance(qce_losses_joint_mean, np.ndarray) and qce_losses_joint_mean.size > 1:
    #     qce_losses_joint_mean = np.mean(qce_losses_joint_mean)
    #     qce_losses_joint_std = np.mean(qce_losses_joint_std)



    # print("################################")
    # print("Regression metrics:")
    # print(f"NLL (posterior) (independent): {nll_losses_mean} ± {nll_losses_std}")
    # print(f"PINBALL (posterior) (independent): {pinball_loss_mean} ± {pinball_loss_std}")
    # print(f"QCE (posterior) (independent): {qce_losses_mean} ± {qce_losses_std}")
    # print(f"PICP (posterior) (independent): {picp_losses_mean} ± {picp_losses_std}")
    # print(f"ENCE (posterior): {ence_loss_mean} ± {ence_loss_stds}")
    # print(f"UCE (posterior): {uce_loss_mean} ± {uce_loss_stds}")
    # print(f"ENCE (posterior) (average): {np.mean(ence_loss_mean)} ± {np.mean(ence_loss_stds)}")
    # print(f"UCE (posterior) (average): {np.mean(uce_loss_mean)} ± {np.mean(uce_loss_stds)}")
    # print(f"NLL (posterior) (joint): {nll_losses_joint_mean} ± {nll_losses_joint_std}")
    # print(f"QCE (posterior) (joint): {qce_losses_joint_mean} ± {qce_losses_joint_std}")
    # print("################################")

    calibration_metrics = {            
        "NLL": {
            "independent": {
                "mean": nll_losses_mean,
                "std": nll_losses_std,
            },
            "joint": {
                "mean": nll_losses_joint_mean,
                "std": nll_losses_joint_std,
            }
        },
        "QCE": {
            "independent": {
                "mean": qce_losses_mean,
                "std": qce_losses_std,
            },
            "joint": {
                "mean": qce_losses_joint_mean,
                "std": qce_losses_joint_std,
            },
        },
        "Pinball": {
            "mean": pinball_loss_mean,
            "std": pinball_loss_std,
        },
        "PICP": {
            "picp" : {
                "mean": picp_losses_mean[0],
                "std": picp_losses_std[0],
            },
            "mpiw" : {
                "mean": picp_losses_mean[1],
                "std": picp_losses_std[1],
            },
        },
        "ENCE": {
            "independent": {
                "mean": ence_loss_mean,
                "std": ence_loss_stds,

            },
            "joint": {
                "mean": np.mean(ence_loss_mean),
                "std": np.mean(ence_loss_stds),
            }
        },
        "UCE": {
            "independent": {
                "mean": uce_loss_mean,
                "std": uce_loss_stds,
            },
            "joint": {
                "mean": np.mean(uce_loss_mean),
                "std": np.mean(uce_loss_stds),
            },
        },
        "Sharpness": {
            "mean": np.var(posterior_samples, axis=1).mean(axis=0),  # mean variance across samples and dimensions
            "std": np.var(posterior_samples, axis=1).std(axis=0),    # std of variance across samples and dimensions

        },
    }

    return calibration_metrics


def save_calibration_metrics_to_csv(
    metrics: dict,
    output_dir: Path,
    labels: List[str],
) -> None:
    """
    Save calibration metrics to a CSV file.
    Args:
        calibration_metrics: A dictionary containing the computed calibration metrics.
        output_dir: Directory where the CSV file will be saved.
    """

    # Ensure the output directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize result DataFrame
    df = pd.DataFrame({"parameters": labels})

    # Recursive flattening
    for metric, metric_dict in metrics.items():
        for scope, scope_dict in metric_dict.items():
            # Case 1: has sub-dict (independent, joint, picp, mpiw)
            if isinstance(scope_dict, dict) and "mean" in scope_dict:
                # Scalar values → broadcast to all parameters
                if np.isscalar(scope_dict["mean"]):
                    mean_vals = [scope_dict["mean"]] * len(labels)
                    std_vals = [scope_dict["std"]] * len(labels)
                else:
                    mean_vals = scope_dict["mean"]
                    std_vals = scope_dict["std"]

                df[f"{metric}_{scope}_mean"] = mean_vals
                df[f"{metric}_{scope}_std"] = std_vals
            
            # Case 2: metrics like Pinball that skip "independent"/"joint" level
            elif scope in ["mean", "std"]:
                vals = metric_dict[scope]
                df[f"{metric}_{scope}"] = vals

    # Save to CSV
    df.to_csv(output_dir / "calibration_metrics_summary.csv", index=False)
    return None

def plot_calibration_diagrams(
    posterior_samples: np.ndarray,
    thetas: np.ndarray,
    output_dir: Path,
    bins: int = 20,
    quantiles: np.ndarray = np.linspace(0.05, 0.95, 19)
) -> None:
    """
    Plot regression calibration diagrams using netcal.
    Args:
        posterior_samples: NumPy n-D array holding posterior samples of shape (n, m, d) with n samples, m posterior samples and d dimensions
        thetas: NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions
        output_dir: output directory to save the calibration diagrams
        bins: number of bins for the calibration diagrams
        quantiles: quantile levels for the calibration diagrams
    Returns:
        None
    """
    # net:cal regression calibration metrics
    # mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
    # stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
    # ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    
    posterior_mean = np.mean(posterior_samples, axis=1)
    posterior_std = np.std(posterior_samples, axis=1)
    posterior_cov = np.array([np.cov(realizations, rowvar=False) for realizations in posterior_samples])

    univariate = (posterior_mean, posterior_std)
    multivariate = (posterior_mean, posterior_cov)

    # -------------------------------------------------
    # visualization

    # initialize the diagram object
    diagram = ReliabilityRegression(quantiles=bins + 1)
    # Reliability diagram in the scope of regression calibration for probabilistic regression models.
    # This diagram visualizes the quantile coverage frequency for several quantile levels and plots these observed
    # coverage scores above the desired quantile levels.
    # In this way, it is possible to compare the predicted and the observed quantile levels with each other.

    # This method is able to visualize the quantile coverage in terms of multiple univariate distributions if the input
    # is given as multiple independent Gaussians.
    # This method is also able to visualize the multivariate quantile coverage for a joint multivariate Gaussian if the
    # input is given with covariance matrices.

    diagram.plot(univariate, thetas, filename=os.path.join(output_dir, "rr_independent.png")) # independent 
    diagram.plot(multivariate, thetas, filename=os.path.join(output_dir, "rr_joint.png")) # joint



    diagram = ReliabilityQCE(bins=bins)
    # Visualizes the Conditional Quantile Calibration Error (C-QCE) in the scope of regression calibration as a bar chart
    # for probabilistic regression models.
    # See :class:`netcal.metrics.regression.QCE` for a detailed documentation of the C-QCE [1]_.
    # This method is able to visualize the C-QCE in terms of multiple univariate distributions if the input is given
    # as multiple independent Gaussians.
    # This method is also able to visualize the multivariate C-QCE for a multivariate Gaussian if the input is given
    # with covariance matrices.

    diagram.plot(univariate, thetas, q=quantiles, filename=os.path.join(output_dir, "qce_independent.png")) # independent
    diagram.plot(multivariate, thetas, q=quantiles, filename=os.path.join(output_dir, "qce_joint.png")) # joint


# extended from netcal.presentation.ReliabilityRegression
class RelibilityRegressions(ReliabilityRegression):
    def __init__(self, quantiles = 11):
        super().__init__(quantiles)

    def plots(
        self,
        X: Union[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]],
        y: np.ndarray,
        colors: List[str],
        model_labels: List[str],
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
        in_quantiles = []
        for x in X:
            in_quantile, _, _, _, _ = is_in_quantile(x, y, self.quantiles, kind) # (q, n, [d]), (q, n, d), (n, d), (n, d, [d])
            in_quantiles.append(in_quantile)


        # get the frequency of which y is within the quantile bounds
        frequencies = []
        for in_quantile in in_quantiles:
            frequency = np.mean(in_quantile, axis=1) # (q, [d])
            frequencies.append(frequency)

        # make frequency array at least 2d
        for idx, frequency in enumerate(frequencies):
            if frequency.ndim == 1:
                frequencies[idx] = np.expand_dims(frequency, axis=1)  # (q, d) or (q, 1)

        n_dims = frequencies[0].shape[-1]

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

            for frequency, color, model_label in zip(frequencies, colors, model_labels):
                ax.plot(self.quantiles, frequency[:, dim], "o-", color=color, label=model_label, alpha=0.5)

            # draw diagonal as perfect calibration line
            ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Perfect Calibration')
            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.0, 1.0))

            # labels and legend of second plot
            ax.set_xlabel('Expected quantile')
            ax.set_ylabel('Observed frequency')
            # ax.legend(model_labels + ['Perfect Calibration'], loc='upper left')
            handles = [Line2D([0], [0], lw=1.0, ls="--", color="black")]
            handles += [
                Line2D([0], [0], color=c, lw=1.0)
                for c in colors
            ]
            ax.legend(
                handles=handles,
                labels=["Perfect Calibration"] + [label for label in model_labels],
                ncols=1,
                # frameon=False,
                loc="center right",
                fontsize=10,
                bbox_to_anchor=(1.5, 0.5),
            )
            
            ax.grid(True)

            # set axis title
            title = 'Reliability Regression Diagram'
            if title_suffix is not None:
                title = title + ' - ' + title_suffix
            if feature_names is not None:
                title = title + ' - ' + feature_names[dim]
            else:
                title = title + ' - dim %02d' % dim

            # ax.set_title(title)
        fig.tight_layout()

        # if tikz is true, create tikz code from matplotlib figure
        if tikz:

            # get tikz code for our specific figure and also pass filename to store possible bitmaps
            tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=filename, **save_args)

            # close matplotlib figure when tikz figure is requested to save memory
            plt.close(fig)
            fig = tikz_fig

        # save figure either as matplotlib PNG or as tikz output file
        if filename is not None:
            if tikz:
                with open(filename, "w") as open_file:
                    open_file.write(fig)
            else:
                fig.savefig(filename, **save_args)

        return fig





def plot_reliability_regression_diagrams(
    univariates: List[Tuple[np.ndarray, np.ndarray]],
    multivariates: List[Tuple[np.ndarray, np.ndarray]],
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: str,
    output_dir: str,
    bins: int = 20,
    ):

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_targets = thetas.shape[-1]

    # -------------------------------------------------
    # visualization
     # initialize the diagram object
    diagram = RelibilityRegressions(quantiles=bins + 1)
    # Reliability diagram in the scope of regression calibration for probabilistic regression models.
    # This diagram visualizes the quantile coverage frequency for several quantile levels and plots these observed
    # coverage scores above the desired quantile levels.
    # In this way, it is possible to compare the predicted and the observed quantile levels with each other.

    # This method is able to visualize the quantile coverage in terms of multiple univariate distributions if the input
    # is given as multiple independent Gaussians.
    # This method is also able to visualize the multivariate quantile coverage for a joint multivariate Gaussian if the
    # input is given with covariance matrices.
    fig = diagram.plots(univariates, thetas, feature_names=labels, colors=colors, model_labels=model_labels)
    # ax = plt.gca()
    # ax.legend(model_labels, loc='upper left')
    
    fig.savefig(os.path.join(output_dir, "rr_independent.png"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(output_dir, "rr_independent.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    fig = diagram.plots(multivariates, thetas, colors=colors, model_labels=model_labels, feature_names=["Aggregated dimensions"])
    # ax = plt.gca()
    # ax.legend(model_labels, loc='upper left')
    fig.savefig(os.path.join(output_dir, "rr_joint.png"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(output_dir, "rr_joint.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)



def plot_qce_diagrams(
    univariates: List[Tuple[np.ndarray, np.ndarray]],
    multivariates: List[Tuple[np.ndarray, np.ndarray]],
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: str,
    output_dir: str,
    bins: int = 20,
    quantiles: np.ndarray = np.linspace(0.05, 0.95, 19) 
):
    
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_targets = thetas.shape[-1]

    diagram = ReliabilityQCE(bins=bins)
    # Visualizes the Conditional Quantile Calibration Error (C-QCE) in the scope of regression calibration as a bar chart
    # for probabilistic regression models.
    # See :class:`netcal.metrics.regression.QCE` for a detailed documentation of the C-QCE [1]_.
    # This method is able to visualize the C-QCE in terms of multiple univariate distributions if the input is given
    # as multiple independent Gaussians.
    # This method is also able to visualize the multivariate C-QCE for a multivariate Gaussian if the input is given
    # with covariance matrices.
    fig = diagram.plot(univariates[0], thetas, feature_names=labels, q=quantiles, colors=colors[0])
    for univariate, color in zip(univariates[1:], colors[1:]):
        diagram.plot(univariate, thetas, feature_names=labels, q=quantiles, colors=color, fig=fig)
    
    ax = fig.gca()
    ax.legend(model_labels)
    fig.savefig(os.path.join(output_dir, "qce_independent.png"))
    plt.close(fig)

    fig = diagram.plot(multivariates[0], thetas, feature_names=labels, q=quantiles, colors=colors[0])
    for multivariate, color in zip(multivariates[1:], colors[1:]):
        diagram.plot(multivariate, thetas, feature_names=labels, q=quantiles, colors=color, fig=fig)
    
    ax = fig.gca()
    ax.legend(model_labels)
    fig.savefig(os.path.join(output_dir, "qce_joint.png"))
    plt.close(fig)


def plot_regression_diagrams(
    posteriors: np.ndarray,
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: str,
    output_dir: str
    ):

    # net:cal regression calibration metrics
    # mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
    # stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
    # ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    posterior_means = [np.mean(posterior_samples, axis=1) for posterior_samples in posteriors]
    posterior_stds = [np.std(posterior_samples, axis=1) for posterior_samples in posteriors]
    posterior_covs = [np.array([np.cov(realizations, rowvar=False) for realizations in posterior_samples]) for posterior_samples in posteriors]

    univariates = [(mean, std) for mean, std in zip(posterior_means, posterior_stds)]
    multivariates = [(mean, cov) for mean, cov in zip(posterior_means, posterior_covs)]

    plot_reliability_regression_diagrams(
        univariates=univariates,
        multivariates=multivariates,
        thetas=thetas,
        labels=labels,
        colors=colors,
        model_labels=model_labels,
        output_dir=output_dir
    )

    # plot_qce_diagrams(
    #     univariates=univariates,
    #     multivariates=multivariates,
    #     thetas=thetas,
    #     labels=labels,
    #     colors=colors,
    #     model_labels=model_labels,
    #     output_dir=output_dir
    # )




