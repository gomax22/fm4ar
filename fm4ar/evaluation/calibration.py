import os
import torch
import numpy as np
from pathlib import Path

from netcal.metrics.regression import NLL, PinballLoss, PICP, QCE
from netcal.presentation import ReliabilityRegression, ReliabilityQCE
from fm4ar.evaluation.netcal import ENCE, UCE, ReliabilityRegression, ReliabilityQCE

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
            }
        },
    }

    return calibration_metrics


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

