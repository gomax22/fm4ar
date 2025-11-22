"""
Methods for flow matching posterior estimation (FMPE) models.
"""

from typing import Any

import torch
from torch import nn as nn

from fm4ar.models.base import Base
from fm4ar.nn.embedding_nets import create_embedding_net
from fm4ar.nn.vectorfield_nets import create_vectorfield_net
from fm4ar.torchutils.general import set_random_seed
from fm4ar.torchutils.odeint import odeint_nfe as odeint
from fm4ar.utils.shapes import validate_dims
from fm4ar.time_priors import get_time_prior
from fm4ar.utils.nfe import NFEProfiler

from torchdiffeq._impl.odeint import SOLVERS
from torchdiffeq._impl.solvers import FixedGridODESolver, AdaptiveStepsizeODESolver

class FMPEModel(Base):
    """
    Wrapper class for FMPE models. This class provides all the methods
    for training, inference, ... around the actual neural network(s);
    for example, the loss function or methods to sample from the model.

    This class is usually instantiated by the `build_model()` function,
    which handles the configuration from the config file.
    """

    # Add type hint for the network
    network: "FMPENetwork"

    def __init__(self, **kwargs: Any):

        super().__init__(**kwargs)

        model_config = self.config["model"]

        self.sigma_min = model_config["sigma_min"]

        time_prior_config = model_config["time_prior"]
        time_prior_config["kwargs"]["device"] = self.device

        self.time_prior_distribution = get_time_prior(time_prior_config)
        self.dim_theta = model_config["dim_theta"]

        self.profiler = NFEProfiler()
        

    def evaluate_vectorfield(
        self,
        t: float | torch.Tensor,
        theta_t: torch.Tensor,
        context: dict[str, torch.Tensor],
        aux_data: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Evaluate the vectorfield at the given time and parameter values.

        Args:
            t: The time at which to evaluate the vectorfield (must be
                a float between 0 and 1). Note: This can also be a
                tensor in case we are evaluating the vectorfield for
                a batch.
            theta_t: The parameter values at which to evaluate the
                vectorfield.
            context: Context (i.e., observed data).

        Returns:
            The vectorfield evaluated at the given time and parameter.
        """

        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=theta_t.device)

        return self.network(  # type: ignore
            t=t,
            theta=theta_t,
            context=context,
            aux_data=aux_data,
        )

    def initialize_network(self) -> None:
        """
        Initialize the neural net that parameterizes the vectorfield.
        """

        # Fix the random seed for reproducibility
        set_random_seed(seed=self.random_seed, verbose=False)

        # Create the FMPE network
        self.network = create_fmpe_network(model_config=self.config["model"])

    @property
    def integration_range(self) -> torch.Tensor:
        """
        Integration range for ODE solver: [0, 1].
        """

        return torch.tensor([0.0, 1.0], device=self.device).float()

    def log_prob_batch(
        self,
        theta: torch.Tensor,
        context: dict[str, torch.Tensor],
        aux_data: torch.Tensor = None,
        tolerance: float = 1e-7,
        method: str = "dopri5",
        num_steps: int = 100,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Evaluates log_probs of theta conditional on provided context.
        For this we solve an ODE backwards in time until we reach the
        initial pure noise distribution.

        There are two contributions, the log_prob of theta_0 (which is
        uniquely determined by theta) under the base distribution, and
        the integrated divergence of the vectorfield.

        Args:
            theta: Parameter values for which to evaluate the log_prob.
            context: Context (i.e., observed data).
            aux_data: Auxiliary data (e.g., planetary system data).
            tolerance: Tolerance (atol and rtol) for adaptive ODE solvers.
            method: ODE solver method. Default is "dopri5".
            num_steps: Number of steps for the ODE solver (only used
                for fixed-step methods).
            epsilon: Small constant to avoid numerical issues with
                fixed-step ODE solvers.

        Returns:
            The log probability of `theta`.
        """

        self.network.eval()

        div_init = torch.zeros(
            (theta.shape[0],), device=theta.device
        ).unsqueeze(1)
        theta_and_div_init = torch.cat((theta, div_init), dim=1)

        # Detect if we are using a fixed-step method
        t_values: torch.Tensor
        if issubclass(SOLVERS[method], FixedGridODESolver):
            if num_steps < 2:
                raise ValueError(
                    "num_steps must be at least 2 for fixed-step ODE solvers."
                )
            
            # In fixed-step case, avoid the boundary points
            # to prevent numerical issues
            t_values = torch.linspace(
                self.integration_range[0] + epsilon,
                self.integration_range[1] - epsilon,
                steps=num_steps,
                device=self.device,
            )
        # Otherwise, use the standard integration range
        # for adaptive-step methods
        else:
            t_values = self.integration_range


        # Integrate backwards in time to get from theta_1 to theta_0;
        # note the `flip()` of the integration range
        theta_and_div_0 = odeint(
            func=lambda t, theta_and_div_t: self.rhs_of_joint_ode(
                t, theta_and_div_t, context, aux_data
            ),
            y0=theta_and_div_init,
            t=torch.flip(t_values, dims=(0,)),
            profiler=self.profiler,
            atol=tolerance,
            rtol=tolerance,
            method=method,
        )[-1]

        theta_0 = theta_and_div_0[:, :-1]
        divergence = theta_and_div_0[:, -1]
        log_prior = self.compute_log_prior(theta_0)

        return (log_prior - divergence).detach()  # type: ignore

    def loss(
        self,
        theta: torch.Tensor,
        context: dict[str, torch.Tensor],
        aux_data: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculates loss as the the mean squared error between the
        predicted vectorfield and the vector field for transporting the
        parameter data to samples from the prior.

        Args:
            theta: Parameters.
            context: Context (i.e., observed data).
            aux_data: Auxiliary data (e.g., planetary system data).

        Returns:
            Loss tensor.
        """

        # # Get the time prior exponent from the kwargs, if provided. This can
        # # be used, e.g., to increase the time prior exponent during training.
        # time_prior_exponent = kwargs.get("time_prior_exponent", None)

        # Sample a time t and some starting parameters theta_0
        t = self.time_prior_distribution.sample_t(
            num_samples=len(theta),
        )
        theta_0 = self.sample_theta_0(num_samples=len(theta))

        # Use optimal transport path to interpolate theta_t between theta_0
        # (starting values) and theta_1 (target parameters)
        theta_t = self.ot_conditional_flow(
            theta_0=theta_0,
            theta_1=theta,
            t=t,
            sigma_min=self.sigma_min,
        )

        # Compute the true vectorfield and the predicted vectorfield
        true_vf = theta - (1 - self.sigma_min) * theta_0
        pred_vf = self.network(
            t=t, 
            theta=theta_t, 
            context=context, 
            aux_data=aux_data
        )

        # Calculate loss as MSE between the true and predicted vectorfield
        loss = torch.nn.functional.mse_loss(pred_vf, true_vf)

        return loss

    def rhs_of_joint_ode(
        self,
        t: float,
        theta_and_div_t: torch.Tensor,
        context: dict[str, torch.Tensor],
        aux_data: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns the right hand side of the neural ODE that is used to
        evaluate the log_prob of theta samples. This is a joint ODE over
        the vectorfield and the divergence. By integrating this ODE, one
        can simultaneously trace the parameter sample theta_t and
        integrate the divergence contribution to the log_prob, see e.g.,
        https://arxiv.org/abs/1806.07366 or Appendix C in
        https://arxiv.org/abs/2210.02747.

        Args:
            t: Time (controls the noise level).
            theta_and_div_t: Concatenated tensor of `(theta_t, div)`.
            context: Context (i.e., observed data).
            aux_data: Auxiliary data (e.g., planetary system data).

        Returns:
            The vector field that generates the continuous flow, plus
            its divergence (required for likelihood evaluation).
        """

        theta_t = theta_and_div_t[:, :-1]  # extract theta_t
        with torch.enable_grad():  # type: ignore
            theta_t.requires_grad_(True)
            vf = self.evaluate_vectorfield(t, theta_t, context, aux_data)
            div_vf = self.compute_divergence(vf=vf, theta_t=theta_t)
        return torch.cat((vf, -div_vf), dim=1)

    def sample_and_log_prob_batch(
        self,
        context: dict[str, torch.Tensor],
        aux_data: torch.Tensor = None,
        num_samples: int = 1,
        tolerance: float = 1e-7,
        method: str = "dopri5",
        num_steps: int = 100,
        epsilon: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw posterior samples and return them together with their log-
        probability (or rather, the log-density under the model.

        Sampling and evaluating the log-probability can be achieved by
        solving a joint ODE forwards in time, which is more efficient
        than first drawing samples and evaluating the log-probability
        separately:
        If d/dt [phi(t), f(t)] = rhs joint with initial conditions
        [theta_0, log p(theta_0)], where theta_0 ~ p_0(theta_0), then
        [phi(1), f(1)] = [theta_1, log p(theta_0) + log p_1(theta_1) -
        log p(theta_0)] = [theta_1, log p_1(theta_1)].

        Args:
            context: Context (i.e., observed data).
            aux_data: Auxiliary data (e.g., planetary system data).
            num_samples: Number of posterior samples to generate in the
                unconditional case. If `context` is provided, the number
                of samples is automatically determined from the context.
            tolerance: Tolerance (atol and rtol) for adaptive ODE solvers.
            method: ODE solver method. Default is "dopri5".
            num_steps: Number of steps for the ODE solver (only used
                for fixed-step methods).
            epsilon: Small constant to avoid numerical issues with
                fixed-step ODE solvers.


        Returns:
            The generated samples and their log probabilities.
        """

        self.network.eval()

        # Get the number of samples, either from context or explicitly
        num_samples = (
            context["flux"].shape[0] if context is not None else num_samples
        )

        theta_0 = self.sample_theta_0(num_samples)
        log_prior = self.compute_log_prior(theta_0)
        theta_and_div_init = torch.cat(
            (theta_0, log_prior.unsqueeze(1)), dim=1
        )
        
        # Detect if we are using a fixed-step method
        # If so, create a fixed grid of time points
        t_values: torch.Tensor
        if issubclass(SOLVERS[method], FixedGridODESolver):
            if num_steps < 2:
                raise ValueError(
                    "num_steps must be at least 2 for fixed-step ODE solvers."
                )

            # In fixed-step case, avoid the boundary points
            # to prevent numerical issues
            t_values = torch.linspace(
                self.integration_range[0] + epsilon,
                self.integration_range[1] - epsilon,
                steps=num_steps,
                device=self.device,
            )
        # Otherwise, use the standard integration range
        # for adaptive-step methods
        else:
            t_values = self.integration_range

        # Integrate forwards in time to get from theta_0 to theta_1
        theta_and_div_1 = odeint(
            func=lambda t, theta_and_div_t: self.rhs_of_joint_ode(
                t, theta_and_div_t, context, aux_data   
            ),
            y0=theta_and_div_init,
            t=t_values,
            profiler=self.profiler,
            atol=tolerance,
            rtol=tolerance,
            method=method,
        )[-1]

        theta_1, log_prob_1 = theta_and_div_1[:, :-1], theta_and_div_1[:, -1]

        return theta_1, log_prob_1

    def sample_batch(
        self,
        context: dict[str, torch.Tensor],
        aux_data: torch.Tensor = None,
        num_samples: int = 1,
        tolerance: float = 1e-7,
        method: str = "dopri5",
        num_steps: int = 100,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Returns (conditional) samples for a batch of contexts by solving
        an ODE forwards in time.

        Args:
            context: Context (i.e., observed data).
            aux_data: Auxiliary data (e.g., planetary system data).
            num_samples: Number of posterior samples to generate in the
                unconditional case. If `context` is provided, the number
                of samples is automatically determined from the context.
            tolerance: Tolerance (atol and rtol) for adaptive ODE solvers.
            method: ODE solver method. Default is "dopri5".
            num_steps: Number of steps for the ODE solver (only used
                for fixed-step methods).
            epsilon: Small constant to avoid numerical issues with
                fixed-step ODE solvers.

        Returns:
            The generated samples.
        """

        self.network.eval()

        # Get the number of samples, either from context or explicitly
        num_samples = (
            context["flux"].shape[0] if context is not None else num_samples
        )

        # Detect if we are using a fixed-step method
        # If so, create a fixed grid of time points
        t_values: torch.Tensor
        if issubclass(SOLVERS[method], FixedGridODESolver):
            if num_steps < 2:
                raise ValueError(
                    "num_steps must be at least 2 for fixed-step ODE solvers."
                )
            
            # In fixed-step case, avoid the boundary points
            # to prevent numerical issues
            t_values = torch.linspace(
                self.integration_range[0] + epsilon,
                self.integration_range[1] - epsilon,
                steps=num_steps,
                device=self.device,
            )
        # Otherwise, use the standard integration range
        # for adaptive-step methods
        else:
            t_values = self.integration_range

        # Solve ODE forwards in time to get from theta_0 to theta_1
        with torch.no_grad():
            theta_0 = self.sample_theta_0(num_samples)
            theta_1 = odeint(
                func=lambda t, theta_t: self.evaluate_vectorfield(
                    t, theta_t, context, aux_data
                ),
                y0=theta_0,
                t=t_values,
                profiler=self.profiler,
                atol=tolerance,
                rtol=tolerance,
                method=method,
            )[-1]

        return torch.Tensor(theta_1)

    def sample_theta_0(self, num_samples: int) -> torch.Tensor:
        """
        Sample `theta_0` from a standard Gaussian prior.
        """

        return torch.randn(num_samples, self.dim_theta, device=self.device)

    @staticmethod
    def compute_divergence(
        vf: torch.Tensor,
        theta_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the divergence.

        Note: Enabling JIT compilation seems to break because at the
            `enable_grad()` statement.
        """

        div: float | torch.Tensor = 0.0
        with torch.enable_grad():  # type: ignore
            vf.requires_grad_(True)
            theta_t.requires_grad_(True)
            for i in range(vf.shape[-1]):
                div += torch.autograd.grad(
                    vf[..., i],
                    theta_t,
                    torch.ones_like(vf[..., i]),
                    create_graph=True,
                )[0][..., i : i + 1]

        return div  # type: ignore

    @staticmethod
    def compute_log_prior(theta_0: torch.Tensor) -> torch.Tensor:
        """
        Log prior value of theta_0 under the Gaussian base distribution.
        """

        # We don't use self.dim_theta here to keep the method JIT-able
        dim_theta = theta_0.shape[1]
        log_of_2pi = 1.8378770664093

        return (  # type: ignore
            -dim_theta * log_of_2pi - torch.sum(theta_0**2, dim=1)
        ) / 2.0

    @staticmethod
    def ot_conditional_flow(
        theta_0: torch.Tensor,
        theta_1: torch.Tensor,
        t: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        """
        Optimal transport for the conditional flow.

        This function basically interpolates between theta_0 and theta_1.

        Args:
            theta_0: Parameters at t=0.
            theta_1: Parameters at t=1.
            t: Time at which to evaluate the flow.
            sigma_min: Standard deviation of the target distribution.

        Returns:
            The parameters at time t.
        """

        return (  # type: ignore
            (1 - (1 - sigma_min) * t)[:, None] * theta_0 + t[:, None] * theta_1
        )


class FMPENetwork(nn.Module):
    """
    Wrapper around the neural networks for an FMPE model. This combines
    the embedding networks for the `context` and `(t, theta)` with the
    actual network that predicts the vectorfield.

    This class is usually instantiated by the `create_fmpe_network()`
    function, which handles the configuration from the config file.
    """

    def __init__(
        self,
        vectorfield_net: nn.Module,
        context_embedding_net: nn.Module,
        t_theta_embedding_net: nn.Module,
        auxiliary_data_embedding_net: nn.Module,
        context_with_glu: bool,
        t_theta_with_glu: bool,
        auxiliary_data_with_glu: bool,
    ) -> None:
        """
        Instantiate a new `FMPENetwork`.

        Args:
            vectorfield_net: The network that learns the vector field.
            context_embedding_net: The context embedding network.
            t_theta_embedding_net: The (t, theta) embedding network.
            auxiliary_data_embedding_net: The auxiliary data embedding network.
            context_with_glu: Whether to use a gated linear unit (GLU)
                for the context embedding.
            t_theta_with_glu: Whether to use a gated linear unit (GLU)
                for the (t, theta) embedding.
            auxiliary_data_with_glu: Whether to use a gated linear unit (GLU)
                for the auxiliary data embedding.
        """

        super(FMPENetwork, self).__init__()

        self.vectorfield_net = vectorfield_net
        self.context_embedding_net = context_embedding_net
        self.t_theta_embedding_net = t_theta_embedding_net
        self.auxiliary_data_embedding_net = auxiliary_data_embedding_net

        self.context_with_glu = context_with_glu
        self.t_theta_with_glu = t_theta_with_glu
        self.auxiliary_data_with_glu = auxiliary_data_with_glu

    def get_context_embedding(
        self,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get the embedding of the context. We wrap this in a separate
        method to allow caching the last result. Since the context is
        a dictionary, we need to convert it to a `frozendict` first to
        allow caching with the `lru_cache` decorator.
        """

        return self.context_embedding_net(context)  # type: ignore

    def forward(
        self,
        t: torch.Tensor,
        theta: torch.Tensor,  # note: this is theta _at time t_
        context: dict[str, torch.Tensor],
        aux_data: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through the continuous flow model, that is, compute
        the embeddings of the `context` and `(t, theta)` and predict the
        vector field.
        """

        # Concatenate `t` and `theta` and embed the result
        t_theta = torch.cat((t.unsqueeze(1), theta), dim=1)
        t_theta_embedding = self.t_theta_embedding_net(t_theta)
        validate_dims(t_theta_embedding, 2)

        # Get embedding of auxiliary data
        if isinstance(self.auxiliary_data_embedding_net, nn.Identity):
            aux_data = None
            auxiliary_data_embedding = None
        else:
            auxiliary_data_embedding = self.auxiliary_data_embedding_net(aux_data)
            validate_dims(auxiliary_data_embedding, 2)

        # Get the embedding of the context
        context_embedding = self.get_context_embedding(context)
        validate_dims(context_embedding, 2)

        # Prepare GLU contexts
        # Standard Modeling - context_embedding as main input, aux_data_embedding and t_theta_embedding as GLU conditioning
        # Alternative Modeling - t_theta_embedding as main input, context_embedding and aux_data_embedding as GLU conditioning
        # Old Modeling - context_embedding as main input, t_theta_embedding and aux_data_embedding (concatenated) as GLU conditioning

        if not self.context_with_glu and self.auxiliary_data_with_glu and self.t_theta_with_glu:
            # Standard Modeling
            first_glu_context = auxiliary_data_embedding
            second_glu_context = t_theta_embedding
            cf_input = context_embedding
        elif not self.context_with_glu and not self.auxiliary_data_with_glu and self.t_theta_with_glu:
            # Standard Modeling - without auxiliary data
            first_glu_context = None
            second_glu_context = t_theta_embedding
            cf_input = context_embedding
        elif self.context_with_glu and not self.auxiliary_data_with_glu and not self.t_theta_with_glu:
            # Standard Modeling - without auxiliary data
            first_glu_context = None
            second_glu_context = context_embedding
            cf_input = t_theta_embedding
        elif self.context_with_glu and self.auxiliary_data_with_glu and not self.t_theta_with_glu:
            # Standard Modeling - with auxiliary data
            first_glu_context = auxiliary_data_embedding
            second_glu_context = context_embedding
            cf_input = t_theta_embedding
        elif not self.context_with_glu and not self.auxiliary_data_with_glu and not self.t_theta_with_glu:
            # Old Modeling - w/ or w/o auxiliary data
            first_glu_context = None
            second_glu_context = torch.cat([t_theta_embedding, auxiliary_data_embedding], dim=1) \
                if auxiliary_data_embedding is not None else t_theta_embedding
            cf_input = context_embedding
        else:
            raise NotImplementedError("The specified GLU configuration is not supported.")
        
        return torch.Tensor(
            self.vectorfield_net(
                x=cf_input, 
                first_context=first_glu_context,
                second_context=second_glu_context
                )
            )


def create_fmpe_network(model_config: dict) -> FMPENetwork:
    """
    Create a `FMPENetwork` instance from the given model configuration.

    Args:
        model_config: Keyword arguments specifying the model, that is,
            the "model" section of the configuration file.

    Returns:
        The FMPE network wrapper.
    """

    # Extract dimensions of `theta` and `context`
    # Note: These are *not* in the config file, but are added by the
    # `prepare_new()` method that is called at the start of training.
    dim_theta = int(model_config["dim_theta"])
    dim_context = int(model_config["dim_context"])
    dim_auxiliary_data = int(model_config["dim_auxiliary_data"]) if "dim_auxiliary_data" in list(model_config.keys()) else 0

    # Check if we use GLU for embedded `t_theta` and / or `context`
    t_theta_with_glu = model_config.get("t_theta_with_glu", False)
    context_with_glu = model_config.get("context_with_glu", False)
    auxiliary_data_with_glu = model_config.get("auxiliary_data_with_glu", False)

    # Construct an embedding network for the context
    context_embedding_net, dim_embedded_context = create_embedding_net(
        input_shape=(dim_context,),
        block_configs=model_config["context_embedding_net"],
        supports_dict_input=True,
    )

    # Construct an embedding network for `(t, theta)`
    t_theta_embedding_net, dim_embedded_t_theta = create_embedding_net(
        input_shape=(dim_theta + 1,),
        block_configs=model_config["t_theta_embedding_net"],
    )
    
    # Construct an embedding network for auxiliary data
    auxiliary_data_embedding_net, dim_embedded_auxiliary_data = create_embedding_net(
        input_shape=(dim_auxiliary_data,),
        block_configs=model_config["auxiliary_data_embedding_net"],
    ) if "auxiliary_data_embedding_net" in list(model_config.keys()) else (nn.Identity(), 0)

    # Compute GLU dimensions and input dimension for continuous flow network
    # Prepare dimensions of GLU contexts
    # Standard Modeling - context_embedding as main input, aux_data_embedding and t_theta_embedding as GLU conditioning
    # Alternative Modeling - t_theta_embedding as main input, context_embedding and aux_data_embedding as GLU conditioning
    # Old Modeling - context_embedding as main input, t_theta_embedding and aux_data_embedding (concatenated) as GLU conditioning

    if not context_with_glu and t_theta_with_glu:
        # Standard Modeling
        dim_first_glu = (
            auxiliary_data_with_glu * dim_embedded_auxiliary_data
            + context_with_glu * dim_embedded_context
        )
        dim_second_glu = (
            t_theta_with_glu * dim_embedded_t_theta
            + context_with_glu * dim_embedded_context
        )
    elif context_with_glu and not t_theta_with_glu:
        # Alternative Modeling
        dim_first_glu = (
            auxiliary_data_with_glu * dim_embedded_auxiliary_data
            + t_theta_with_glu * dim_embedded_t_theta
        )
        dim_second_glu = (
            context_with_glu * dim_embedded_context
            + t_theta_with_glu * dim_embedded_t_theta
        )
    elif not context_with_glu and not t_theta_with_glu:
        # Old Modeling - w/ or w/o auxiliary data
        dim_first_glu = 0

        dim_second_glu = (
            t_theta_with_glu * dim_embedded_t_theta
            + auxiliary_data_with_glu * dim_embedded_auxiliary_data
            + context_with_glu * dim_embedded_context
        )
    else:
        raise NotImplementedError("The specified GLU configuration is not supported.")

    dim_input = dim_embedded_context + dim_embedded_auxiliary_data + dim_embedded_t_theta - dim_first_glu - dim_second_glu
    dim_first_glu = dim_first_glu if dim_first_glu > 0 else None
    dim_second_glu = dim_second_glu if dim_second_glu > 0 else None

    # Construct the continuous flow network
    vectorfield_net = create_vectorfield_net(
        dim_input=dim_input,
        dim_first_glu=dim_first_glu,
        dim_second_glu=dim_second_glu,
        dim_output=dim_theta,
        network_config=model_config["vectorfield_net"],
    )
    # Bundle everything into a `FMPENetwork` wrapper
    network = FMPENetwork(
        vectorfield_net=vectorfield_net,
        context_embedding_net=context_embedding_net,
        t_theta_embedding_net=t_theta_embedding_net,
        auxiliary_data_embedding_net=auxiliary_data_embedding_net,
        t_theta_with_glu=t_theta_with_glu,
        context_with_glu=context_with_glu,
        auxiliary_data_with_glu=auxiliary_data_with_glu,
    )

    return network
