"""All functions and modules related to model definition.
"""

import torch
import sde_lib
import prob_path_lib
import numpy as np
import torch.autograd as autograd
import math
import torch.nn as nn
from torchdiffeq import odeint_adjoint


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max),
            np.log(config.model.sigma_min),
            config.model.num_scales,
        )
    )

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "beta_min": beta_start * (num_diffusion_timesteps - 1),
        "beta_max": beta_end * (num_diffusion_timesteps - 1),
        "num_diffusion_timesteps": num_diffusion_timesteps,
    }


def create_model(config, name=None):
    """Create the score model."""
    model_name = config.model.name
    if name:
        print("using supplied model name {}".format(name))
        model_name = name
    # TODO: HACK (can remove if you want, but linear embedding works best here)
    # if model_name == "nscnunet_t":
    #     assert config.model.embedding_type == "linear"
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)

    return score_model


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            # model.eval()
            return model(x, labels)
        else:
            # model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, train=train)
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.Z_VPSDE):

        def score_fn(x, t):
            assert continuous
            # labels = t * 999
            labels = t * 1  # TODO: scaling the t's seems to hurt performance atm
            score = model_fn(x, labels)
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]

            # for joint training
            if isinstance(score, list) or isinstance(score, tuple):
                score_x, score_t = score
                if len(x) < 4:
                    score_x = score_x / std[:, None]
                else:
                    score_x = score_x / std[:, None, None, None]
                return [score_x, score_t.squeeze()]
            else:
                if len(x) < 4:
                    return score / std[:, None]
                else:
                    return score / std[:, None, None, None]

    # elif isinstance(sde, sde_lib.VESDE):

    #     def score_fn(x, t):
    #         if continuous:
    #             labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
    #         else:
    #             # For VE-trained models, t=0 corresponds to the highest noise level
    #             labels = sde.T - t
    #             labels *= sde.N - 1
    #             labels = torch.round(labels).long()

    #         score = model_fn(x, labels)
    #         return score

    elif isinstance(sde, sde_lib.Z_RQNSF_VPSDE) or isinstance(
        sde, sde_lib.Z_RQNSF_TFORM_VPSDE
    ):

        def score_fn(x, t):
            assert continuous
            score = model_fn(x, t)

            # for joint training
            if isinstance(score, list) or isinstance(score, tuple):
                score_x, score_t = score
                return [score_x, score_t.squeeze()]
            else:
                return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def get_time_score_fn(sde, model, train=False, continuous=False):
    model_fn = get_model_fn(model, train=train)

    if (
        isinstance(sde, sde_lib.VPSDE)
        or isinstance(sde, sde_lib.Z_VPSDE)
        or isinstance(sde, sde_lib.Z_RQNSF_VPSDE)
        or isinstance(sde, sde_lib.Z_RQNSF_TFORM_VPSDE)
    ):

        def score_fn(x, t):
            return model_fn(x, t).squeeze()

    # if (
    #     isinstance(sde, sde_lib.VPSDE)
    #     or isinstance(sde, sde_lib.Z_VPSDE)
    #     or isinstance(sde, sde_lib.Z_RQNSF_VPSDE)
    #     or isinstance(sde, sde_lib.Z_RQNSF_TFORM_VPSDE)
    # ):

    #     def score_fn(x, t):
    #         assert continuous
    #         # labels = t * 999
    #         labels = t * 1  # TODO: scaling the t's seems to hurt performance atm
    #         score = model_fn(x, labels)

    #         if isinstance(score, list) or isinstance(score, tuple):
    #             score_x, score_t = score
    #         else:
    #             score_t = score
    #         return score_t.squeeze()

    # elif isinstance(sde, sde_lib.Z_RQNSF_VPSDE) or isinstance(
    #     sde, sde_lib.Z_RQNSF_TFORM_VPSDE
    # ):

    #     def score_fn(x, t):
    #         assert continuous
    #         score = model_fn(x, t)  # just feed in the t's for now?
    #         if isinstance(score, list) or isinstance(score, tuple):
    #             score_x, score_t = score
    #         else:
    #             score_t = score
    #         return score_t.squeeze()

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def get_c_time_score_fn(prob_path, model, train=False, continuous=False):
    model_fn = get_model_fn(model, train=train)

    if isinstance(prob_path, prob_path_lib.OneVP) or isinstance(
        prob_path, prob_path_lib.OneRQNSFVP
    ):

        def score_fn(x, t):
            return model_fn(x, t).squeeze()

        # def score_fn(x, t):
        #     assert continuous
        #     # labels = t * 999
        #     # labels = t * 1  # TODO: scaling the t's seems to hurt performance atm
        #     score = model_fn(x, t)

        #     # if isinstance(score, list) or isinstance(score, tuple):
        #     #     score_x, score_t = score
        #     # else:
        #     #     score_t = score
        #     # return score_t.squeeze()
        #     return score

    else:
        raise NotImplementedError(
            f"Prob path class {prob_path.__class__.__name__} not yet supported."
        )

    return score_fn


def get_c_time_epsilons_fn(prob_path, model, train=False, continuous=False):
    model_fn = get_model_fn(model, train=train)

    if isinstance(prob_path, prob_path_lib.OneRQNSFVP):

        def score_fn(x, t):
            return model_fn(x, t)

    else:
        raise NotImplementedError(
            f"Prob path class {prob_path.__class__.__name__} not yet supported."
        )

    return score_fn


def get_c_time_epsilons_score_fn(prob_path, model, train=False, continuous=False):
    model_fn = get_model_fn(model, train=train)

    if isinstance(prob_path, prob_path_lib.OneRQNSFVP):

        def score_fn(x, t):
            return prob_path.score_from_epsilons(model_fn(x, t), t)

        # def score_fn(x, t):
        #     # assert continuous
        #     # labels = t * 999
        #     # labels = t * 1  # TODO: scaling the t's seems to hurt performance atm
        #     score = model_fn(x, t)

        #     # if isinstance(score, list) or isinstance(score, tuple):
        #     #     score_x, epsilons = score
        #     # else:
        #     #     epsilons = score
        #     return prob_path.score_from_epsilons(score, t)

    else:
        raise NotImplementedError(
            f"Prob path class {prob_path.__class__.__name__} not yet supported."
        )

    return score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_sde_prior_logp_fn(z_space_model_name, sde):
    if "none" not in z_space_model_name:
        return sde.prior_logp
    else:

        # a bit hacky
        def prior_logp(flow, x):
            return flow.log_prob(x)

        return prior_logp


def get_prior_logp_fn(z_space_model_name):
    if "none" not in z_space_model_name:
        if "noise" in z_space_model_name or "copula" in z_space_model_name:
            # Z_RQNSF_TFORM_VPSDE
            def prior_logp(flow, x):
                # evaluates log p(x), where p(x) is a flow trained on MNIST
                n = x.size(0)
                shape = x.shape
                N = np.prod(shape[1:])
                with torch.no_grad():
                    flow.eval()
                    # TODO: input to flow needs to be uniformly dequantized and logit-transformed
                    # TODO (HACK)
                    # undo rescaling, then logit transform
                    x = (x + 1.0) / 2.0
                    x *= 256.0

                    # TODO: this will only be called for validation/test
                    log_p = flow.module._log_prob(
                        torch.clamp(x, 0.0, 256.0),
                        context=None,
                        transform=True,
                        train=False,
                    )
                    # try:
                    #   log_p = flow.module._log_prob(x, context=None, transform=True, train=False)
                    # except:
                    #   log_p = flow.module._log_prob(torch.clamp(x, 0., 256.), context=None, transform=True, train=False)
                # we need another log_det for undoing the rescaling operation
                log_p = log_p + N * np.log(256)
                log_p = log_p - N * np.log(2)

                return log_p

        else:
            # Z_RQNSF_VPSDE
            def prior_logp(flow, x):
                # evaluates log p(x), where p(x) is a flow trained on MNIST
                n = x.size(0)
                shape = x.shape
                N = np.prod(shape[1:])
                with torch.no_grad():
                    flow.eval()
                    # TODO: input to flow needs to be uniformly dequantized and logit-transformed
                    # TODO (HACK)
                    # undo rescaling, then logit transform
                    x = (x + 1.0) / 2.0
                    x *= 256.0
                    log_p = flow.module._log_prob(x, context=None)
                # we need another log_det for undoing the rescaling operation
                log_p = log_p + N * np.log(256)
                log_p = log_p - N * np.log(2)

                return log_p

    else:

        #  a bit hacky
        def prior_logp(flow, x):
            return flow.log_prob(x)

    return prior_logp


def get_score_fn_from_model(
    score_model,
    flow,
    flow_name,
    use_zt,
    mlp=False,
    method="RK45",
    eps=1e-5,
    conditional=False,
    device=None,
    sde=None,
    epsilons=False,
    prob_path=None,
    rtol=1e-3,
    atol=1e-6,
):
    """Create a function to compute the density ratios of a given point.
    NOTE: this is the one that's being used for the DDPM noise schedule!
    TODO: we are using this function to evaluate q(x) = MNIST, p(x) = flow trained on MNIST
    """

    if not conditional:
        score_fn_fn = lambda score_model: get_time_score_fn(
            sde, score_model, train=False, continuous=True
        )
    elif not epsilons:
        score_fn_fn = lambda score_model: get_c_time_score_fn(
            prob_path, score_model, train=False, continuous=True
        )
    else:
        score_fn_fn = lambda score_model: get_c_time_epsilons_score_fn(
            prob_path, score_model, train=False, continuous=True
        )

    def ratio_fn(u, time):
        # time is diffusion time
        time = time[0].item()

        # if math.isclose(time, 1.0):
        #     return torch.zeros(
        #         (u.shape[0]), requires_grad=False, dtype=torch.float32, device=device
        #     ), torch.zeros_like(
        #         u, requires_grad=False, dtype=torch.float32, device=device
        #     )

        if not conditional:
            times = (1.0, max(time, eps))
        else:
            times = (0.0, min(1.0 - time, 1.0 - eps))

        num_samples = u.shape[0]

        if times[0] == times[1]:
            return -u.detach()

        with torch.enable_grad():

            class ODEFunction(nn.Module):
                def __init__(self, score_model, u):
                    super(ODEFunction, self).__init__()
                    # self.score_model = score_model
                    self.score_fn = score_fn_fn(score_model)

                    self.u = nn.Parameter(u)

                    if use_zt:
                        self.x = self.u.view(self.u.shape[0], 1, 28, 28)
                    else:
                        if "none" not in flow_name:
                            if flow_name in ["mintnet", "nice", "realnvp"]:
                                # map z -> x via flow, then rescale to [-1, 1]
                                self.x = flow.module.sampling(self.u, rescale=True)
                            else:
                                if "noise" in flow_name or "copula" in flow_name:
                                    self.x = flow.module.sample(
                                        self.u.view(self.u.shape[0], -1),
                                        context=None,
                                        rescale=True,
                                        transform=True,
                                        train=False,
                                    )
                                else:
                                    self.x = flow.module.sample(
                                        self.u.view(self.u.shape[0], -1),
                                        context=None,
                                        rescale=True,
                                    )
                        else:
                            self.x = self.u.view(self.u.shape[0], 1, 28, 28)

                def forward(self, t, y):
                    t_tensor = t.expand(num_samples).to(dtype=torch.float32)
                    # assume it is only time score
                    return self.score_fn(self.x, t_tensor).reshape(-1)

            batch = u.view(num_samples, -1) if mlp else u
            ode_func = ODEFunction(score_model=score_model, u=batch).to(device)

            t = torch.tensor([times[0], times[1]], device=device)
            log_qp = odeint_adjoint(
                ode_func,
                torch.zeros(num_samples, device=device, dtype=torch.float64) + eps,
                t,
                # method=method,
                method="scipy_solver",
                options={"solver": method},
                atol=atol,
                rtol=rtol,
            )[-1]
            temp = -u.detach() + autograd.grad(log_qp.sum(), ode_func.u)[0].detach().to(
                dtype=torch.float32
            )
            return temp

    return ratio_fn
