"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""

import abc
import torch
import numpy as np
import torch.nn.functional as F
from scipy.integrate import quad


def logit_transform(image, lambd=1e-6):
    image = lambd + (1 - 2 * lambd) * image
    image = torch.log(image) - torch.log1p(-image)
    ldj = F.softplus(image) + F.softplus(-image) + np.log(1 - 2 * lambd)
    ldj = ldj.view(image.size(0), -1)  # (batch,)
    return image, ldj


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                # TODO: added this
                if isinstance(score, list) or isinstance(score, tuple):
                    score = score[0]
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                # TODO: added this
                score = score_fn(x, t)
                if isinstance(score, list) or isinstance(score, tuple):
                    score = score[0]
                rev_f = f - G[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    def get_path_var_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        raise NotImplementedError

    def get_obj_var_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        raise NotImplementedError

    def get_dummy_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        def time_weighting_quantities(t):
            lambda_t = 1
            lambda_t0 = 1
            lambda_t1 = 1
            lambda_dt = 0
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_time_weighting_quantities(
        self, likelihood_weighting, t0, t1, eps1, eps2, factor
    ):
        if likelihood_weighting == "dummy":
            return self.get_dummy_time_weighting_quantities(t0, t1, eps1, eps2, factor)
        elif likelihood_weighting == "path_var":
            return self.get_path_var_time_weighting_quantities(
                t0, t1, eps1, eps2, factor
            )
        elif likelihood_weighting == "obj_var":
            return self.get_obj_var_time_weighting_quantities(
                t0, t1, eps1, eps2, factor
            )
        else:
            raise NotImplementedError


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    # TODO: ugly code for compatibility with z-space training and x-space training;
    # make sure to clean this up later
    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        if len(x.size()) < 4:
            drift = -0.5 * beta_t[:, None] * x
        else:
            drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        if len(x.size()) < 4:
            mean = torch.exp(log_mean_coeff[:, None]) * x
        else:
            mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        n = z.size(0)
        N = np.prod(shape[1:])
        logps = (
            -N / 2.0 * np.log(2 * np.pi) - torch.sum(z.view(n, -1) ** 2, dim=-1) / 2.0
        )
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        if len(x.size()) < 4:
            f = torch.sqrt(alpha)[:, None] * x - x
        else:
            f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class Z_VPSDE(SDE):
    def __init__(self, flow, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.
        NOTE: this only works for MintNet, NICE, and RealNVP due to the preprocessing used!

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.flow = flow
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        # data will be in z-space due to flow, so 2-D
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        """
        draw a sample x ~ p(x) = f(z), where z ~ N(0,I)
        :param shape:
        :return:
        """
        # you shouldn't be calling this
        raise NotImplementedError
        # TODO (HACK): hardcoded cuda and module.sampling is not good
        z = torch.randn(*shape, device="cuda")
        with torch.no_grad():
            # TODO: note that samples will be centered to [-1, +1]
            x = self.flow.module.sampling(z)
        # return torch.randn(*shape)
        return x

    def prior_logp(self, flow, x):
        # evaluates log p(x), where p(x) is a flow trained on MNIST
        n = x.size(0)
        shape = x.shape
        N = np.prod(shape[1:])
        with torch.no_grad():
            flow.eval()
            # TODO: input to flow needs to be uniformly dequantized and logit-transformed
            # TODO (HACK)
            # undo rescaling, then logit transform
            x = torch.clamp((x + 1) / 2.0, 0.0, 1.0)
            x, log_det_logit = logit_transform(x)
            z, flow_log_det = flow(x, reverse=False)
        # N(0,I) log probability
        log_prob_z = (
            -N / 2.0 * np.log(2 * np.pi) - torch.sum(z.view(n, -1) ** 2, dim=-1) / 2.0
        )
        log_p = log_prob_z + flow_log_det + log_det_logit.sum(-1)
        # we need another log_det for undoing the rescaling operation
        log_p = log_p - N * np.log(2)

        return log_p

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        if len(x.size()) < 4:
            f = torch.sqrt(alpha)[:, None] * x - x
        else:
            f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


# TODO (HACK): you eventually want to get rid of this SDE @____@
class Z_RQNSF_VPSDE(SDE):
    def __init__(self, flow, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.flow = flow
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        # data will be in z-space due to flow, so 2-D
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, flow, x):
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

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        if len(x.size()) < 4:
            f = torch.sqrt(alpha)[:, None] * x - x
        else:
            f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    # def get_scale(self, likelihood_weighting, factor=1):
    #     if likelihood_weighting == "path_var":
    #         return 1.0
    #     else:
    #         # get scaling of path_var
    #         def get_alpha(t):
    #             return np.exp(
    #                 -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    #             )

    #         def g(t):
    #             return 1 - get_alpha(t) ** 2

    #         base, _ = quad(g, 0, 1)

    #         if likelihood_weighting == "dummy":
    #             return 1.0 / base
    #         else:

    #             def get_d_alpha(t):
    #                 return (
    #                     -0.5
    #                     * (self.beta_0 + (self.beta_1 - self.beta_0) * t)
    #                     * get_alpha(t)
    #                 )

    #             def f(t):
    #                 alpha = get_alpha(t)
    #                 d_alpha = get_d_alpha(t)
    #                 return (1 - alpha**2) ** 2 / (
    #                     2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
    #                 )

    #             result, _ = quad(f, 0, 1)
    #             scale = base / result
    #             return scale

    def get_time_weighting_quantities(self, likelihood_weighting, factor=1):
        if likelihood_weighting == "dummy":
            return self.get_dummy_time_weighting_quantities(factor)
        elif likelihood_weighting == "sde_var":
            return self.get_sde_var_time_weighting_quantities(factor)
        elif likelihood_weighting == "path_var":
            return self.get_path_var_time_weighting_quantities(factor)
        elif likelihood_weighting == "obj_var":
            return self.get_obj_var_time_weighting_quantities(factor)
        else:
            raise NotImplementedError

    def get_dummy_time_weighting_quantities(self, factor):
        def time_weighting_quantities(t, t0, t1):
            lambda_t = 1
            lambda_t0 = 1
            lambda_t1 = 1
            lambda_dt = 0
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_sde_var_time_weighting_quantities(self, factor):
        def time_weighting_quantities(t, t0, t1):
            lambda_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
            lambda_t0 = self.beta_0 + t0 * (self.beta_1 - self.beta_0)
            lambda_t1 = self.beta_0 + t1 * (self.beta_1 - self.beta_0)
            lambda_dt = self.beta_1 - self.beta_0
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_path_var_time_weighting_quantities(self, factor):
        def get_log_mean_coeff(t):
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        def time_weighting_quantities(t, t0, t1):
            log_mean_coeff = get_log_mean_coeff(t)
            lambda_t = 1.0 - torch.exp(2.0 * log_mean_coeff)
            lambda_t0 = 1.0 - torch.exp(2.0 * get_log_mean_coeff(t0))
            lambda_t1 = 1.0 - torch.exp(2.0 * get_log_mean_coeff(t1))
            alpha = torch.exp(log_mean_coeff)
            d_alpha = -0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * t) * alpha
            lambda_dt = -2.0 * alpha * d_alpha

            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_obj_var_time_weighting_quantities(self, factor):

        def get_alpha(t):
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        def get_d_alpha(t):
            return (
                0.5
                * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t))
                * get_alpha(t)
            )

        def get_weighting(t):
            alpha = get_alpha(t)
            d_alpha = get_d_alpha(t)
            return (1 - alpha**2) ** 2 / (
                2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
            )

        def time_weighting_quantities(t, t0, t1):
            with torch.enable_grad():
                t.requires_grad_(True)
                lambda_t = get_weighting(t)
                lambda_dt = torch.autograd.grad(lambda_t.sum(), t)[0]

            lambda_t = lambda_t.detach()
            lambda_t0 = get_weighting(t0)
            lambda_t1 = get_weighting(t1)
            lambda_dt = lambda_dt.detach()
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities


# TODO (HACK): you eventually want to get rid of this SDE @____@
class Z_RQNSF_TFORM_VPSDE(SDE):
    def __init__(self, flow, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        print("initialized Z_RQNSF_TFORM_VPSDE!")
        self.flow = flow
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        # data will be in z-space due to flow, so 2-D
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, flow, x):
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
                torch.clamp(x, 0.0, 256.0), context=None, transform=True, train=False
            )
            # try:
            #   log_p = flow.module._log_prob(x, context=None, transform=True, train=False)
            # except:
            #   log_p = flow.module._log_prob(torch.clamp(x, 0., 256.), context=None, transform=True, train=False)
        # we need another log_det for undoing the rescaling operation
        log_p = log_p + N * np.log(256)
        log_p = log_p - N * np.log(2)

        return log_p

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        if len(x.size()) < 4:
            f = torch.sqrt(alpha)[:, None] * x - x
        else:
            f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    # def get_scale(self, likelihood_weighting, factor=1):
    #     if likelihood_weighting == "path_var":
    #         return 1.0
    #     else:
    #         # get scaling of path_var
    #         def get_alpha(t):
    #             return np.exp(
    #                 -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    #             )

    #         def g(t):
    #             return 1 - get_alpha(t) ** 2

    #         base, _ = quad(g, 0, 1)

    #         if likelihood_weighting == "dummy":
    #             return 1.0 / base
    #         else:

    #             def get_d_alpha(t):
    #                 return (
    #                     -0.5
    #                     * (self.beta_0 + (self.beta_1 - self.beta_0) * t)
    #                     * get_alpha(t)
    #                 )

    #             def f(t):
    #                 alpha = get_alpha(t)
    #                 d_alpha = get_d_alpha(t)
    #                 return (1 - alpha**2) ** 2 / (
    #                     2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
    #                 )

    #             result, _ = quad(f, 0, 1)
    #             scale = base / result
    #             return scale

    def get_time_weighting_quantities(self, likelihood_weighting, factor=1):
        if likelihood_weighting == "dummy":
            return self.get_dummy_time_weighting_quantities(factor)
        elif likelihood_weighting == "sde_var":
            return self.get_sde_var_time_weighting_quantities(factor)
        elif likelihood_weighting == "path_var":
            return self.get_path_var_time_weighting_quantities(factor)
        elif likelihood_weighting == "obj_var":
            return self.get_obj_var_time_weighting_quantities(factor)
        else:
            raise NotImplementedError

    def get_dummy_time_weighting_quantities(self, factor):
        def time_weighting_quantities(t, t0, t1):
            lambda_t = 1
            lambda_t0 = 1
            lambda_t1 = 1
            lambda_dt = 0
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_sde_var_time_weighting_quantities(self, factor):
        def time_weighting_quantities(t, t0, t1):
            lambda_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
            lambda_t0 = self.beta_0 + t0 * (self.beta_1 - self.beta_0)
            lambda_t1 = self.beta_0 + t1 * (self.beta_1 - self.beta_0)
            lambda_dt = self.beta_1 - self.beta_0
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_path_var_time_weighting_quantities(self, factor):
        def get_log_mean_coeff(t):
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        def time_weighting_quantities(t, t0, t1):
            log_mean_coeff = get_log_mean_coeff(t)
            lambda_t = 1.0 - torch.exp(2.0 * log_mean_coeff)
            lambda_t0 = 1.0 - torch.exp(2.0 * get_log_mean_coeff(t0))
            lambda_t1 = 1.0 - torch.exp(2.0 * get_log_mean_coeff(t1))
            alpha = torch.exp(log_mean_coeff)
            d_alpha = -0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * t) * alpha
            lambda_dt = -2.0 * alpha * d_alpha

            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_obj_var_time_weighting_quantities(self, factor):

        def get_alpha(t):
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        def get_d_alpha(t):
            return (
                0.5
                * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t))
                * get_alpha(t)
            )

        def get_weighting(t):
            alpha = get_alpha(t)
            d_alpha = get_d_alpha(t)
            return (1 - alpha**2) ** 2 / (
                2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
            )

        def time_weighting_quantities(t, t0, t1):
            with torch.enable_grad():
                t.requires_grad_(True)
                lambda_t = get_weighting(t)
                lambda_dt = torch.autograd.grad(lambda_t.sum(), t)[0]

            lambda_t = lambda_t.detach()
            lambda_t0 = get_weighting(t0)
            lambda_t1 = get_weighting(t1)
            lambda_dt = lambda_dt.detach()
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities


class ToyInterpXt(SDE):
    def __init__(self, t_min=0.0, t_max=1.0, N=1000):
        """Construct a linear interpolation procedure. Note that this is not necessarily an SDE."""
        super().__init__(N)
        self.t_min = t_min
        self.t_max = t_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        raise NotImplementedError

    def marginal_prob(self, x, t):
        mean = x * t
        var = 1 - t**2
        std = torch.sqrt(var)
        return mean, std, var

    # def get_scale(self, likelihood_weighting, factor):
    #     if likelihood_weighting == "dummy":
    #         return 2 / 3
    #     elif likelihood_weighting == "path_var":
    #         return 1.0
    #     elif likelihood_weighting == "obj_var":
    #         if factor == 1:
    #             scale = 2.0 / (3.0 * torch.pi - 8.0)
    #         else:
    #             base = 2 / 3

    #             def f(t):
    #                 return (1 - t**2) ** 2 / (2 * t**2 + factor * (1 - t**2))

    #             result, _ = quad(f, 0, 1)
    #             scale = base / result
    #         return scale
    #     else:
    #         raise NotImplementedError

    def get_path_var_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        def time_weighting_quantities(t):
            lambda_t = 1 - t**2
            lambda_t0 = 1 - t0**2 + eps1**2
            lambda_t1 = 1 - t1**2 + eps2**2
            lambda_dt = -2 * t
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_obj_var_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        def time_weighting_quantities(t):
            var = 1 - t**2
            temp = 2 * t**2 + factor * var
            lambda_t = var**2 / temp
            lambda_t0 = (1 - t0**2) ** 2 / (factor + (2 - factor) * t0**2) + eps1**2
            lambda_t1 = (1 - t1**2) ** 2 / (factor + (2 - factor) * t1**2) + eps2**2
            lambda_dt = -2 * var**2 * (2 - factor) * t / temp**2 - 4 * var * t / temp
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=-1) / 2.0

    def discretize(self, x, t):
        raise NotImplementedError
