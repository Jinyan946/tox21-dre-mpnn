import abc
import torch
import math
import numpy as np
import torch.nn.functional as F
from scipy.integrate import quad


sqrt_two = math.sqrt(2)


class ProbPath(abc.ABC):
    name = "ProbPath"

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    @property
    @abc.abstractmethod
    def T(self):
        return 1

    @abc.abstractmethod
    def marginal_prob(self, x0, x1, t):
        raise NotImplementedError

    # def get_scale(self, likelihood_weighting, factor):
    #     raise NotImplementedError

    def get_dummy_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        def time_weighting_quantities(t):
            lambda_t = 1
            lambda_t0 = 1
            lambda_t1 = 1
            lambda_dt = 0
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_path_var_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        raise NotImplementedError

    def get_obj_var_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        raise NotImplementedError

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


class OneVP(ProbPath):
    name = "OneVP"

    # Note: p(x0) is standard Gaussian
    def __init__(self, dim):
        super().__init__(dim)

    @property
    def T(self):
        return 1

    def marginal_prob(self, x1, t):
        mean = t * x1
        var = 1 - t**2
        std = torch.sqrt(var)
        return mean, std, var

    def epsilon_partial_t_log_prob(self, epsilon, x1, t, var):
        # parameterized using epsilon
        # -\frac{1}{2}\partial_{t}k_{t}
        # temp = t * (1 - self.k_min)
        # kt = (1 - torch.square(t)) * (1 - self.k_min) + self.k_min
        # kt = 1 - torch.square(t)
        return (
            self.dim * t / var
            - t / var * torch.sum(torch.square(epsilon), dim=-1, keepdim=True)
            + 1 / torch.sqrt(var) * torch.sum(epsilon * x1, dim=-1, keepdim=True)
        )

    def epsilon_target(self, epsilon, x1, t, factor):
        var = 1 - t**2
        std = torch.sqrt(var)
        temp = torch.sqrt(2 * t**2 + factor * (1 - t**2))
        return (
            var / temp,
            (
                self.dim * t
                - t * torch.sum(torch.square(epsilon), dim=-1, keepdim=True)
                + std * torch.sum(epsilon * x1, dim=-1, keepdim=True)
            )
            / temp,
        )

    def full_epsilon_target(self, epsilon, x1, t, factor):
        var = 1 - t**2
        std = torch.sqrt(var)
        temp = torch.sqrt(2 * t**2 + factor * (1 - t**2))
        return (
            var / temp,
            (t - t * torch.square(epsilon) + std * epsilon * x1) / temp,
        )

    def x_partial_t_log_prob(self, x, x1, t, mean, var):
        # parameterized using x
        # -\frac{1}{2}\partial_{t}k_{t}
        # mut = t * x1
        # kt = (1 - torch.square(t)) * (1 - self.k_min) + self.k_min
        # kt = 1 - torch.square(t)
        diff_x_mu = x - mean

        return (
            self.dim * t
            - t / var * torch.sum(torch.square(diff_x_mu), dim=-1, keepdim=True)
            + torch.sum(diff_x_mu * x1, dim=-1, keepdim=True)
        ) / var

    def scaling(self, t, factor):
        var = 1 - t**2
        return torch.square(var) / (2 * t**2 + factor * var)

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


class OneRQNSFVP(ProbPath):
    name = "OneRQNSFVP"

    # Note: p(x0) is standard Gaussian
    # Expect image inputs
    def __init__(self, dim, beta_min=0.1, beta_max=20):
        super().__init__(dim)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.sqrt_dim = int(math.sqrt(dim))

    @property
    def T(self):
        return 1

    def marginal_prob(self, x1, t):
        log_mean_coeff = (
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x1
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def epsilon_partial_t_log_prob(self, epsilon, x1, t, var):
        alpha = torch.exp(
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        temp1 = alpha * d_alpha / var
        temp2 = d_alpha / torch.sqrt(var)
        return (
            self.dim * temp1
            - temp1 * torch.sum(torch.square(epsilon), dim=(1, 2, 3), keepdim=False)
            + temp2 * torch.sum(epsilon * x1, dim=(1, 2, 3), keepdim=False)
        )

    def full_epsilon_partial_t_log_prob(self, epsilon, x1, t, var):
        alpha = torch.exp(
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        temp1 = alpha * d_alpha / var
        temp1 = temp1[:, None, None, None]
        temp2 = d_alpha / torch.sqrt(var)
        temp2 = temp2[:, None, None, None]
        return temp1 - temp1 * torch.square(epsilon) + temp2 * epsilon * x1

    def epsilon_target(self, epsilon, x1, t, factor):
        log_mean_coeff = (
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        alpha = torch.exp(log_mean_coeff)
        var = 1.0 - torch.exp(2.0 * log_mean_coeff)
        std = torch.sqrt(var)
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        temp = torch.sqrt(
            2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
        )
        return (
            var / temp,
            (
                self.dim * alpha * d_alpha
                - alpha
                * d_alpha
                * torch.sum(torch.square(epsilon), dim=(1, 2, 3), keepdim=False)
                + d_alpha * std * torch.sum(epsilon * x1, dim=(1, 2, 3), keepdim=False)
            )
            / temp,
        )

    def get_a_b_c(self, t):
        log_mean_coeff = (
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        alpha = torch.exp(log_mean_coeff)
        var = 1.0 - torch.exp(2.0 * log_mean_coeff)
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        a = self.dim * alpha * d_alpha / (1 - alpha**2)
        b = -d_alpha / var / alpha
        c = d_alpha / torch.sqrt(var) / alpha
        return a, b, c

    def full_epsilon_target(self, epsilon, x1, t, factor):
        log_mean_coeff = (
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        alpha = torch.exp(log_mean_coeff)
        var = 1.0 - torch.exp(2.0 * log_mean_coeff)
        std = torch.sqrt(var)
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        temp = torch.sqrt(
            2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
        )
        alpha = alpha[:, None, None, None]
        d_alpha = d_alpha[:, None, None, None]
        temp = temp[:, None, None, None]

        return (
            alpha * d_alpha
            - alpha * d_alpha * torch.square(epsilon)
            + d_alpha * std[:, None, None, None] * epsilon * x1
        ) / temp

    def noise_pred_target(self, epsilon, x1, t, factor):
        log_mean_coeff = (
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        alpha = torch.exp(log_mean_coeff)
        var = 1.0 - torch.exp(2.0 * log_mean_coeff)
        std = torch.sqrt(var)
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        temp = torch.sqrt(
            2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
        )
        return (
            (
                self.dim * alpha * d_alpha
                - alpha
                * d_alpha
                * torch.sum(torch.square(epsilon), dim=(1, 2, 3), keepdim=False)
                + d_alpha * std * torch.sum(epsilon * x1, dim=(1, 2, 3), keepdim=False)
            )
            / temp
            / self.sqrt_dim
        )

    def noise_pred_scale(self, t, factor):
        # 1 / sqrt_var
        log_mean_coeff = (
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        alpha = torch.exp(log_mean_coeff)
        var = 1.0 - torch.exp(2.0 * log_mean_coeff)
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        temp = torch.sqrt(
            2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
        )
        return var / temp / self.sqrt_dim

    # def noise_pred_scaling(self, t, factor):
    #     # 1 / var
    #     log_mean_coeff = (
    #         -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
    #         - 0.5 * (1 - t) * self.beta_0
    #     )
    #     alpha = torch.exp(log_mean_coeff)
    #     var = 1.0 - torch.exp(2.0 * log_mean_coeff)
    #     d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
    #     temp = 2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
    #     return var**2 / temp / self.dim

    def x_partial_t_log_prob(self, x, x1, t, mean, var):
        diff_x_mu = x - mean
        alpha = torch.exp(
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        return (
            self.dim * alpha * d_alpha
            - alpha
            * d_alpha
            / var
            * torch.sum(torch.square(diff_x_mu), dim=(1, 2, 3), keepdim=False)
            + d_alpha * torch.sum(diff_x_mu * x1, dim=(1, 2, 3), keepdim=False)
        ) / var

    def scaling(self, t, factor):
        alpha = torch.exp(
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        return (1 - alpha**2) ** 2 / (
            2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
        )

    def inv_sqrt_scaling(self, t, factor):
        alpha = torch.exp(
            -0.25 * (1 - t) ** 2 * (self.beta_1 - self.beta_0)
            - 0.5 * (1 - t) * self.beta_0
        )
        d_alpha = 0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * (1 - t)) * alpha
        return torch.sqrt(
            2 * alpha**2 * d_alpha**2 + d_alpha**2 * (1 - alpha**2) * factor
        ) / (1 - alpha**2)

    def score_from_epsilons(self, epsilons, t):
        return self.inv_sqrt_scaling(t, 1) * torch.sum(
            epsilons, dim=(1, 2, 3), keepdim=False
        )


class TwoSB(ProbPath):
    name = "TwoSB"

    def __init__(self, dim, var=2.0):
        super().__init__(dim)
        self.sigma = math.sqrt(var)
        self.var = var
        self.sqrt2 = math.sqrt(2)

    @property
    def T(self):
        return 1

    def marginal_prob(self, x0, x1, t):
        mean = (1 - t) * x0 + t * x1
        var = t * (1 - t) * self.var
        std = torch.sqrt(var)
        return mean, std, var

    def epsilon_partial_t_log_prob(self, epsilon, x0, x1, t, var):
        # parameterized using epsilon
        mut_d = x1 - x0
        return (
            -0.5 * self.dim * (1 - 2 * t) / t / (1 - t)
            + 0.5
            * (1 - 2 * t)
            / t
            / (1 - t)
            * torch.sum(torch.square(epsilon), dim=1, keepdim=True)
            + 1
            / torch.sqrt(t * (1 - t))
            / self.sigma
            * torch.sum(epsilon * mut_d, dim=1, keepdim=True)
        )

    def epsilon_target(self, epsilon, x0, x1, t, factor):
        temp1 = torch.sqrt(1 - 4 * t + 4 * t**2 + 2 * factor * t - 2 * factor * t**2)
        lambda_t = self.sqrt2 * t * (1 - t) / temp1

        temp2 = (1 - 2 * t) / temp1

        mut_d = x1 - x0
        return (
            lambda_t,
            -1 / self.sqrt2 * self.dim * temp2
            + temp2 / self.sqrt2 * torch.sum(torch.square(epsilon), dim=1, keepdim=True)
            + self.sqrt2
            * torch.sqrt(t * (1 - t))
            / temp1
            / self.sigma
            * torch.sum(epsilon * mut_d, dim=1, keepdim=True),
        )

    def full_epsilon_target(self, epsilon, x0, x1, t, factor):
        temp1 = torch.sqrt(1 - 4 * t + 4 * t**2 + 2 * factor * t - 2 * factor * t**2)
        lambda_t = self.sqrt2 * t * (1 - t) / temp1

        temp2 = (1 - 2 * t) / temp1

        mut_d = x1 - x0
        return (
            lambda_t,
            -1 / self.sqrt2 * temp2
            + temp2 / self.sqrt2 * torch.square(epsilon)
            + self.sqrt2
            * torch.sqrt(t * (1 - t))
            / temp1
            / self.sigma
            * epsilon
            * mut_d,
        )

    def x_partial_t_log_prob(self, x, x0, x1, t, mean, var):
        # parameterized using x
        mut_d = x1 - x0
        diff_x_mu = x - mean
        return (
            -0.5 * self.dim * (1 - 2 * t) / t / (1 - t)
            + 0.5
            * (1 - 2 * t)
            / (t * (1 - t)) ** 2
            / self.var
            * torch.sum(torch.square(diff_x_mu), dim=-1, keepdim=True)
            + 1 / var * torch.sum(diff_x_mu * mut_d, dim=-1, keepdim=True)
        )

    def get_path_var_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        def time_weighting_quantities(t):
            lambda_t = t * (1 - t) * self.var
            lambda_t0 = t0 * (1 - t0) * self.var + eps1**2
            lambda_t1 = t1 * (1 - t1) * self.var + eps2**2
            lambda_dt = (1 - 2 * t) * self.var
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def get_obj_var_time_weighting_quantities(self, t0, t1, eps1, eps2, factor):
        def time_weighting_quantities(t):
            temp = 1 - 4 * t + 4 * t**2 + 2 * factor * t - 2 * factor * t**2
            lambda_t = 2 * t**2 * (1 - t) ** 2 / temp
            lambda_t0 = (
                2
                * t0**2
                * (1 - t0) ** 2
                / (1 - 4 * t0 + 4 * t0**2 + 2 * factor * t0 - 2 * factor * t0**2)
            ) + eps1**2
            lambda_t1 = (
                2
                * t1**2
                * (1 - t1) ** 2
                / (1 - 4 * t1 + 4 * t1**2 + 2 * factor * t1 - 2 * factor * t1**2)
            ) + eps2**2
            lambda_dt = (
                4
                * t
                * (t - 1)
                * (
                    t * (t - 1) * (2 * factor * t - factor - 4 * t + 2)
                    + (2 * t - 1) * temp
                )
                / temp**2
            )
            return lambda_t, lambda_t0, lambda_t1, lambda_dt

        return time_weighting_quantities

    def scaling(self, t, factor):
        temp = 1 - 4 * t + 4 * t**2 + 2 * factor * t - 2 * factor * t**2
        return 2 * t**2 * (1 - t) ** 2 / temp
