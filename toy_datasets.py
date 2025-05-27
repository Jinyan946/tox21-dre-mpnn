import math
import torch
import numpy as np
from torch.distributions import (
    Uniform,
    MultivariateNormal,
    Normal,
    Independent,
    MixtureSameFamily,
    Categorical,
    TransformedDistribution,
)
from torch.distributions.transforms import ReshapeTransform
import os

if not os.path.exists("val_sets"):
    os.mkdir("val_sets")


class PeakedGaussians(object):
    """
    The ratio we are estimating is: r(x) = log q(x) - log p(x)
    where q(x) = N(0, 1e-6) and p(x) = N(0, 1)
    q(x) corresponds to T = 1, p(x) corresponds to T = 0
    """

    def __init__(self, dim, sigmas, device):
        self.means = [0, 0]
        self.sigmas = sigmas
        self.dim = dim
        self.device = device

        self.q = Normal(0, self.sigmas[0])
        self.p = Normal(0, self.sigmas[1])

    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        return torch.sqrt(1 - t**2) * px + (t * qx)

    def sample(self, n, t):
        qx = torch.randn((n, self.dim)) * self.sigmas[0]
        px = torch.randn((n, self.dim)) * self.sigmas[1]
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def c_sample(self, n, t):
        qx = torch.randn((n, self.dim)) * self.sigmas[0]
        px = torch.randn((n, self.dim)) * self.sigmas[1]

        return px.to(self.device), qx.to(self.device)

    def log_density_ratios(self, samples):
        log_p = self.p.log_prob(samples)
        log_q = self.q.log_prob(samples)
        log_ratios = log_q - log_p

        return log_ratios

    def log_prob(self, samples, t):
        # get exact form of gaussian
        mu = 0.0
        var = (1 - t**2) * (self.sigmas[1] ** 2) + (t**2) * (self.sigmas[0] ** 2)

        log_q = -((samples - mu) ** 2).sum(dim=-1) / (2 * var) - 0.5 * torch.log(
            2 * math.pi * var
        )
        return log_q


# @title Define GMM dataset object
class ToyGMM(object):
    """
    The ratio we are estimating is: r(x) = log q(x) - log p(x)
    where q(x) = N(4, I) and p(x) = N(0, I)
    q(x) corresponds to T = 1 (data), p(x) corresponds to T = 0 (noise)
    """

    def __init__(self, sde, mean_q, mean_p, dim):
        self.sigmas = [1, 1]
        self.means = [mean_q, mean_p]
        self.dim = dim
        self.sde = sde

        self.q = Normal(self.means[0], 1)
        self.p = Normal(self.means[1], 1)

    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        mean, std = self.sde.marginal_prob(qx, t)  # qx is data
        xt = (torch.randn((len(t), 1)) * std + mean).view(-1, 1)

        return xt

    def sample(self, n, t):
        qx = self.q.sample((n, self.dim))
        px = self.p.sample((n, self.dim))
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px, qx, xt

    def sample_data(self, n):
        return self.q.sample((n, self.dim))

    def log_density_ratios(self, samples):
        log_p = self.p.log_prob(samples)
        log_q = self.q.log_prob(samples)
        log_ratios = log_q.sum(-1, keepdims=True) - log_p.sum(-1, keepdims=True)

        return log_ratios

    def log_prob(self, x, t, std=1):
        mean_t, std_t = self.sde.marginal_prob(self.means[0] * torch.ones_like(t), t)
        # marginal dist rescales the mean
        std_t = torch.sqrt(std_t**2 + (std**2 * mean_t**2 / self.means[0] ** 2))
        log_q = -((x - mean_t) ** 2).sum(-1, keepdims=True) / (
            2 * std_t**2
        ) - 0.5 * torch.log(2 * math.pi * std_t**2)
        assert log_q.size() == mean_t.size()
        return log_q

    # NOTE: this is the function that's being used for ratio computation
    def log_prob_mixture(self, x, t, batch):
        """
        x: samples (xt)
        t: time
        samples: batch
        """
        mu, sigma = self.sde.marginal_prob(batch, t)
        log_qs = []
        for i in range(len(mu)):
            log_q = (
                -((x[i] - mu[i]) ** 2) / (2 * sigma[i] ** 2)
                - 0.5 * torch.log(2 * math.pi * sigma[i] ** 2)
            ) + math.log(1.0 / len(mu))
            log_qs.append(log_q)
        log_q = torch.logsumexp(torch.stack(log_qs, dim=0), dim=0)
        return log_q


class GMMDist(object):
    """
    The ratio we are estimating is: r(x) = log q(x) - log p(x)
    where q(x) = N(4, I) and p(x) = N(0, I)
    q(x) corresponds to T = 1 (data), p(x) corresponds to T = 0 (noise)
    """

    def __init__(self, dim, device):
        self.means = [4, 0]  # 0: q, 1: p
        self.sigmas = [1, 1]
        self.dim = dim
        self.device = device

        self.q = Normal(self.means[0], 1)
        self.p = Normal(self.means[1], 1)

    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        return torch.sqrt(1 - t**2) * px + (t * qx)

    def sample(self, n, t):
        qx = self.q.sample((n, self.dim))
        px = self.p.sample((n, self.dim))
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def log_density_ratios(self, samples):
        log_p = self.p.log_prob(samples)
        log_q = self.q.log_prob(samples)
        log_ratios = log_q - log_p

        return log_ratios

    def log_prob(self, samples, t, sigma=1):
        # get exact form of gaussian
        mu = (self.means[0] * t) + (self.means[1] * torch.sqrt(1 - t**2)).to(
            samples.device
        )
        # sigma still remains 1 (HACK)
        log_q = -((samples - mu) ** 2).sum(dim=-1) / (2 * sigma**2) - 0.5 * np.log(
            2 * np.pi * sigma**2
        )
        return log_q


class OneSided(object):

    def __init__(self, dim, q, device, cov_trace, mean_sqnorm, unit_factor):
        self.dim = dim
        self.p = Independent(
            Normal(torch.zeros(dim, device=device), torch.ones(dim, device=device)),
            reinterpreted_batch_ndims=1,
        )
        self.q = q
        self.device = device
        if unit_factor:
            self.factor = 1.0
        else:
            self.factor = (cov_trace + mean_sqnorm) / self.dim

    # def sample_sequence_on_the_fly(self, px, qx, t):
    #     return torch.sqrt(1 - t**2) * px + (t * qx)

    def sample_sequence_on_the_fly(self, qx, t):
        px = torch.randn_like(qx)
        return px, qx, torch.sqrt(1 - t**2) * px + (t * qx)

    def sample(self, n, t):
        qx = self.q.sample((n,))
        px = self.p.sample((n,))
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def one_sample(self, n):
        qx = self.q.sample((n,))

        return [qx.to(self.device)]

    def two_sample(self, n):
        qx = self.q.sample((n,))
        px = self.p.sample((n,))

        return [px.to(self.device), qx.to(self.device)]

    def log_density_ratios(self, samples):
        log_p = self.p.log_prob(samples)
        log_q = self.q.log_prob(samples)
        log_ratios = log_q - log_p

        return log_ratios


class TwoSided(object):

    def __init__(
        self,
        dim,
        p,
        q,
        device,
        cov_trace,
        mean_sqnorm,
        unit_factor,
        two_sb_var,
        use_two_sb,
    ):
        self.dim = dim
        self.p = p
        self.q = q
        self.device = device
        if unit_factor:
            self.factor = 1.0
        else:
            self.factor = (cov_trace + mean_sqnorm) / self.dim
        self.two_sb_var = two_sb_var
        self.two_sb_sigma = math.sqrt(self.two_sb_var)

    # def sample_sequence_on_the_fly(self, px, qx, t):
    #     return torch.sqrt(1 - t**2) * px + (t * qx)

    # def sample_sequence_on_the_fly_sb(self, px, qx, t):
    #     return (
    #         t * qx
    #         + (1 - t) * px
    #         + torch.sqrt(t * (1 - t) * self.two_sb_var) * torch.randn_like(px)
    #     )

    def sample_sequence_on_the_fly(self, px, qx, t):
        return px, qx, torch.sqrt(1 - t**2) * px + (t * qx)

    def sample_sequence_on_the_fly_ot(self, px, qx, t):
        return px, qx, t * qx + (1 - t) * px

    def sample_sequence_on_the_fly_sb(self, px, qx, t):
        return (
            px,
            qx,
            (
                t * qx
                + (1 - t) * px
                + torch.sqrt(t * (1 - t) * self.two_sb_var) * torch.randn_like(px)
            ),
        )

    # def sample(self, n, t):
    #     qx = self.q.sample((n,))
    #     px = self.p.sample((n,))
    #     xt = self.sample_sequence_on_the_fly(px, qx, t)

    #     return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def sample(self, n, t):
        qx = self.q.sample((n,))
        px = self.p.sample((n,))
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def sample_two_sb(self, n, t):
        qx = self.q.sample((n,))
        px = self.p.sample((n,))
        xt = self.sample_sequence_on_the_fly_sb(px, qx, t)

        return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def one_sample(self, n):
        # does not make sense
        raise NotImplementedError

    def two_sample(self, n):
        qx = self.q.sample((n,))
        px = self.p.sample((n,))

        return [px.to(self.device), qx.to(self.device)]

    def log_density_ratios(self, samples):
        log_p = self.p.log_prob(samples)
        log_q = self.q.log_prob(samples)
        log_ratios = log_q - log_p

        return log_ratios


class GaussiansforMI(object):
    """
    The ratio we are estimating is: r(x) = log q(x) - log p(x)
    where q(x) = N(0, 1e-6) and p(x) = N(0, 1)
    q(x) corresponds to T = 1, p(x) corresponds to T = 0

    some code adapted from: https://github.com/benrhodes26/tre_code/blob/master/data_handlers/gaussians.py
    """

    def __init__(self, dim, device):
        self.means = [0, 0]
        self.dim = dim
        self.true_mutual_info = self.get_true_mi()
        self.device = device
        self.rho = self.get_rho_from_mi(
            self.true_mutual_info, self.dim
        )  # correlation coefficient
        # self.rhos = np.ones(self.dim // 2) * self.rho
        # self.variances = np.ones(self.dim)
        # self.cov_matrix = block_diag(
        #     *[[[1, self.rho], [self.rho, 1]] for _ in range(self.dim // 2)]
        # )
        # self.denom_cov_matrix = np.diag(self.variances)
        cov_tensor = torch.tensor(
            [[[1, self.rho], [self.rho, 1]] for _ in range(self.dim // 2)],
            dtype=torch.float,
            device=self.device,
        )
        base_dist = Independent(
            MultivariateNormal(
                torch.zeros(dim // 2, 2, device=self.device), cov_tensor
            ),
            1,
        )
        transform = ReshapeTransform((dim // 2, 2), (dim,))
        self.dist = TransformedDistribution(base_dist, [transform])
        self.denom_dist = Independent(
            Normal(
                torch.zeros(dim, device=self.device),
                torch.ones(dim, device=self.device),
            ),
            1,
        )

        self.factor = 1.0

    @staticmethod
    def get_mi_from_rho(self):
        return -0.5 * np.log(1 - self.rho**2) * self.dim

    @staticmethod
    def get_rho_from_mi(mi, n_dims):
        """Get correlation coefficient from true mutual information"""
        x = (4 * mi) / n_dims  # wtf??
        # x = (2 * mi) / n_dims
        return (1 - np.exp(-x)) ** 0.5  # correlation coefficient

    def get_true_mi(self):
        if self.dim == 20:
            mi = 5
        elif self.dim == 40:
            mi = 10
        elif self.dim == 80:
            mi = 20
        elif self.dim == 160:
            mi = 40
        elif self.dim == 320:
            mi = 80
        else:
            raise NotImplementedError
        return mi

    # def sample_gaussian(self, n_samples, cov_matrix):
    #     prod_of_marginals = multivariate_normal(mean=np.zeros(self.dim), cov=cov_matrix)
    #     return prod_of_marginals.rvs(n_samples)

    def sample_data(self, n_samples):
        # p_0 (correlated distribution) -> q(x)
        # qx = torch.from_numpy(self.sample_gaussian(n_samples, self.cov_matrix)).float()
        # qx = torch.from_numpy(self.dist.rvs(n_samples)).float()
        qx = self.dist.sample((n_samples,))
        return qx

    def sample_data_detach(self, n_samples):
        # p_0 (correlated distribution) -> q(x)
        # qx = (
        #     torch.from_numpy(self.sample_gaussian(n_samples, self.cov_matrix))
        #     .float()
        #     .detach()
        # )
        qx = self.dist.sample((n_samples,)).detach()
        return qx

    # indexes = (torch.arange(n_samples)[:, None] + torch.arange(n_ref)) % N

    def sample_denominator(self, n_samples):
        # p_m (noise distribution) -> p(x)
        # return torch.from_numpy(
        #     self.sample_gaussian(n_samples, self.denom_cov_matrix)
        # ).float()
        return self.denom_dist.sample((n_samples,))

    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        return torch.sqrt(1 - t**2) * px + (t * qx)

    def sample(self, n, t):
        qx = self.sample_data(n)  # p_0(x)
        px = self.sample_denominator(n)  # p_m(x)
        xt = self.sample_sequence_on_the_fly(px, qx, t).float()

        # return noise, data, interp
        return px.to(self.device), qx.to(self.device), xt.to(self.device)

    def numerator_log_prob(self, u):
        # bivariate_normal = multivariate_normal(
        #     mean=np.zeros(self.dim), cov=self.cov_matrix
        # )
        # log_probs = bivariate_normal.logpdf(u)
        # return log_probs
        return self.dist.log_prob(u)

    def denominator_log_prob(self, u):
        # prod_of_marginals = multivariate_normal(
        #     mean=np.zeros(self.dim), cov=self.denom_cov_matrix
        # )
        # return prod_of_marginals.logpdf(u)
        return self.denom_dist.log_prob(u)

    def empirical_mutual_info(self, samples=None):
        if samples is None:
            samples = self.sample_data(100000)
        return (
            torch.mean(
                self.numerator_log_prob(samples) - self.denominator_log_prob(samples)
            )
            .cpu()
            .detach()
            .numpy()
        )


def get_dataset(config, sde=None):
    # prob_path = get_prob_path(config.data.dim, config.training.prob_path)
    device = config.device

    # if config.data.dataset == "GMM":
    #     return GMMDist(config.data.dim, config.device)
    # elif config.data.dataset == "ToyGMM":
    #     return ToyGMM(sde, config.data.mean_q, config.data.mean_p, config.data.dim)
    if config.data.dataset == "Gaussians":
        dim = config.data.dim
        number = 4.0
        q = Independent(
            Normal(
                torch.full((dim,), number, device=device),
                torch.ones((dim,), device=device),
            ),
            reinterpreted_batch_ndims=1,
        )
        current_dataset = OneSided(
            dim,
            q,
            config.device,
            cov_trace=dim,
            mean_sqnorm=dim * number**2,
            unit_factor=config.training.unit_factor,
        )
        val_path = f"val_sets/{config.data.dataset}_{config.data.dim}.pt"
        # HACK: get val set
        if config.training.n_iters == -1:
            torch.manual_seed(1)
            qs = current_dataset.q.sample((5000,))
            ps = current_dataset.p.sample((5000,))
            mesh = torch.cat([qs, ps])
            torch.save(mesh, val_path)
            torch.manual_seed(config.seed)

        return current_dataset

    elif config.data.dataset == "GMMs":
        dim = config.data.dim
        mix = Categorical(torch.full((2,), 0.5))
        # k is defined as \delta\mu = k\sigma
        k = config.data.k
        sigma = math.sqrt(4 / (4 + k**2))
        half_delta_mu = 0.5 * k * sigma
        mu0 = 2.0
        mu1 = -2.0

        comp1 = Independent(
            Normal(
                torch.stack(
                    [
                        torch.full((dim,), mu0 - half_delta_mu, device=device),
                        torch.full((dim,), mu0 + half_delta_mu, device=device),
                    ],
                    dim=0,
                ),
                torch.full(
                    (2, dim),
                    sigma,
                    device=device,
                ),
            ),
            1,
        )
        p = MixtureSameFamily(mix, comp1)
        comp2 = Independent(
            Normal(
                torch.stack(
                    [
                        torch.full((dim,), mu1 - half_delta_mu, device=device),
                        torch.full((dim,), mu1 + half_delta_mu, device=device),
                    ],
                    dim=0,
                ),
                torch.full(
                    (2, dim),
                    sigma,
                    device=device,
                ),
            ),
            1,
        )
        q = MixtureSameFamily(mix, comp2)

        # both p and q have variance equals to 1 for each component
        # TwoSB prob_path is configured to use var=2, such that cov_trace=dim
        current_dataset = TwoSided(
            dim,
            p,
            q,
            config.device,
            cov_trace=(
                2.0 * dim / config.training.two_sb_var
                if config.training.two_sb_var != 0.0
                else 1.0
            ),  # dummy
            mean_sqnorm=(
                16.0 * dim / config.training.two_sb_var
                if config.training.two_sb_var != 0.0
                else 1.0
            ),  # dummy
            unit_factor=config.training.unit_factor,
            two_sb_var=config.training.two_sb_var,
            use_two_sb=config.training.use_two_sb,
        )
        val_path = (
            f"val_sets/{config.data.dataset}_{config.data.dim}_{config.data.k}.pt"
        )
        # HACK
        if config.training.n_iters == -1:
            torch.manual_seed(1)
            qs = current_dataset.q.sample((5000,))
            ps = current_dataset.p.sample((5000,))
            mesh = torch.cat([qs, ps])
            torch.save(mesh, val_path)
            torch.manual_seed(config.seed)

        return current_dataset

    elif config.data.dataset == "GaussiansforMI":
        current_dataset = GaussiansforMI(config.data.dim, config.device)

        val_path = f"val_sets/{config.data.dataset}_{config.data.dim}.pt"
        if not os.path.exists(val_path):
            torch.manual_seed(1)
            samples = current_dataset.sample_data(10000).to(device)
            torch.save(samples, val_path)
            torch.manual_seed(config.seed)

        return current_dataset

    elif config.data.dataset == "PeakedGaussians":
        return PeakedGaussians(config.data.dim, config.data.sigmas, config.device)
    else:
        raise NotImplementedError
