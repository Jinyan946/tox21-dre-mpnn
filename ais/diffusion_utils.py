import math
from models import utils as mutils
import torch
import torch.nn as nn
from scipy import integrate
import numpy as np
from functools import partial
from torch import autograd
from torchdiffeq import odeint_adjoint


def log_normal(x, mean, logvar):
    """Log-pdf for factorized Normal distributions."""
    return -0.5 * (
        (math.log(2 * math.pi) + logvar).sum(1)
        + ((x - mean).pow(2) / torch.exp(logvar)).sum(1)
    )


def logmeanexp(x, dim=1):
    max_, _ = torch.max(x, dim=dim, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), dim=dim)) + max_.squeeze(dim=dim)


def safe_repeat(x, n):
    return x.repeat(n, *[1 for _ in range(len(x.size()) - 1)])


def get_ratio_fn_flow(
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
        score_fn_fn = lambda score_model: mutils.get_time_score_fn(
            sde, score_model, train=False, continuous=True
        )
    elif not epsilons:
        score_fn_fn = lambda score_model: mutils.get_c_time_score_fn(
            prob_path, score_model, train=False, continuous=True
        )
    else:
        score_fn_fn = lambda score_model: mutils.get_c_time_epsilons_score_fn(
            prob_path, score_model, train=False, continuous=True
        )

    # print('I am in the correct DRE function!')
    def ratio_fn(u, time1, time2):
        time1 = time1.item()
        time2 = time2.item()

        if not conditional:
            times = (max(1.0 - time1, eps), max(1.0 - time2, eps))
        else:
            times = (min(time1, 1.0 - eps), min(time2, 1.0 - eps))

        with torch.no_grad():
            if use_zt:
                x = u.view(u.shape[0], 1, 28, 28)
            else:
                if "none" not in flow_name:
                    if flow_name in ["mintnet", "nice", "realnvp"]:
                        # map z -> x via flow, then rescale to [-1, 1]
                        x = flow.module.sampling(u, rescale=True)
                    else:
                        if "noise" in flow_name or "copula" in flow_name:
                            x = flow.module.sample(
                                u.view(u.shape[0], -1),
                                context=None,
                                rescale=True,
                                transform=True,
                                train=False,
                            )
                        else:
                            x = flow.module.sample(
                                u.view(u.shape[0], -1), context=None, rescale=True
                            )
                else:
                    x = u.view(u.shape[0], 1, 28, 28)

            num_samples = x.shape[0]

            def ode_func(t, y, x, score_model):
                score_fn = score_fn_fn(score_model)

                t = torch.full((num_samples,), t, device=device)
                t = t.detach()
                # assume it is only time score
                rx = score_fn(x, t)  # get timewise-scores only
                rx = np.reshape(rx.detach().cpu().numpy(), -1)

                return rx

            # now just a function of t
            batch = x.view(num_samples, -1) if mlp else x
            p_get_rx = partial(ode_func, x=batch, score_model=score_model)
            # TODO: flipped (eps, 1) for DDPM noise
            solution = integrate.solve_ivp(
                fun=p_get_rx,
                t_span=times,
                y0=np.zeros((x.shape[0],)),
                method=method,
                rtol=rtol,
                atol=atol,
            )
            # nfe = solution.nfev
            density_ratio = solution.y[:, -1]
            # print("ratio computation took {} function evaluations.".format(nfe))

            log_qp = torch.tensor(
                density_ratio, device=device, dtype=torch.float32
            ).detach()
            return log_qp

    return ratio_fn


def get_torchdiffeq_dratio_fn_flow(
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
        score_fn_fn = lambda score_model: mutils.get_time_score_fn(
            sde, score_model, train=False, continuous=True
        )
    elif not epsilons:
        score_fn_fn = lambda score_model: mutils.get_c_time_score_fn(
            prob_path, score_model, train=False, continuous=True
        )
    else:
        score_fn_fn = lambda score_model: mutils.get_c_time_epsilons_score_fn(
            prob_path, score_model, train=False, continuous=True
        )

    def ratio_fn(u, time):
        time = time.item()

        if math.isclose(time, 0.0):
            return torch.zeros(
                (u.shape[0]), requires_grad=False, dtype=torch.float32, device=device
            ), torch.zeros_like(
                u, requires_grad=False, dtype=torch.float32, device=device
            )

        if not conditional:
            times = (1.0, max(1.0 - time, eps))
        else:
            times = (0.0, min(time, 1.0 - eps))

        num_samples = u.shape[0]

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
        return (
            log_qp.detach().to(dtype=torch.float32),
            autograd.grad(log_qp.sum(), ode_func.u)[0].detach().to(dtype=torch.float32),
        )

    return ratio_fn
