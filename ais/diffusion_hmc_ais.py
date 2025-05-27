from typing import List
from typing import Optional
from typing import Union
import torch.autograd as autograd
import numpy as np

# GPT solution
import os
import sys

# Step 1: Determine the parent directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Step 2: Add the parent directory to sys.path
sys.path.append(parent_dir)

from datasets import logit_transform

import torch
from tqdm import tqdm

import ais.diffusion_hmc as hmc
import ais.diffusion_utils as utils
from torch.distributions import Independent, Normal


def ais_fn(
    flow,
    flow_name,
    score_model,
    use_zt,
    conditional,
    batch_size,
    dataloader,
    num_ais_samples,
    num_ais_steps,
    num_steps_per_ais_step,
    num_continue,
    ais_method,
    num_hmc_steps,
    scaler,
    inverse_scaler,
    sde,
    epsilons,
    prob_path,
    rtol,
    atol,
    eps=1e-5,
    initial_step_size: Optional[int] = 0.01,
    device: Optional[torch.device] = None,
):
    """Compute annealed importance sampling trajectories for a batch of data.

    Could be used for *both* forward and reverse chain in BDMC.

    Sampling is carried out in the latent space of the flow.

    Args:
      flow: the normalizating flow
      score_model: the trained score model
      loader (iterator): iterator that returns pairs, with first component
        being `x`, second would be `z` or label (will not be used)
      forward: indicate forward/backward chain
      schedule: temperature schedule, i.e. `p(z)p(x|z)^t`
      n_sample: number of importance samples
      device: device to run all computation on
      initial_step_size: step size for leap-frog integration;
        note that this is the step size all along as I am not adapting it

    Returns:
        a list where each element is a torch.Tensor that contains the
        log importance weights for a single batch of data
    """

    if "none" not in flow_name:
        flow.eval()
        flow.to(device)
    score_model.eval()
    score_model.to(device)

    if ais_method == "ais":
        print("Running AIS")
        forward = True
    elif ais_method == "raise":
        print("Running RAISE")
        forward = False
    else:
        raise NotImplementedError

    schedule = torch.linspace(
        0.0, 1.0 - eps, num_ais_steps + 1, device=device
    ).contiguous()
    if not forward:
        schedule = torch.flip(schedule, dims=(0,)).contiguous()

    ratio_fn = utils.get_ratio_fn_flow(
        score_model,
        flow,
        flow_name,
        use_zt=use_zt,
        conditional=conditional,
        device=device,
        sde=sde,
        epsilons=epsilons,
        prob_path=prob_path,
        rtol=rtol,
        atol=atol,
    )
    dratio_fn = utils.get_torchdiffeq_dratio_fn_flow(
        score_model,
        flow,
        flow_name,
        use_zt=use_zt,
        conditional=conditional,
        device=device,
        sde=sde,
        epsilons=epsilons,
        prob_path=prob_path,
        rtol=rtol,
        atol=atol,
    )

    # @torch.enable_grad()
    # def dlogp_0_fn(x):
    #     x = x.clone().requires_grad_()
    #     (grad,) = autograd.grad(torch.sum(flow.module._distribution.log_prob(x)), x)
    #     return grad

    base_dist = Independent(
        Normal(torch.zeros((784,), device=device), torch.ones((784,), device=device)),
        reinterpreted_batch_ndims=1,
    )

    def logp_0_fn(z):
        return base_dist.log_prob(z)

    def dlogp_0_fn(z):
        return -z

    def normalized_kinetic(v):
        zeros = torch.zeros_like(v)
        return -utils.log_normal(v, zeros, zeros)

    logws = []
    samples = []
    z_samples = []
    accept_hists = []
    init_zs = []

    for batch in dataloader:

        accept_hist = torch.zeros(size=(batch_size,), device=device)
        logw = torch.zeros(size=(batch_size,), device=device)

        # initial sample of z
        if forward:
            current_z = torch.randn(size=(batch_size, 784), device=device)
        else:
            train = False
            batch = batch[0]

            batch = batch.to(device).float()

            batch = batch * 255.0 / 256.0
            batch += torch.rand_like(batch) / 256.0

            batch = scaler(batch)

            if "none" not in flow_name:

                # adapted from losses.py
                with torch.no_grad():
                    flow.eval()
                    current_z = (batch + 1.0) / 2.0
                    if flow_name in ["mintnet", "nice", "realnvp"]:
                        # undo rescaling, apply logit transform, pass through flow
                        current_z = logit_transform(current_z)
                        current_z, _ = flow(current_z, reverse=False)
                        # current_z = current_z.view(batch.size())
                    else:
                        current_z *= 256.0
                        # annoying, but now we need to branch to RQ-NSF flow vs [noise, copula]
                        if "noise" in flow_name or "copula" in flow_name:
                            # apply data transform here (1/256, logit transform, mean-centering)
                            current_z = flow.module.transform_to_noise(
                                current_z, transform=True, train=train
                            )
                        else:
                            # for the RQ-NSF flow, the data is dequantized and between [0, 256]
                            # and the flow's preprocessing module takes care of normalization
                            current_z = flow.module.transform_to_noise(current_z)
                        # current_z = current_z.view(batch.size())

            else:

                current_z = batch.view((batch.shape[0], 784))

        init_zs.append(current_z.detach().cpu().clone())

        num_accept_reject = 0

        epsilon = torch.full(
            size=(batch_size,), device=device, fill_value=initial_step_size
        )

        for t0, t1 in tqdm(zip(schedule[:-1], schedule[1:])):

            # update log importance weight
            logw += ratio_fn(current_z, t0, t1)

            @torch.enable_grad
            def grad_U(z):
                # assuming the base distribution is standard normal, which is true for all flows that we actually use
                logp_0 = logp_0_fn(z)
                dlogp_0 = dlogp_0_fn(z)
                ratio, dratio = dratio_fn(z, t1)
                return (-logp_0 - ratio).detach(), (-dlogp_0 - dratio).detach()

            for _ in range(num_steps_per_ais_step):
                # resample velocity
                current_v = torch.randn_like(current_z)
                z, v, initial_U, final_U = hmc.hmc_trajectory(
                    current_z=current_z,
                    current_v=current_v,
                    grad_U=grad_U,
                    epsilon=epsilon,
                    L=num_hmc_steps,
                )
                current_z, accept_hist = hmc.accept_reject(
                    current_z=current_z,
                    current_v=current_v,
                    z=z,
                    v=v,
                    accept_hist=accept_hist,
                    initial_U=initial_U,
                    final_U=final_U,
                    K=normalized_kinetic,
                )
                num_accept_reject += 1

        t1 = schedule[-1]

        # Let's continue to run the sampler to obtain more accurate samples

        @torch.enable_grad
        def grad_U(z):
            # assuming the base distribution is standard normal, which is true for all flows that we actually use
            logp_0 = logp_0_fn(z)
            dlogp_0 = dlogp_0_fn(z)
            ratio, dratio = dratio_fn(z, t1)
            return (-logp_0 - ratio).detach(), (-dlogp_0 - dratio).detach()

        for _ in tqdm(range(num_continue)):
            current_v = torch.randn_like(current_z)
            z, v, initial_U, final_U = hmc.hmc_trajectory(
                current_z=current_z,
                current_v=current_v,
                grad_U=grad_U,
                epsilon=epsilon,
                L=num_hmc_steps,
            )
            current_z, accept_hist = hmc.accept_reject(
                current_z=current_z,
                current_v=current_v,
                z=z,
                v=v,
                accept_hist=accept_hist,
                initial_U=initial_U,
                final_U=final_U,
                K=normalized_kinetic,
            )
            num_accept_reject += 1

        assert num_accept_reject == (
            (len(schedule) - 1) * num_steps_per_ais_step + num_continue
        )

        # adapted from losses.py
        train = False

        with torch.no_grad():
            if "none" not in flow_name:
                if flow_name in ["mintnet", "nice", "realnvp"]:
                    # map z -> x via flow, then rescale to [-1, 1]
                    ais_x = flow.module.sampling(current_z, rescale=True)
                else:
                    if "noise" in flow_name or "copula" in flow_name:
                        ais_x = flow.module.sample(
                            current_z.view(batch_size, -1),
                            context=None,
                            rescale=True,
                            transform=True,
                            train=train,
                        )
                    else:
                        ais_x = flow.module.sample(
                            current_z.view(batch_size, -1), context=None, rescale=True
                        )
            else:
                ais_x = current_z.view((-1, 1, 28, 28))

            ais_x = inverse_scaler(ais_x)

        logws.append(logw)
        samples.append(ais_x)
        z_samples.append(current_z)
        accept_hists.append(accept_hist)

    init_zs = torch.cat(init_zs, dim=0)
    logws = torch.cat(logws, dim=0)
    samples = torch.cat(samples, dim=0)
    z_samples = torch.cat(z_samples, dim=0)
    accept_hists = torch.cat(accept_hists, dim=0)

    assert init_zs.shape[0] == num_ais_samples
    assert logws.shape[0] == num_ais_samples
    assert samples.shape[0] == num_ais_samples
    assert accept_hists.shape[0] == num_ais_samples

    acceptance_rate = np.mean(accept_hists.cpu().numpy()) / num_accept_reject
    print(f"Acceptance rate: {acceptance_rate:.3f}")

    log_normalizer = utils.logmeanexp(
        logws.view(
            -1,
        ),
        dim=0,
    )
    if not forward:
        log_normalizer = -log_normalizer

    return samples, z_samples, init_zs, logws, log_normalizer, acceptance_rate
