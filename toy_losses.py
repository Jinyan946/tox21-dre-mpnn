import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == "Adam":
        optimizer = optim.Adam(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
            amsgrad=config.optim.amsgrad,
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )

    return optimizer


def toy_optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(
        optimizer,
        params,
        step,
        lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip,
    ):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g["lr"] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_toy_joint_score_estimation(
    prob_path, likelihood_weighting, factor, eps1, eps2, device, batch_size
):
    t0 = torch.zeros((batch_size, 1), device=device) + eps1
    t1 = torch.ones((batch_size, 1), device=device) - eps2

    data_weighting_quantities = lambda t: 1 - t**2

    time_weighting_quantities = prob_path.get_time_weighting_quantities(
        likelihood_weighting=likelihood_weighting, t0=t0, t1=t1, eps=eps1, factor=factor
    )

    def toy_joint_score_estimation(scorenet, samples, t):
        """
        in objective, T = [0, 1]
        px, qx, xt: (batch_size, 1)
        t: (batch_size, 1)
        """
        # sample appropriate data
        px, qx, xt = samples

        # get data score -- this is SSM!
        xt.requires_grad_()
        t.requires_grad_()
        vectors = torch.randn_like(xt, device=device)
        score_x, score_t = scorenet(xt, t)
        # grad1 = torch.cat([score_x, score_t], dim=-1)
        gradv = torch.sum(score_x * vectors)
        grad2 = autograd.grad(gradv, xt, create_graph=True)[0]

        lambda_t = data_weighting_quantities(t)

        # SSM loss. Note: score_t has been moved outside
        # ssm_loss1 = torch.sum(grad1 * grad1, dim=-1, keepdim=True) / 2.0 * lambda_t
        ssm_loss1 = 0.5 * torch.sum(score_x * score_x, dim=-1, keepdim=True) * lambda_t
        ssm_loss2 = torch.sum(vectors * grad2, dim=-1, keepdim=True) * lambda_t
        ssm_loss = ssm_loss1 + ssm_loss2

        # TSM loss
        lambda_t, lambda_t0, lambda_t1, lambda_dt = time_weighting_quantities(t=t)
        # reweighted version
        term1 = (scorenet(px, t0)[-1]) * lambda_t0  # T=0 is noise
        term2 = (scorenet(qx, t1)[-1]) * lambda_t1  # T=1 is data

        # need to differentiate score wrt t
        xt_score_dt = autograd.grad(score_t.sum(), t, create_graph=True)[0]
        term3 = xt_score_dt * lambda_t
        term4 = score_t * lambda_dt
        term5 = 0.5 * score_t**2 * lambda_t

        time_loss = term1 - term2 + term3 + term4 + term5
        loss = ssm_loss + time_loss

        # 1-d so we can just take the mean rather than summing
        return loss.mean()

    return toy_joint_score_estimation


def get_toy_timewise_score_estimation(
    prob_path,
    likelihood_weighting,
    factor,
    eps1,
    eps2,
    eps_factor,
    device,
    batch_size,
    interpolate_fn,
):
    t0 = torch.zeros((batch_size, 1), device=device) + eps1
    t1 = torch.ones((batch_size, 1), device=device) - eps2

    time_weighting_quantities = prob_path.get_time_weighting_quantities(
        likelihood_weighting=likelihood_weighting,
        t0=t0,
        t1=t1,
        eps1=eps1,
        eps2=eps2,
        factor=factor,
    )

    def toy_timewise_score_estimation(scorenet, samples):
        """
        in objective, T = [0, 1]
        px, qx, xt: (batch_size, 1)
        t: (batch_size, 1)

        we are reweighting the output of the score network (most recent version)
        """
        t = torch.rand(batch_size, 1, device=device) * eps_factor + eps1

        # Note: using squeeze()s were unnecessary, and are removed
        px, qx, xt = interpolate_fn(*samples, t)

        # reweighted version

        lambda_t, lambda_t0, lambda_t1, lambda_dt = time_weighting_quantities(t=t)

        term1 = 2 * scorenet(px, t0) * lambda_t0
        term2 = 2 * scorenet(qx, t1) * lambda_t1

        # need to differentiate score wrt t
        t.requires_grad_(True)
        xt_score = scorenet(xt, t)  # dim = 1
        xt_score_dt = autograd.grad(xt_score.sum(), t, create_graph=True)[0]
        term3 = 2 * xt_score_dt * lambda_t
        # fix
        term4 = 2 * xt_score * lambda_dt
        term5 = xt_score**2 * lambda_t

        loss = term1 - term2 + term3 + term4 + term5

        # 1-d so we can just take the mean rather than summing
        # Use this to unify the API
        return loss.mean()

    return toy_timewise_score_estimation


def get_cat_toy_timewise_score_estimation(
    prob_path,
    likelihood_weighting,
    factor,
    eps1,
    eps2,
    eps_factor,
    device,
    batch_size,
    interpolate_fn,
):
    print("Using torch.cat")
    t0 = torch.zeros((batch_size, 1), device=device) + eps1
    t1 = torch.ones((batch_size, 1), device=device) - eps2

    time_weighting_quantities = prob_path.get_time_weighting_quantities(
        likelihood_weighting=likelihood_weighting,
        t0=t0,
        t1=t1,
        eps1=eps1,
        eps2=eps2,
        factor=factor,
    )

    def toy_timewise_score_estimation(scorenet, samples):
        """
        in objective, T = [0, 1]
        px, qx, xt: (batch_size, 1)
        t: (batch_size, 1)

        we are reweighting the output of the score network (most recent version)
        """
        t = torch.rand(batch_size, 1, device=device) * eps_factor + eps1
        # Note: using squeeze()s were unnecessary, and are removed
        px, qx, xt = interpolate_fn(*samples, t)

        # reweighted version

        lambda_t, lambda_t0, lambda_t1, lambda_dt = time_weighting_quantities(t=t)

        t.requires_grad_(True)
        xs = torch.cat([px, qx, xt], dim=0)
        ts = torch.cat([t0, t1, t], dim=0)

        scores = scorenet(xs, ts)
        scores1 = scores[:batch_size, :]
        scores2 = scores[batch_size : 2 * batch_size, :]
        scores3 = scores[2 * batch_size :, :]

        term1 = 2 * scores1 * lambda_t0
        term2 = 2 * scores2 * lambda_t1

        # need to differentiate score wrt t
        xt_score = scores3  # dim = 1
        xt_score_dt = autograd.grad(xt_score.sum(), t, create_graph=True)[0]
        term3 = 2 * xt_score_dt * lambda_t
        # fix
        term4 = 2 * xt_score * lambda_dt
        term5 = xt_score**2 * lambda_t

        loss = term1 - term2 + term3 + term4 + term5

        # 1-d so we can just take the mean rather than summing
        # Use this to unify the API
        return loss.mean()

    return toy_timewise_score_estimation


def get_toy_c_timewise_score_estimation(
    prob_path,
    likelihood_weighting,
    factor,
    eps1,
    eps2,
    eps_factor,
    batch_size,
    device,
    full=False,
):
    if likelihood_weighting != "obj_var":
        raise NotImplementedError

    if full:

        def loss_fn(scorenet, epsilon, xs, t, mean, std):
            xt = epsilon * std + mean
            lambda_t, targets = prob_path.full_epsilon_target(epsilon, *xs, t, factor)
            ctsm_loss = torch.mean(
                torch.square(targets - lambda_t * scorenet.forward_full(xt, t)),
                dim=-1,
            )
            return ctsm_loss

    else:

        def loss_fn(scorenet, epsilon, xs, t, mean, std):
            xt = epsilon * std + mean
            lambda_t, targets = prob_path.epsilon_target(epsilon, *xs, t, factor)
            ctsm_loss = torch.square(targets - lambda_t * scorenet(xt, t))
            return ctsm_loss

    def toy_c_timewise_score_estimation(scorenet, samples):

        t = torch.rand(batch_size, 1, device=device) * eps_factor + eps1

        mean, std, var = prob_path.marginal_prob(*samples, t)
        epsilon = torch.randn((batch_size, prob_path.dim), device=device)

        # Note: lambda_t here has a different interpretation from Choi et al.
        loss = loss_fn(scorenet, epsilon, samples, t, mean, std)

        return loss.mean()

    return toy_c_timewise_score_estimation


def get_step_fn(
    sde,
    train,
    eps1,
    eps2,
    eps_factor,
    joint=False,
    dsm=False,
    optimize_fn=None,
    reweight=False,
    conditional=False,
    prob_path=None,
    factor=1.0,
    device=torch.device("cpu"),
    batch_size=None,
    full=False,
    interpolate_fn=None,
):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE. (not used here)
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if not joint:
        # loss_fn = time_loss
        if not conditional:
            loss_fn = get_toy_timewise_score_estimation(
                prob_path=prob_path,
                likelihood_weighting=reweight,
                factor=factor,
                eps1=eps1,
                eps2=eps2,
                eps_factor=eps_factor,
                device=device,
                batch_size=batch_size,
                interpolate_fn=interpolate_fn,
            )
        else:
            loss_fn = get_toy_c_timewise_score_estimation(
                prob_path=prob_path,
                likelihood_weighting=reweight,
                factor=factor,
                eps1=eps1,
                eps2=eps2,
                eps_factor=eps_factor,
                batch_size=batch_size,
                device=device,
                full=full,
            )
    else:
        # should not use these (yet)
        raise NotImplementedError

    # if reweight:
    #     print("reweighting loss function!")

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        model = state["model"]
        if train:
            model.train()
            optimizer = state["optimizer"]
            optimizer.zero_grad()
            # if joint:
            #     # loss, loss1, loss2, loss3, loss4, edge1, edge2 = loss_fn(model, batch, t)
            #     loss = loss_fn(model, batch, t, likelihood_weighting=reweight)
            # else:
            #     loss, loss1, loss2, loss3, edge1, edge2 = loss_fn(
            #         model, batch, t, likelihood_weighting=reweight
            #     )
            # if not conditional:
            #     loss = loss_fn(model, batch, t)
            # else:
            #     loss = loss_fn(model, batch, t)
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state["step"])
            state["step"] += 1
        else:
            model.eval()
            with torch.no_grad():
                # if joint:
                #     # loss, loss1, loss2, loss3, loss4, edge1, edge2 = loss_fn(model, batch, t)
                #     loss = loss_fn(model, batch, t, likelihood_weighting=reweight)
                # else:
                #     loss, loss1, loss2, loss3, edge1, edge2 = loss_fn(
                #         model, batch, t, likelihood_weighting=reweight
                #     )
                # if not conditional:
                #     loss = loss_fn(model, batch, t)
                # else:
                #     loss = loss_fn(model, batch, t)
                loss = loss_fn(model, batch)
        # return loss in a single dictionary
        loss_dict = {
            "loss": loss.item(),
            # 'loss1': loss1.item(),
            # 'loss2': loss2.item(),
            # 'loss3': loss3.item(),
            # 'edge1': edge1.item(),
            # 'edge2': edge2.item()
        }
        # ugh
        # if joint:
        #   loss_dict['loss4'] = loss4.item()
        return loss_dict

    return step_fn
