import gc
import io
import os
import copy
import pickle
import time

import numpy as np
import logging

import sde_lib
from models.toy_networks import *
import toy_losses, toy_mi_losses
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import toy_datasets
import density_ratios
from absl import flags
import torch
import torch.autograd as autograd
from utils import save_checkpoint, restore_checkpoint, get_prob_path
import torch.optim as optim

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("poster")
sns.set_style("white")

import wandb

FLAGS = flags.FLAGS


# def seed_all(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    # Initialize model.
    score_model = mutils.create_model(config, name=config.model.name)
    assert config.model.ema is False  # this is overkill
    ema = None

    optimizer = toy_losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # create figures directory
    figures_dir = os.path.join(workdir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    metrics_dir = os.path.join(workdir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state["step"])

    # Build data iterators
    train_ds = toy_datasets.get_dataset(config)

    # Build one-step training and evaluation functions
    optimize_fn = toy_losses.toy_optimization_manager(config)
    joint = config.training.joint
    eps1 = config.data.eps1
    eps2 = config.data.eps2
    conditional = config.training.conditional
    if conditional:
        assert config.training.reweight == "obj_var"
    dsm = config.training.dsm
    data_dataset = config.data.dataset

    prob_path_name = config.training.prob_path
    prob_path = get_prob_path(config.data.dim, prob_path_name, config)
    one_sided = prob_path_name.startswith("One")

    batch_size = config.training.batch_size

    if joint:
        print("Using joint training!")
    sde = sde_lib.ToyInterpXt()

    # see if using a learning rate scheduler helps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        config.training.n_iters // config.training.eval_freq,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
    )

    # get appropriate functions
    eps_factor = 1.0 - eps1 - eps2

    if data_dataset == "GaussiansforMI":
        train_step_fn = toy_mi_losses.get_step_fn(
            sde=sde,
            train=True,
            eps1=eps1,
            eps2=eps2,
            eps_factor=eps_factor,
            joint=joint,
            dsm=dsm,
            optimize_fn=optimize_fn,
            reweight=config.training.reweight,
            conditional=conditional,
            prob_path=prob_path,
            device=config.device,
            batch_size=batch_size,
            full=config.training.full,
        )
    else:
        if not one_sided and config.training.use_two_sb:
            if config.training.two_sb_var == 0:
                interpolate_fn = train_ds.sample_sequence_on_the_fly_ot
            else:
                interpolate_fn = train_ds.sample_sequence_on_the_fly_sb
        else:
            interpolate_fn = train_ds.sample_sequence_on_the_fly
        train_step_fn = toy_losses.get_step_fn(
            sde=sde,
            train=True,
            eps1=eps1,
            eps2=eps2,
            eps_factor=eps_factor,
            joint=joint,
            dsm=dsm,
            optimize_fn=optimize_fn,
            reweight=config.training.reweight,
            conditional=conditional,
            prob_path=prob_path,
            factor=train_ds.factor,
            device=config.device,
            batch_size=batch_size,
            full=config.training.full,
            interpolate_fn=interpolate_fn,
        )
    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    # in case we are estimating mutual information
    if data_dataset == "GaussiansforMI":
        mi_db = []
        mse_errors = []
        val_mse_errors = []
        nfes = []
        mi_metrics = {
            "step": [],
            "mi": [],
            "nfe": [],
            "true_mi": train_ds.true_mutual_info,
        }
    else:
        mse_errors = {"step": [], "mse": [], "val_mse": [], "nfe": []}

    best_diff = np.inf
    best_step = 0

    if data_dataset == "GaussiansforMI":
        assert one_sided
        batch_fn = train_ds.sample_data_detach
    else:

        if one_sided:
            batch_fn = train_ds.one_sample
        else:
            batch_fn = train_ds.two_sample

    if data_dataset != "GaussiansforMI":
        val_evaluate_fn = get_toy_val_evaluate_fn(
            config, dataset=train_ds, device=config.device, prob_path=prob_path
        )
    else:
        val_evaluate_fn = get_mi_val_evaluate_fn(
            config, teacher=train_ds, device=config.device
        )

    all_times = []
    for step in range(initial_step, num_train_steps + 1):
        # n = config.training.batch_size
        if data_dataset == "GaussiansforMI":
            batch = batch_fn(n_samples=batch_size)

            t1 = time.perf_counter()
            loss_dict = train_step_fn(state, batch)
            all_times.append(time.perf_counter() - t1)
        else:
            # TODO: what is going on??
            # fix here
            batch = batch_fn(n=batch_size)
            # TODO: there are also some differences. right now timewise should work, but not joint

            t1 = time.perf_counter()
            loss_dict = train_step_fn(state, batch)
            all_times.append(time.perf_counter() - t1)

        # Execute one training step
        # loss_dict = train_step_fn(state, batch.detach())
        loss_dict["step"] = step
        wandb.log(loss_dict)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.4f" % (step, loss_dict["loss"]))

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0 and step > 0:
            if data_dataset != "GaussiansforMI":

                val_mse_error = val_evaluate_fn(score_model)
                mse_error, nfe = visualize(
                    config,
                    train_ds,
                    score_model,
                    savefig=figures_dir,
                    step=step,
                    device=config.device,
                )
                mse_errors["step"].append(step)
                mse_errors["mse"].append(mse_error)
                # let's add this here
                mse_errors["val_mse"].append(val_mse_error)
                mse_errors["nfe"].append(nfe)

                with open(os.path.join(metrics_dir, "metrics.p"), "wb") as fp:
                    pickle.dump(mse_errors, fp)
            else:

                val_mse_error = val_evaluate_fn(score_model)
                val_mse_errors.append(val_mse_error)
                est_mi, nfe = estimate_mi(
                    config, score_model, train_ds, device=config.device
                )
                mi_db.append(est_mi)
                nfes.append(nfe)

                mse_errors.append(np.square(est_mi - train_ds.true_mutual_info))

                visualize_mi(
                    config, mi_db, train_ds.true_mutual_info, savefig=figures_dir
                )
                # also save metrics
                mi_metrics["step"].append(step)
                mi_metrics["mi"] = mi_db
                # let's add this here
                mi_metrics["val_mse_error"] = val_mse_errors
                mi_metrics["nfe"] = nfes

                mi_metrics["mse_error"] = mse_errors

                # should you save checkpoints?
                diff = np.abs(mi_db[-1] - train_ds.true_mutual_info)
                if diff <= best_diff:
                    best_diff = diff
                    best_step = step
                    mi_metrics["best_diff"] = best_diff
                    mi_metrics["best_step"] = best_step
                    fpath = os.path.join(checkpoint_dir, "best_ckpt.pth")
                else:
                    fpath = os.path.join(checkpoint_dir, "ckpt.pth")
                torch.save(score_model.state_dict(), fpath)

                # save metrics
                with open(os.path.join(metrics_dir, "metrics.p"), "wb") as fp:
                    pickle.dump(mi_metrics, fp)

                # take a scheduler step
                if config.optim.scheduler:
                    scheduler.step()

    if num_train_steps >= config.training.eval_freq:
        if data_dataset != "GaussiansforMI":
            temp = mse_errors["val_mse"]
            index = np.argmin(temp)
            print(
                f"Best MSE error on val set: {temp[index]} at {mse_errors['step'][index]}"
            )
        else:
            temp = mi_metrics["val_mse_error"]
            index = np.argmin(temp)
            print(
                f"Best MSE error on val set: {temp[index]} at {mi_metrics['step'][index]}"
            )

        with open(os.path.join(metrics_dir, "all_times.p"), "wb") as fp:
            pickle.dump(all_times, fp)
        print(f"Total training time: {np.sum(all_times)}")


def get_toy_val_evaluate_fn(config, dataset, device, prob_path=None):
    # seed_all(1)
    # qs = dataset.q.sample((5000,))
    # ps = dataset.p.sample((5000,))
    # mesh = torch.cat([qs, ps])
    # val_dir = os.path.join(workdir, "val")
    # os.makedirs(val_dir, exist_ok=True)
    # torch.save(mesh, os.path.join(val_dir, "val_mesh.pt"))
    # seed_all(config.seed)
    if config.data.dataset != "GMMs":
        mesh = torch.load(
            os.path.join("val_sets", f"{config.data.dataset}_{config.data.dim}.pt"),
            map_location=device,
        )
    else:
        mesh = torch.load(
            os.path.join(
                "val_sets",
                f"{config.data.dataset}_{config.data.dim}_{config.data.k}.pt",
            ),
            map_location=device,
        )

    logr_true = dataset.log_density_ratios(mesh.to(device)).squeeze().numpy()

    density_ratio_fn = density_ratios.get_toy_density_ratio_fn(
        rtol=config.eval.rtol,
        atol=config.eval.atol,
        eps1=config.data.eps1,
        eps2=config.data.eps2,
    )

    def val_evaluate(model):
        print("-----")
        print("val set")
        est_logr, _ = density_ratio_fn(
            model.to(device), mesh, score_type=config.model.type
        )
        print("true log ratios:", np.min(logr_true), np.max(logr_true))
        print("est. log ratios:", np.min(est_logr), np.max(est_logr))
        val_mse = np.mean(np.square(est_logr - logr_true))
        print(f"MSE on val set: {val_mse}")
        print("-----")
        return val_mse

    return val_evaluate


def get_mi_val_evaluate_fn(config, teacher, device):
    # mi_true = teacher.true_mutual_info
    # seed_all(1)
    # n = 10000
    # samples = teacher.sample_data(n).to(device)
    # val_dir = os.path.join(workdir, "val")
    # os.makedirs(val_dir, exist_ok=True)
    # torch.save(samples, os.path.join(val_dir, "val_samples.pt"))
    # seed_all(config.seed)

    samples = torch.load(
        f"val_sets/{config.data.dataset}_{config.data.dim}.pt", map_location=device
    )

    density_ratio_fn = density_ratios.get_toy_density_ratio_fn(
        rtol=config.eval.rtol,
        atol=config.eval.atol,
        eps1=config.data.eps1,
        eps2=config.data.eps2,
    )

    emp_mi = teacher.empirical_mutual_info(samples)
    score_type = config.model.type

    def val_evaluate(model):
        print("-----")
        print("val set")
        est_mi, _ = density_ratio_fn(model.to(device), samples, score_type=score_type)
        est_mi = np.mean(est_mi)
        val_mse = np.square(emp_mi - est_mi)
        # print("true MI", mi_true)
        print("empirical MI", emp_mi)
        print("est MI", est_mi)
        print(f"MSE on val set: {val_mse}")
        if est_mi < 0.0:
            raise Exception("Diverged! Terminating")
        print("-----")
        return val_mse

    return val_evaluate


def visualize(config, dataset, model, savefig=None, step=None, device=None):
    model.eval()
    data_dataset = config.data.dataset

    with torch.no_grad():
        # Build density ratio estimation functions
        density_ratio_fn = density_ratios.get_toy_density_ratio_fn(
            rtol=config.eval.rtol,
            atol=config.eval.atol,
            eps1=config.data.eps1,
            eps2=config.data.eps2,
        )

        if data_dataset == "PeakedGaussians":
            grid_size = 10000
            left_bound = -2
            right_bound = 2
            mesh = (
                torch.linspace(left_bound, right_bound, grid_size)
                .view(-1, 1)
                .to(device)
            )
        elif data_dataset == "Checkerboard":
            mesh = dataset.q.sample((10000,))
        else:
            # what if instead of a mesh you just sampled from both datasets
            qs = dataset.q.sample((5000,))
            ps = dataset.p.sample((5000,))
            mesh = torch.cat([qs, ps])
            # if device is not None:
            #   mesh = mesh.to(device)

        plt.figure(figsize=(8, 5))

        # plot data
        logr_true = dataset.log_density_ratios(mesh.to(device)).squeeze().numpy()

        # plot estimated ratios
        print("-----")
        est_logr, nfe = density_ratio_fn(
            model.to(device), mesh, score_type=config.model.type
        )

        if data_dataset in ["GaussiansforMI", "PeakedGaussians"]:
            plt.scatter(mesh.squeeze().cpu().numpy(), est_logr, label="est", s=10)
            plt.scatter(mesh.squeeze().cpu().numpy(), logr_true, label="true", s=10)
        else:
            plt.hist(est_logr, bins=50, label="est", alpha=0.7)
            plt.hist(logr_true, bins=50, label="true", alpha=0.7)
        plt.legend()
        sns.despine()
        plt.tight_layout()

        if savefig is not None:
            plt.savefig(
                savefig + "/{}_log_ratios.png".format(step), bbox_inches="tight"
            )
            plt.close()
        else:
            plt.show()

        print("true log ratios:", np.min(logr_true), np.max(logr_true))
        print("est. log ratios:", np.min(est_logr), np.max(est_logr))
        mse_error = np.mean(np.square(est_logr - logr_true))
        print("MSE error:", mse_error)
        print("-----")

        if config.training.plot_scatter:
            np_mesh = mesh.detach().cpu().numpy()
            logp = dataset.p.log_prob(mesh).detach().cpu().numpy() + est_logr

            # GPT hack
            fig, ax = plt.subplots(figsize=(8, 6))

            scatter = ax.scatter(
                np_mesh[..., 0], np_mesh[..., 1], c=logp, cmap="viridis"
            )

            ax.set_aspect("equal", "box")

            cbar = fig.colorbar(scatter, ax=ax)

            plt.axis("off")
            if savefig is not None:
                plt.savefig(
                    savefig + "/{}_scatter.pdf".format(step), bbox_inches="tight"
                )
                plt.close()
            else:
                plt.show()

        return mse_error, nfe


def estimate_mi(config, model, teacher, device):
    # Build density ratio estimation functions
    # technically we can safely use eps1=eps2=0.0 here
    density_ratio_fn = density_ratios.get_toy_density_ratio_fn(
        rtol=config.eval.rtol,
        atol=config.eval.atol,
        eps1=config.data.eps1,
        eps2=config.data.eps2,
    )

    model.eval()
    mi_true = teacher.true_mutual_info
    print("computing mutual information estimates...")

    with torch.no_grad():
        n = 10000
        samples = teacher.sample_data(n).to(device)
        emp_mi = teacher.empirical_mutual_info(samples)
        print("-----")
        est_mi, nfe = density_ratio_fn(
            model.to(device), samples, score_type=config.model.type
        )
        est_mi = np.mean(est_mi)

        print("true MI", mi_true)
        print("empirical MI", emp_mi)
        print("est MI", est_mi)
        print("-----")

    return est_mi, nfe


def visualize_mi(config, mi_db, mi_true, savefig=None):

    plt.figure(figsize=(12, 5))
    plt.plot(
        range(0, len(mi_db) * config.training.eval_freq, config.training.eval_freq),
        mi_db,
        "-o",
        label="est. MI",
    )
    plt.hlines(
        mi_true,
        xmin=0,
        xmax=int(len(mi_db) * config.training.eval_freq),
        color="black",
        label="true MI",
    )
    plt.legend(loc="lower right")
    sns.despine()
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig + "/mi.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def time_score(log_prob_fn, x, t):
    t.requires_grad_(True)
    y = log_prob_fn(x, t).sum()
    return autograd.grad(y, t, create_graph=True)[0]
