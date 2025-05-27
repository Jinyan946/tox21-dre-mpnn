"""
NOTE: this training code is specific to the RQ-NSF interpolation mechanism!
"""

import gc
import io
import os
import time
import copy

import numpy as np
import logging
from models import ncsn_flow, ncsn_unet
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets

# from evaluations import ais
import likelihood
import sde_lib
from absl import flags
import torch
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint, load_history, get_prob_path
import wandb
import density_ratios
import matplotlib.pyplot as plt
import pickle

FLAGS = flags.FLAGS


# first, load some flow-specific code
import sys

top_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(top_path, "nsf"))


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    metrics_dir = os.path.join(workdir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    if config.optim.manager == "v1":
        # this is what it always has been
        optimize_fn = losses.optimization_manager(config)
    else:
        print("using slightly different optimization schedule")
        optimize_fn = losses.v2_optimization_manager(config)
    history = None

    # importance weighting via the loss history buffer
    if config.training.iw:
        print("using history for dynamic reweighting!")
        from loss_history import (
            LossSecondMomentResampler,
            InterpolateLossSecondMomentResampler,
        )

        print(
            "using history buffer with size {} and batch size of {}!".format(
                config.training.buffer_size, config.training.batch_size
            )
        )
        if not config.training.interpolate:
            history = LossSecondMomentResampler(
                batch_size=config.training.batch_size,
                history_per_term=config.training.buffer_size,
            )
        else:
            history = InterpolateLossSecondMomentResampler(
                batch_size=config.training.batch_size,
                history_per_term=config.training.buffer_size,
            )
    else:
        # no reweighting and no importance sampling
        assert history is None
        print("no importance weighting on the loss!")
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Resume training when intermediate checkpoints are detected
    if config.training.resume_ckpt > 0:
        resume_ckpt = config.training.resume_ckpt
        print("resume training from ckpt {}".format(resume_ckpt))
        state = restore_checkpoint(
            os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(resume_ckpt)),
            state,
            config.device,
        )
        initial_step = int(state["step"]) - 1
    else:
        state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
        initial_step = int(state["step"]) - 1

    # automatically check if we need to reload the buffer after loading checkpoint
    if initial_step > 0:
        if history:
            load_history(workdir, history, interpolate=config.training.interpolate)
            print("reloaded pre-saved history to continue training!")

    # Build data iterators
    logging.info("Loading MNIST dataset to be encoded using the flow!")
    train_ds, eval_ds = datasets.get_dataset_for_flow(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
    )
    # Create data normalizer and its inverse
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # load pre-trained normalizing flow checkpoint
    if config.training.z_space:
        logging.info("Loading pre-trained flow checkpoint...")
        flow = ncsn_flow.load_pretrained_flow(config)
        if config.training.z_space_model != "rq_nsf_none":
            flow.eval()  # no training
    else:
        flow = None

    flow_name = config.training.z_space_model

    # Setup SDEs
    if config.training.sde.lower() == "z_vpsde":
        # TODO: check if we need to feed in the flow
        assert flow is not None
        print("using variant of Z_RQNSF_VPSDE due to awkward preprocessing!")
        if "noise" in flow_name or "copula" in flow_name:
            sde = sde_lib.Z_RQNSF_TFORM_VPSDE(
                flow,
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales,
            )
        else:
            # TODO: this is just with the RQNSF-flow, but i've trained it after scaling the time label.
            # try adjusting this later
            print("BE CAREFUL HERE!!!! TODO")
            sde = sde_lib.Z_RQNSF_VPSDE(
                flow,
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales,
            )
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    prob_path = get_prob_path(784, config.training.prob_path, config)

    train_eps = config.training.eps

    # get appropriate loss function
    from losses import get_step_fn

    # Build one-step training and evaluation functions
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    joint = config.training.joint
    alpha = config.optim.alpha
    algo = config.training.algo
    z_space = config.training.z_space
    z_interpolate = config.training.z_interpolate
    mlp = True if "mlp" in config.model.name else False
    iw = config.training.iw
    interpolate = config.training.interpolate
    if interpolate:
        print("trying out interpolation during training!")

    conditional = config.training.conditional

    likelihood_weighting = config.training.likelihood_weighting
    if conditional:
        assert likelihood_weighting == "obj_var"

    use_zt = config.training.use_zt
    train_step_fn = get_step_fn(
        sde,
        train=True,
        algo=algo,
        joint=joint,
        z_space=z_space,
        iw=iw,
        mlp=mlp,
        alpha=alpha,
        z_interpolate=z_interpolate,
        optimize_fn=optimize_fn,
        eps=train_eps,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
        resample_t=config.training.resample_t,
        flow=flow,
        flow_name=flow_name,
        history=history,
        interpolate=interpolate,
        prob_path=prob_path,
        conditional=conditional,
        # batch_size=config.training.batch_size,
        device=config.device,
        use_zt=use_zt,
        epsilons=config.training.epsilons,
    )
    eval_step_fn = get_step_fn(
        sde,
        train=False,
        algo=algo,
        joint=joint,
        z_space=z_space,
        iw=iw,
        mlp=mlp,
        alpha=alpha,
        z_interpolate=z_interpolate,
        optimize_fn=optimize_fn,
        eps=train_eps,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
        resample_t=config.training.resample_t,
        flow=flow,
        flow_name=flow_name,
        history=history,
        interpolate=interpolate,
        prob_path=prob_path,
        conditional=conditional,
        # batch_size=config.eval.batch_size,
        device=config.device,
        use_zt=use_zt,
        epsilons=config.training.epsilons,
    )
    # TODO: also need to fix likelihood fn and dre_v2 fn for z-space joint training
    likelihood_fn = likelihood.get_likelihood_fn_flow(sde, inverse_scaler)
    if config.training.algo != "baseline":
        if not config.training.z_space:
            density_ratio_fn = density_ratios.get_density_ratio_fn(
                sde, inverse_scaler, eps=train_eps
            )
        else:
            if z_interpolate:
                density_ratio_fn = density_ratios.get_z_interp_density_ratio_fn_flow(
                    sde,
                    inverse_scaler,
                    mlp=mlp,
                    # rtol=config.eval.rtol,
                    # atol=config.eval.atol,
                    # eps=train_eps,
                    use_zt=use_zt,
                    flow=flow,
                    z_space_model_name=flow_name,
                    prob_path=prob_path,
                    conditional=conditional,
                    epsilons=config.training.epsilons,
                )
            else:
                density_ratio_fn = density_ratios.get_density_ratio_fn_flow(
                    sde, inverse_scaler, eps=train_eps
                )
    if config.training.dre_bpd_v2:
        density_ratio_fn_pathwise = (
            density_ratios.get_z_interp_pathwise_density_ratio_fn(
                sde, inverse_scaler, eps=train_eps
            )
        )

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.training.batch_size,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps
        )

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))
    logging.info("Using model type %s." % config.model.name)
    if joint:
        logging.info("Using alpha %.3e for joint training" % (alpha))
    if config.training.rescale_t:
        logging.info("rescaling output of time score network!")

    all_dre_bpds = dict()
    all_checkpoint_steps = dict()
    all_times = []
    for step in range(initial_step, num_train_steps + 1):
        try:
            batch, _ = next(train_iter)  # ignore labels
        except StopIteration:
            train_iter = iter(train_ds)
            batch, _ = next(train_iter)
        batch = batch.to(config.device).float()

        # add uniform noise, then rescale to [-1, +1]
        # NOTE: should flip the order for adding gaussian noise
        batch = batch * 255.0 / 256.0
        batch += torch.rand_like(batch) / 256.0

        # automatically assuming we'll be doing z_interpolate
        # rescale to [-1, 1]
        batch = scaler(batch)

        # Execute one training step
        t1 = time.perf_counter()
        summary = train_step_fn(state, batch.detach())
        all_times.append(time.perf_counter() - t1)

        summary["step"] = step
        wandb.log(summary)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, summary["loss"]))

        # visualize weights if possible
        if "weights" in summary:
            weights = summary["weights"].detach().cpu().numpy()
            plt.hist(weights.reshape(-1), bins="auto")
            plt.savefig(os.path.join(workdir, "weights_is.png"))
            plt.close()

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            try:
                eval_batch, _ = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_ds)
                eval_batch, _ = next(eval_iter)
            eval_batch = eval_batch.to(config.device).float()

            # uniform dequantization then [-1, 1] rescaling
            eval_batch = eval_batch * 255.0 / 256.0
            eval_batch += torch.rand_like(eval_batch) / 256.0

            # if invert_flow or z_interpolate:  # p(x) = flow
            eval_batch = scaler(eval_batch)
            log_det_logit = torch.zeros(len(eval_batch), device=config.device)
            flow_log_det = torch.zeros_like(log_det_logit)

            # NOTE: no additional dequantization on z embeddings!
            # dre_eval_batch = copy.copy(eval_batch)
            dre_eval_batch = eval_batch.detach().clone()
            eval_loss = eval_step_fn(state, eval_batch)
            # if not history:
            summary = dict(test_loss=eval_loss["loss"], step=step)
            # else:
            #   summary = dict(
            #     test_loss=eval_loss['loss'],
            #     unweighted_test_loss=eval_loss['unweighted_loss'],
            #     step=step
            #   )
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss["loss"]))

            # only compute density ratios when network is sufficiently smooth
            if step > 100 and step % config.training.ratio_freq == 0:
                if config.eval.enable_bpd:
                    # use EMA for ratio computation
                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())

                    # different types of density ratios for energy-based modeling
                    if config.training.pf_ode_bpd:
                        bpd = likelihood_fn(
                            score_model, dre_eval_batch, flow_log_det, log_det_logit
                        )[0]
                        if len(bpd) > 1:
                            bpd = bpd.detach().cpu().numpy().reshape(-1)
                            summary["test_bpds"] = bpd.mean()
                        else:
                            summary["test_bpds"] = bpd.item()
                        logging.info("step: %d, eval_bpd: %.5f" % (step, bpd.mean()))
                    if config.training.dre_bpd:
                        # TODO TODO TODO
                        # dre_bpd = \
                        #     density_ratio_fn(score_model=score_model, flow=flow, x=dre_eval_batch)[0]
                        # IS
                        dre_bpd = density_ratio_fn(
                            score_model=score_model, x=dre_eval_batch
                        )[0]
                        # dre_bpd = dre_bpd.reshape(-1)
                        # summary['test_dre_bpds'] = dre_bpd.mean()
                        summary["test_dre_bpds"] = (
                            dre_bpd.item()
                        )  # TODO: changed this to sum
                        logging.info(
                            "step: %d, eval_dre_bpd: %.5f" % (step, dre_bpd.mean())
                        )

                        all_dre_bpds[step] = dre_bpd.mean()

                    if config.training.dre_bpd_v2:
                        dre_bpd_v2 = density_ratio_fn_pathwise(
                            score_model=score_model, flow=flow, x=dre_eval_batch
                        )[0]
                        dre_bpd_v2 = dre_bpd_v2.reshape(-1)
                        summary["test_dre_bpds_v2"] = dre_bpd_v2.mean()
                        logging.info(
                            "step: %d, eval_dre_bpd_v2: %.5f"
                            % (step, dre_bpd_v2.mean())
                        )

                    ema.restore(score_model.parameters())

            wandb.log(summary)

        # Save a checkpoint periodically and generate samples if needed
        if (
            step != 0
            and step % config.training.snapshot_freq == 0
            or step == num_train_steps
        ):
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth"), state
            )

            all_checkpoint_steps[step] = save_step

            # save weights
            if history:
                if interpolate:
                    weights = history._weight_history[:, -1]
                else:
                    weights = history.weights()
                weights /= weights.max()
                plt.hist(weights.reshape(-1), bins="auto")
                # TODO: make weights a separate directory to avoid clutter
                plt.savefig(os.path.join(workdir, "weights_is_{}.png".format(step)))
                plt.close()

                # save everything in the buffer
                loss_history = history._loss_history
                time_history = history._time_history
                loss_counts = history._loss_counts
                np_record = {
                    "weights": weights,
                    "loss_history": loss_history,
                    "time_history": time_history,
                    "loss_counts": loss_counts,
                }
                if interpolate:
                    np_record["weight_history"] = history._weight_history
                np.savez(os.path.join(workdir, "history"), **np_record)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                # log generations to wandb
                wandb.log({"samples": [wandb.Image(i) for i in sample[0:64]]})
                ema.restore(score_model.parameters())
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                os.makedirs(this_sample_dir, exist_ok=True)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(
                    sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255
                ).astype(np.uint8)
                with open(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                    np.save(fout, sample)

                with open(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    save_image(image_grid, fout)

    with open(os.path.join(metrics_dir, "all_dre_bpds.p"), "wb") as fp:
        pickle.dump(all_dre_bpds, fp)
    with open(os.path.join(metrics_dir, "all_checkpoint_steps.p"), "wb") as fp:
        pickle.dump(all_checkpoint_steps, fp)
    with open(os.path.join(metrics_dir, "all_times.p"), "wb") as fp:
        pickle.dump(all_times, fp)
    print(f"Total training time: {np.sum(all_times)}")


def evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # Build data pipeline
    # train_ds, eval_ds, _ = datasets.get_dataset(config,
    #                                             uniform_dequantization=config.data.uniform_dequantization,
    #                                             evaluation=True)
    eval_ds = datasets.get_test_set_for_flow(config)

    # load pre-trained normalizing flow checkpoint
    if config.training.z_space:
        logging.info("Loading pre-trained flow checkpoint...")
        flow = ncsn_flow.load_pretrained_flow(config, test=True)
        if config.training.z_space_model != "rq_nsf_none":
            flow.eval()  # no training
    else:
        flow = None

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    conditional = config.training.conditional
    prob_path = get_prob_path(784, config.training.prob_path, config)
    epsilons = config.training.epsilons

    train_eps = config.training.eps

    flow_name = config.training.z_space_model

    use_zt = config.training.use_zt

    # Setup SDEs
    if config.training.sde.lower() == "z_vpsde":
        assert flow is not None
        print("using variant of Z_RQNSF_VPSDE due to awkward preprocessing!")
        if "noise" in flow_name or "copula" in flow_name:
            # regular MLP, no special time embedding
            print("not using sinusoidal positional embeddings for t")
            sde = sde_lib.Z_RQNSF_TFORM_VPSDE(
                flow,
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales,
            )
        else:
            sde = sde_lib.Z_RQNSF_VPSDE(
                flow,
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales,
            )
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting
        flow_name = config.training.z_space_model

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(
            sde,
            train=False,
            optimize_fn=optimize_fn,
            reduce_mean=reduce_mean,
            continuous=continuous,
            likelihood_weighting=likelihood_weighting,
            flow=flow,
            flow_name=flow_name,
        )

    # if config.eval.bpd_dataset.lower() == 'train':
    #   ds_bpd = train_ds_bpd
    #   bpd_num_repeats = 1
    if config.eval.bpd_dataset.lower() == "test":
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds_bpd = eval_ds
        # bpd_num_repeats = 5
        bpd_num_repeats = 1  # let's just do once for now
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd:
        likelihood_fn = likelihood.get_likelihood_fn_flow(sde, inverse_scaler)
        if config.training.algo != "baseline":
            if config.training.sde.lower() in ["interpxt", "flow_interpxt"]:
                if not config.training.z_space:
                    density_ratio_fn = density_ratios.get_interp_density_ratio_fn(
                        sde, inverse_scaler
                    )
                else:
                    density_ratio_fn = density_ratios.get_interp_density_ratio_fn_flow(
                        sde,
                        inverse_scaler,
                        prob_path=prob_path,
                        conditional=conditional,
                    )
            else:
                if not config.eval.ais:
                    density_ratio_fn = density_ratios.get_z_interp_density_ratio_fn_flow(
                        sde,
                        inverse_scaler,
                        # rtol=config.eval.rtol,
                        # atol=config.eval.atol,
                        # eps=train_eps,
                        use_zt=use_zt,
                        flow=flow,
                        z_space_model_name=flow_name,
                        prob_path=prob_path,
                        conditional=conditional,
                        epsilons=config.training.epsilons,
                    )
                else:
                    # TODO: complete AIS density ratio evaluation
                    # raise NotImplementedError
                    density_ratio_fn = density_ratios.get_ais_z_interp_density_ratio_fn_flow(
                        sde,
                        inverse_scaler,
                        # rtol=config.eval.rtol,
                        # atol=config.eval.atol,
                        # eps=train_eps,
                        use_zt=use_zt,
                        flow=flow,
                        z_space_model_name=flow_name,
                        prob_path=prob_path,
                        conditional=conditional,
                        epsilons=config.training.epsilons,
                    )
        if config.training.dre_bpd_v2:
            density_ratio_fn_v2 = density_ratios.get_z_interp_pathwise_density_ratio_fn(
                sde, inverse_scaler
            )

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (
            config.eval.batch_size,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, flow, flow_name, inverse_scaler, sampling_eps
        )

    # Use inceptionV3 for images with resolution higher than 256.
    # inceptionv3 = config.data.image_size >= 256
    # inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}.pth")
        # try:
        state = restore_checkpoint(ckpt_path, state, device=config.device, test=True)
        # except:
        # time.sleep(60)
        # try:
        #     state = restore_checkpoint(
        #         ckpt_path, state, device=config.device, test=True
        #     )
        # except:
        #     time.sleep(120)
        #     state = restore_checkpoint(
        #         ckpt_path, state, device=config.device, test=True
        #     )

        print("checkpoint is from step {}".format(state["step"]))
        ema.copy_to(score_model.parameters())
        # print('turned EMA off')

        if config.eval.enable_sampling:
            score_fn = mutils.get_score_fn_from_model(
                score_model=score_model,
                flow=flow,
                flow_name=flow_name,
                use_zt=use_zt,
                conditional=conditional,
                device=config.device,
                sde=sde,
                epsilons=config.training.epsilons,
                prob_path=prob_path,
            )
            samples, nfe = sampling_fn(score_fn)
            print(f"Sampling took {nfe} steps")
            samples = samples.detach().cpu()
            save_image(
                samples[:64, :, :, :], os.path.join(eval_dir, "diffusion_samples.png")
            )
            results_dict = {"samples": samples.numpy(), "nfe": nfe}
            np.savez(os.path.join(eval_dir, "diffusion_sampling"), **results_dict)

        # Compute the loss function on the full evaluation dataset if loss computation is enabled
        if config.eval.enable_loss:
            all_losses = []
            for i, (eval_batch, _) in enumerate(eval_ds):
                eval_batch = (
                    (eval_batch * 255.0) + torch.rand_like(eval_batch)
                ) / 256.0

                eval_batch = scaler(eval_batch)
                eval_batch = eval_batch.to(config.device)
                eval_loss = eval_step(state, eval_batch)
                all_losses.append(eval_loss["loss"])
                if (i + 1) % 1000 == 0:
                    logging.info("Finished %dth step loss evaluation" % (i + 1))

            # Save loss values to disk or Google Cloud Storage
            all_losses = np.asarray(all_losses)
            print(f"mean loss : {np.mean(all_losses)}")
            # raise Exception
            with open(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(
                    io_buffer, all_losses=all_losses, mean_loss=all_losses.mean()
                )
                fout.write(io_buffer.getvalue())

        # Compute log-likelihoods (bits/dim) if enabled
        print("starting density ratio estimation")
        if config.eval.enable_bpd:
            all_bpds = []
            total_bpd = 0
            total_n_data = 0
            for repeat in range(bpd_num_repeats):
                # TODO: if we're running AIS, we'll just compute the normalizing constant once and reuse it for every batch
                # you don't actually need to feed in eval_batch, but just for convenience of API
                if config.eval.ais:
                    # NUTS takes forever to run
                    # if config.eval.mcmc_algo == "nuts":
                    #     from ais.nuts_ais import ais_fn
                    # elif config.eval.mcmc_algo == "hmc":
                    #     from ais.hmc_ais import ais_fn
                    from ais.diffusion_hmc_ais import ais_fn

                    # Let's try these settings
                    # ais_batch_size = config.eval.batch_size
                    # ais_dataloader = eval_ds
                    ais_batch_size = config.eval.ais_batch_size
                    ais_dataloader = datasets.get_ais_test_set_for_flow(config)
                    #
                    n_ais_samples = config.eval.n_ais_samples
                    n_ais_steps = config.eval.n_ais_steps
                    n_steps_per_ais_step = config.eval.n_steps_per_ais_step
                    n_continue = config.eval.n_continue
                    ais_method = config.eval.ais_method
                    num_hmc_steps = config.eval.n_hmc_steps
                    initial_step_size = config.eval.initial_step_size
                    ais_x, ais_z, init_z, logws, log_normalizer, acceptance_rate = (
                        ais_fn(
                            flow=flow,
                            flow_name=flow_name,
                            score_model=score_model,
                            use_zt=use_zt,
                            conditional=conditional,
                            batch_size=ais_batch_size,
                            dataloader=ais_dataloader,
                            num_ais_samples=n_ais_samples,
                            num_ais_steps=n_ais_steps,
                            num_steps_per_ais_step=n_steps_per_ais_step,
                            num_continue=n_continue,
                            ais_method=ais_method,
                            num_hmc_steps=num_hmc_steps,
                            scaler=scaler,
                            inverse_scaler=inverse_scaler,
                            initial_step_size=initial_step_size,
                            device=config.device,
                            sde=sde,
                            epsilons=epsilons,
                            prob_path=prob_path,
                            rtol=config.eval.ais_rtol,
                            atol=config.eval.ais_atol,
                        )
                    )
                    ais_x = ais_x.view(-1, 1, 28, 28)
                    ais_z = ais_z.view(-1, 1, 28, 28)
                    log_normalizer = log_normalizer.detach().cpu().numpy()
                    print(f"estimated log normalizer: {log_normalizer}")
                    ais_dict = {
                        "x": ais_x.detach().cpu().numpy(),
                        "z": ais_z.detach().cpu().numpy(),
                        "init_z": init_z.detach().cpu().numpy(),
                        "logws": logws.detach().cpu().numpy(),
                        "log_normalizer": log_normalizer,
                        "acceptance_rate": acceptance_rate,
                    }
                    save_image(
                        ais_x.detach().cpu()[:64, :, :, :],
                        os.path.join(
                            eval_dir,
                            "ais_samples_{}chains_{}steps.png".format(
                                n_ais_samples, n_ais_steps
                            ),
                        ),
                    )
                    np.savez(
                        os.path.join(
                            eval_dir, "ais_x_{}_ckpt_{}_output".format(ais_method, ckpt)
                        ),
                        **ais_dict,
                    )
                    print(
                        "finished running {} for estimating log partition function!".format(
                            ais_method
                        )
                    )
                    # print("exiting program...")
                    # import sys

                    # sys.exit(0)
                # TODO: need a better system of running AIS and doing eval

                if not config.eval.ais:
                    nfes = []

                for batch_id, (eval_batch, _) in enumerate(eval_ds):
                    eval_batch = (
                        (eval_batch * 255.0) + torch.rand_like(eval_batch)
                    ) / 256.0

                    # you can do this because eval_batch is from the test set, and already has been uniformly dequantized
                    eval_batch = scaler(eval_batch)
                    eval_batch = eval_batch.to(config.device)
                    if not config.eval.ais:
                        # add nfe records
                        bpd, _, nfe = density_ratio_fn(
                            score_model=score_model, x=eval_batch
                        )
                        nfes.append(nfe)
                    else:
                        # if AIS, save outputs as well for safety
                        bpd = density_ratio_fn(
                            score_model=score_model,
                            x=eval_batch,
                            log_normalizer=log_normalizer,
                        )[0]
                    # NOTE: we've converted bpds from a list to average bpd per batch
                    total_bpd += bpd.item() * eval_batch.shape[0]
                    total_n_data += eval_batch.shape[0]
                    logging.info(
                        "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f"
                        % (ckpt, repeat, batch_id, total_bpd / total_n_data)
                    )
                    bpd_round_id = batch_id + len(ds_bpd) * repeat
                    # Save bits/dim to disk or Google Cloud Storage
                    # TODO: add AIS evaluation
                    fname = f"vanilla_{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"
                    with open(os.path.join(eval_dir, fname), "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, total_bpd, total_n_data)
                        fout.write(io_buffer.getvalue())
                avg_bpd = float(total_bpd) / total_n_data
                print(
                    "Completed bpd evaluation, total average bpd is: {}".format(avg_bpd)
                )
                all_bpds.append(avg_bpd)

                if not config.eval.ais:
                    print(
                        "Total average number of function evaluations is: {}".format(
                            np.mean(nfes)
                        )
                    )
                    with open(os.path.join(eval_dir, "nfes.p"), "wb") as fp:
                        pickle.dump(nfes, fp)

            with open(os.path.join(eval_dir, "all_bpds.p"), "wb") as f:
                pickle.dump(all_bpds, f)
