import torch
import os
import logging
import numpy as np
from prob_path_lib import OneVP, TwoSB, OneRQNSFVP


def restore_checkpoint(ckpt_dir, state, device, test=False):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input"
        )
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        # TODO: do we need this?
        if not test:
            state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=True)
        print("Loaded model")
        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]
        try:
            state["scheduler"] = loaded_state["scheduler"]
        except:
            pass
        return state


def load_history(file_path, history, interpolate=False):
    record = np.load(os.path.join(file_path, "history.npz"))
    history._loss_history = record["loss_history"]

    # for previously trained models, these two things may not have been saved
    try:
        history._time_history = record["time_history"]
    except:
        print("time history had not been saved, skipping...")

    try:
        history._loss_counts = record["loss_counts"]
    except:
        print("loss counts have not been saved, automatically warming up")
        history._loss_counts = (
            np.ones([history.batch_size], dtype=np.int) * history.history_per_term
        )

    # only interpolation has a saved weight history
    assert history._warmed_up()
    if interpolate:
        history._weight_history = record["weight_history"]
        assert history._initialized_weights()
        print("history weights have been initialized from prior run!")

    return history


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
        # 'scheduler': state['scheduler']
    }
    torch.save(saved_state, ckpt_dir)


def get_prob_path(dim, prob_path, config):
    if prob_path is not None:
        if prob_path == "OneVP":
            return OneVP(dim)
        elif prob_path == "TwoSB":
            return TwoSB(dim, var=config.training.two_sb_var)
        elif prob_path == "OneRQNSFVP":
            return OneRQNSFVP(
                dim, beta_min=config.model.beta_min, beta_max=config.model.beta_max
            )
        # elif prob_path == "TwoOT":
        #     return TwoOT(dim)
        # elif prob_path == "TwoRegSB":
        #     return TwoRegSB(dim)

        # elif prob_path == "OneRegVP":
        #     return OneRegVP(dim)
        # elif prob_path == "OneOT":
        #     return OneOT(dim)
        else:
            raise NotImplementedError
    else:
        return None
