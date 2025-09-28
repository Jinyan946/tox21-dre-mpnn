# eval_tox21.py
import os, glob, argparse, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_curve
from scipy.special import expit  # stable sigmoid
import torch_only_lib as run_lib  # torch-only helper

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def batched_logits(model, X, device, batch=2048):
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i+batch]).to(device)
        lb = run_lib.estimate_log_ratio(model, xb)  # logits ~ log r(x) + const
        outs.append(lb.detach().cpu())
    return torch.cat(outs).numpy()

def choose_threshold_from_valid(valid_logits, y_valid, rule="youden"):
    """
    Returns:
      thr_logit: float (threshold in logit space)
      info: dict with selection stats
      offset: float (log(π1/(1-π1)) for prior correction)
    """
    pi1 = float(y_valid.mean())
    pi1 = min(max(pi1, 1e-6), 1 - 1e-6)   # guard
    offset = np.log(pi1 / (1 - pi1))

    if rule == "youden":
        fpr, tpr, thr = roc_curve(y_valid, valid_logits)
        j = tpr - fpr
        i = int(j.argmax())
        return float(thr[i]), {"rule": "youden", "valid_tpr": float(tpr[i]), "valid_fpr": float(fpr[i])}, offset

    # rule == "max_f1": search prob threshold on prior-corrected probabilities
    p = expit(valid_logits + offset)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_valid, p >= t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    thr_logit = float(np.log(best_t / (1 - best_t)) - offset)  # back to logit
    return thr_logit, {"rule": "max_f1", "valid_f1": float(best_f1), "valid_thr_prob": float(best_t)}, offset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/tox21_NR-AR_fp2048")
    ap.add_argument("--ckpt", default="", help="explicit checkpoint path")
    ap.add_argument("--ckpt_dir", default="../results/tox21_NR-AR", help="dir with ckpt_*.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--rule", choices=["youden", "max_f1"], default="youden")
    args = ap.parse_args()

    # resolve checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        cands = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_*.pt")))
        if not cands:
            raise FileNotFoundError(f"No ckpt_*.pt found in {args.ckpt_dir}")
        ckpt_path = cands[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    # load data
    valid = np.load(os.path.join(args.data_dir, "valid.npz"))
    test  = np.load(os.path.join(args.data_dir, "test.npz"))
    Xv, yv = valid["X"], valid["y"]
    Xt, yt = test["X"],  test["y"]

    # model
    device = get_device()
    model = run_lib.build_model(input_dim=Xt.shape[1], hidden=args.hidden, depth=args.depth).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # logits
    logits_v = batched_logits(model, Xv, device)
    logits_t = batched_logits(model, Xt, device)

    # choose threshold on VALID (logit space), compute both prob views
    thr_logit, info, offset = choose_threshold_from_valid(logits_v, yv, rule=args.rule)
    thr_prob_raw = float(expit(thr_logit))              # WITHOUT prior correction
    thr_prob_cal = float(expit(thr_logit + offset))     # WITH  prior correction (using VALID π1)

    print(f"Chosen threshold (rule={info.get('rule','')})")
    print(f"  • logit: {thr_logit:.6f}")
    print(f"  • prob_raw (sigmoid(logit)): {thr_prob_raw:.6f}")
    print(f"  • prob_cal (sigmoid(logit + log(pi1/(1-pi1)))): {thr_prob_cal:.6f}")
    print(f"  • info: {info}")

    # predictions on TEST using LOGIT threshold
    yhat = (logits_t >= thr_logit).astype(int)

    # AUCs (threshold-free) on TEST using logits
    roc = roc_auc_score(yt, logits_t)
    pr  = average_precision_score(yt, logits_t)

    # metrics at chosen threshold
    prec, rec, f1, _ = precision_recall_fscore_support(yt, yhat, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yt, yhat).ravel()

    # optional: calibrated prob summaries (using TEST logits + VALID prior)
    p_test_raw = expit(logits_t)             # uncorrected
    p_test_cal = expit(logits_t + offset)    # prior-corrected
    pos_mask = (yt == 1); neg_mask = (yt == 0)
    cal_pos_mean = float(p_test_cal[pos_mask].mean()) if pos_mask.any() else float("nan")
    cal_neg_mean = float(p_test_cal[neg_mask].mean()) if neg_mask.any() else float("nan")
    raw_pos_mean = float(p_test_raw[pos_mask].mean()) if pos_mask.any() else float("nan")
    raw_neg_mean = float(p_test_raw[neg_mask].mean()) if neg_mask.any() else float("nan")

    print(f"Test ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
    print(f"Test @logit_thr={thr_logit:.4f} | P: {prec:.4f} R: {rec:.4f} F1: {f1:.4f}")
    print(f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Prob means (TEST) → raw: pos {raw_pos_mean:.4f}, neg {raw_neg_mean:.4f} | calibrated: pos {cal_pos_mean:.4f}, neg {cal_neg_mean:.4f}")
