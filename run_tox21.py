# run_tox21.py
import os
import json
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import warnings

from tox21_dataset import TwoClassArrayDataset
from torch_only_lib import RatioMLP, logistic_nce_loss

# ---------- optional Chemprop (for MPNN embeddings) ----------
CHEMPROP_OK = True
try:
    from chemprop import data as cp_data, featurizers as cp_feat, models as cp_models, nn as cp_nn
except Exception:
    CHEMPROP_OK = False

# ---------- RDKit (for Morgan fallback) ----------
from rdkit import Chem
from rdkit.Chem import AllChem


# -------------------- utils --------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_config(args, workdir):
    with open(os.path.join(workdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def read_split_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns ['smiles','label'] in {csv_path}. Found: {list(df.columns)}")
    smiles = df["smiles"].astype(str).tolist()
    y = df["label"].astype(float).values
    return smiles, y


# ---------------- MPNN embedding builder (robust) ----------------
@torch.no_grad()
def build_mpnn_embeddings_from_csv(csv_path: str, batch_size: int, chemprop_ckpt: str | None) -> np.ndarray:
    """
    Build Chemprop MPNN graph embeddings from a CSV with columns smiles,label.
    Tries to be compatible across Chemprop v2 variants. Keeps encoder on CPU for stability.
    """
    if not CHEMPROP_OK:
        raise RuntimeError("Chemprop not available; cannot build MPNN embeddings.")

    smiles, _ = read_split_csv(csv_path)

    # 1) datapoints / featurizer / dataset / loader
    dps = [cp_data.MoleculeDatapoint.from_smi(s) for s in smiles]
    featurizer = cp_feat.SimpleMoleculeMolGraphFeaturizer()
    dset = cp_data.MoleculeDataset(dps, featurizer=featurizer)
    loader = cp_data.build_dataloader(dset, shuffle=False, batch_size=batch_size, num_workers=0)

    # 2) build/load encoder (CPU)
    if chemprop_ckpt and os.path.isfile(chemprop_ckpt):
        mpnn = cp_models.MPNN.load_from_checkpoint(chemprop_ckpt)
    else:
        # Compose from building blocks; ctor signature varies across builds
        mp = cp_nn.BondMessagePassing()
        agg = cp_nn.MeanAggregation()
        ffn = cp_nn.RegressionFFN()
        mpnn = None
        ctor_errors = []
        for ctor in (
            lambda: cp_models.MPNN(mp, agg, ffn, True, []),                   # 5 positional
            lambda: cp_models.MPNN(mp, agg, ffn, True),                       # 4 positional
            lambda: cp_models.MPNN(mp, agg, ffn),                             # 3 positional
            lambda: cp_models.MPNN(mp=mp, agg=agg, ffn=ffn, batch_norm=True), # kwargs (no metric_list)
        ):
            try:
                mpnn = ctor(); break
            except Exception as e:
                ctor_errors.append(repr(e))
        if mpnn is None:
            raise RuntimeError(
                "Could not construct chemprop.models.MPNN. Tried several signatures:\n" +
                "\n".join(ctor_errors)
            )

    mpnn = mpnn.cpu()
    mpnn.eval()

    # 3) forward → embeddings
    chunks = []
    for batch in loader:
        # Try common enc signatures in order
        try:
            z = mpnn.encoding(batch.bmg, i=0)  # preferred (graph only)
        except Exception:
            try:
                z = mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=0)  # graph + descriptors
            except Exception:
                z = mpnn.encoding(batch, i=0)  # whole training batch
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        chunks.append(z.astype(np.float32))
    X = np.concatenate(chunks, axis=0)
    return X


# ---------------- Morgan fallback (fast, local) ----------------
def morgan_from_csv(csv_path: str, nbits=2048, radius=2) -> np.ndarray:
    smiles, _ = read_split_csv(csv_path)
    X = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            X.append(np.zeros(nbits, dtype=np.float32))
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(bv, arr)
        X.append(arr.astype(np.float32))
    return np.vstack(X)


# ---------------- ensure feature cache (.npz) ----------------
def ensure_npz_cache(data_dir: str, cache_dir: str, batch: int, chemprop_ckpt: str | None, prefer_mpnn: bool):
    """
    Ensures {train,valid,test}.npz exist in cache_dir.
    If prefer_mpnn=True, tries Chemprop embeddings first; if that fails, falls back to Morgan.
    """
    os.makedirs(cache_dir, exist_ok=True)
    csvs = {k: os.path.join(data_dir, f"{k}.csv") for k in ["train", "valid", "test"]}
    for k, p in csvs.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing {k}.csv at {p}")

    npzs = {k: os.path.join(cache_dir, f"{k}.npz") for k in ["train", "valid", "test"]}
    if all(os.path.isfile(p) for p in npzs.values()):
        return npzs  # already cached

    # choose path
    use_mpnn = prefer_mpnn and CHEMPROP_OK
    path_tried = "mpnn" if use_mpnn else "morgan"

    def write_npz(split):
        csv_path = csvs[split]
        smiles, y = read_split_csv(csv_path)
        if use_mpnn:
            X = build_mpnn_embeddings_from_csv(csv_path, batch_size=batch, chemprop_ckpt=chemprop_ckpt)
        else:
            X = morgan_from_csv(csv_path, nbits=2048, radius=2)
        if X.shape[0] != len(y):
            raise RuntimeError(f"X/Y size mismatch on {split}: {X.shape[0]} vs {len(y)}")
        np.savez_compressed(npzs[split], X=X.astype(np.float32), y=y.astype(np.float32))

    try:
        print(f"Building {'Chemprop MPNN' if use_mpnn else 'Morgan'} features → caching to .npz ...")
        for split in ["train", "valid", "test"]:
            print(f"  • {split}.csv → {split}.npz")
            write_npz(split)
        return npzs
    except Exception as e:
        if use_mpnn:
            warnings.warn(f"MPNN embedding failed ({e}). Falling back to Morgan fingerprints.")
            # fall back to Morgan
            use_mpnn = False
            for split in ["train", "valid", "test"]:
                print(f"  • {split}.csv → {split}.npz (Morgan fallback)")
                write_npz(split)
            return npzs
        raise


# ---------------- eval helper ----------------
@torch.no_grad()
def eval_loss(model, dataset, batch, device, num_batches=50):
    model.eval()
    losses = []
    for _ in range(num_batches):
        xp = dataset.sample_p(batch).to(device).float()
        xq = dataset.sample_q(batch).to(device).float()
        losses.append(logistic_nce_loss(model, xp, xq).item())
    return float(np.mean(losses))


# ---------------- training ----------------
def train(args):
    device = get_device()
    print("Using device:", device)
    os.makedirs(args.workdir, exist_ok=True)
    save_config(args, args.workdir)
    set_seed(args.seed)

    # decide cache name by feature type
    feat_tag = "mpnn" if (args.prefer_mpnn and CHEMPROP_OK) else "morgan"
    cache_dir = os.path.join(args.data_dir, f"cache_{feat_tag}" + ("_ckpt" if args.chemprop_ckpt else ""))
    npzs = ensure_npz_cache(
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        batch=args.batch,
        chemprop_ckpt=args.chemprop_ckpt if args.chemprop_ckpt else None,
        prefer_mpnn=args.prefer_mpnn
    )

    # datasets
    train_ds = TwoClassArrayDataset(npzs["train"])
    valid_ds = TwoClassArrayDataset(npzs["valid"])
    test_ds  = TwoClassArrayDataset(npzs["test"])

    input_dim = train_ds.dim
    print(f"Detected input dimension: {input_dim}  (feature type: {'MPNN' if 'mpnn' in cache_dir else 'Morgan'})")

    model = RatioMLP(input_dim=input_dim, hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ema_valid = None
    for step in range(1, args.steps + 1):
        model.train()
        xp = train_ds.sample_p(args.batch).to(device).float()
        xq = train_ds.sample_q(args.batch).to(device).float()
        loss = logistic_nce_loss(model, xp, xq)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and math.isfinite(args.grad_clip):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"[train] step {step:6d} | loss {loss.item():.4f}")

        if step % args.eval_every == 0:
            vloss = eval_loss(model, valid_ds, batch=args.batch, device=device, num_batches=50)
            ema_valid = vloss if ema_valid is None else 0.5 * ema_valid + 0.5 * vloss
            print(f"[valid] step {step:6d} | loss {vloss:.4f} | ema {ema_valid:.4f}")

        if step % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.workdir, f"ckpt_{step}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

    # Final eval
    test_loss = eval_loss(model, test_ds, batch=args.batch, device=device, num_batches=100)
    print(f"[test] loss {test_loss:.4f}")

    final_ckpt = os.path.join(args.workdir, "ckpt_final.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": args.steps}, final_ckpt)
    print(f"Saved final checkpoint to: {final_ckpt}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Train ratio model on Tox21 (prefers Chemprop MPNN; falls back to Morgan).")
    # I/O
    ap.add_argument("--data_dir", type=str, required=True, help="Dir with train.csv, valid.csv, test.csv")
    ap.add_argument("--workdir", type=str, required=True, help="Where to save checkpoints/results")

    # training config
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)

    # model (RatioMLP) sizes
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)

    # feature control
    ap.add_argument("--prefer_mpnn", action="store_true",
                    help="Try Chemprop MPNN embeddings first; fallback to Morgan on failure.")
    ap.add_argument("--chemprop_ckpt", type=str, default="",
                    help="Path to a Chemprop checkpoint (improves MPNN embeddings).")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
