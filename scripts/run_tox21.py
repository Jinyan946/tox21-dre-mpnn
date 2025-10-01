# scripts/run_tox21.py
# ------------------------------------------------------------
# Default: trains the DRE head (RatioMLP) on cached features.
# New:     --ctsm trains a time-conditioned head along a VP path
#          (keeps the paper's core idea in embedding space).
# ------------------------------------------------------------

import os, json, math, argparse, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- project modules (unchanged)
from tox21_dataset import TwoClassArrayDataset
from torch_only_lib import RatioMLP, logistic_nce_loss

# --- optional Chemprop for MPNN embeddings
CHEMPROP_OK = True
try:
    from chemprop import data as cp_data, featurizers as cp_feat, models as cp_models, nn as cp_nn
except Exception:
    CHEMPROP_OK = False

# --- RDKit for Morgan fallback
from rdkit import Chem
from rdkit.Chem import AllChem


# =========================
# utils
# =========================
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


# =========================
# MPNN embeddings (robust)
# =========================
@torch.no_grad()
def build_mpnn_embeddings_from_csv(csv_path: str, batch_size: int, chemprop_ckpt: str | None) -> np.ndarray:
    """
    Build Chemprop MPNN embeddings from a CSV with columns smiles,label.
    Compatible across common Chemprop v2 variants. Keeps encoder on CPU.
    """
    if not CHEMPROP_OK:
        raise RuntimeError("Chemprop not available; cannot build MPNN embeddings.")

    smiles, _ = read_split_csv(csv_path)

    # data / featurizer / dataset / loader
    dps = [cp_data.MoleculeDatapoint.from_smi(s) for s in smiles]
    featurizer = cp_feat.SimpleMoleculeMolGraphFeaturizer()
    dset = cp_data.MoleculeDataset(dps, featurizer=featurizer)
    loader = cp_data.build_dataloader(dset, shuffle=False, batch_size=batch_size, num_workers=0)

    # encoder
    if chemprop_ckpt and os.path.isfile(chemprop_ckpt):
        mpnn = cp_models.MPNN.load_from_checkpoint(chemprop_ckpt)
    else:
        mp = cp_nn.BondMessagePassing()
        agg = cp_nn.MeanAggregation()
        ffn = cp_nn.RegressionFFN()

        mpnn = None
        ctor_errors = []
        for ctor in (
            lambda: cp_models.MPNN(mp, agg, ffn, True, []),                   # 5 positional
            lambda: cp_models.MPNN(mp, agg, ffn, True),                       # 4 positional
            lambda: cp_models.MPNN(mp, agg, ffn),                             # 3 positional
            lambda: cp_models.MPNN(mp=mp, agg=agg, ffn=ffn, batch_norm=True), # kwargs
        ):
            try:
                mpnn = ctor(); break
            except Exception as e:
                ctor_errors.append(repr(e))
        if mpnn is None:
            raise RuntimeError("Could not construct chemprop.models.MPNN:\n" + "\n".join(ctor_errors))

    mpnn = mpnn.cpu()
    mpnn.eval()

    # forward → embeddings
    chunks = []
    for batch in loader:
        try:
            z = mpnn.encoding(batch.bmg, i=0)  # preferred signature
        except Exception:
            try:
                z = mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=0)
            except Exception:
                z = mpnn.encoding(batch, i=0)
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        chunks.append(z.astype(np.float32))

    X = np.concatenate(chunks, axis=0)
    return X


# =========================
# Morgan fallback
# =========================
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


# =========================
# feature cache (.npz)
# =========================
def ensure_npz_cache(data_dir: str, cache_dir: str, batch: int, chemprop_ckpt: str | None, prefer_mpnn: bool):
    """
    Ensure {train,valid,test}.npz exist in cache_dir.
    If prefer_mpnn=True, try Chemprop embeddings first; else Morgan.
    """
    os.makedirs(cache_dir, exist_ok=True)
    csvs = {k: os.path.join(data_dir, f"{k}.csv") for k in ["train", "valid", "test"]}
    for k, p in csvs.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing {k}.csv at {p}")

    npzs = {k: os.path.join(cache_dir, f"{k}.npz") for k in ["train", "valid", "test"]}
    if all(os.path.isfile(p) for p in npzs.values()):
        return npzs  # already there

    use_mpnn = prefer_mpnn and CHEMPROP_OK

    def write_npz(split):
        csv_path = csvs[split]
        _, y = read_split_csv(csv_path)
        if use_mpnn:
            X = build_mpnn_embeddings_from_csv(csv_path, batch_size=batch, chemprop_ckpt=chemprop_ckpt)
        else:
            X = morgan_from_csv(csv_path, nbits=2048, radius=2)
        if X.shape[0] != len(y):
            raise RuntimeError(f"X/Y size mismatch on {split}: {X.shape[0]} vs {len(y)}")
        np.savez_compressed(npzs[split], X=X.astype(np.float32), y=np.asarray(y, dtype=np.float32))

    try:
        print(f"Building {'Chemprop MPNN' if use_mpnn else 'Morgan'} features → caching to .npz ...")
        for split in ["train", "valid", "test"]:
            print(f"  • {split}.csv → {split}.npz")
            write_npz(split)
        return npzs
    except Exception as e:
        if use_mpnn:
            warnings.warn(f"MPNN embedding failed ({e}). Falling back to Morgan fingerprints.")
            use_mpnn = False
            for split in ["train", "valid", "test"]:
                print(f"  • {split}.csv → {split}.npz (Morgan fallback)")
                write_npz(split)
            return npzs
        raise


# =========================
# eval helper (DRE)
# =========================
@torch.no_grad()
def eval_loss(model, dataset, batch, device, num_batches=50):
    model.eval()
    losses = []
    for _ in range(num_batches):
        xp = dataset.sample_p(batch).to(device).float()
        xq = dataset.sample_q(batch).to(device).float()
        losses.append(logistic_nce_loss(model, xp, xq).item())
    return float(np.mean(losses))


# =========================
# DRE training (default)
# =========================
def train_dre(args):
    device = get_device()
    print("Using device:", device)
    os.makedirs(args.workdir, exist_ok=True)
    save_config(args, args.workdir)
    set_seed(args.seed)

    feat_tag = "mpnn" if (args.prefer_mpnn and CHEMPROP_OK) else "morgan"
    cache_dir = os.path.join(args.data_dir, f"cache_{feat_tag}" + ("_ckpt" if args.chemprop_ckpt else ""))
    npzs = ensure_npz_cache(
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        batch=args.batch,
        chemprop_ckpt=args.chemprop_ckpt if args.chemprop_ckpt else None,
        prefer_mpnn=args.prefer_mpnn
    )

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

    test_loss = eval_loss(model, test_ds, batch=args.batch, device=device, num_batches=100)
    print(f"[test] loss {test_loss:.4f}")

    final_ckpt = os.path.join(args.workdir, "ckpt_final.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": args.steps}, final_ckpt)
    print(f"Saved final checkpoint to: {final_ckpt}")


# ===========================================================
# CTSM mode (paper-style time-conditioned estimator in embed space)
# ===========================================================
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), steps=half, device=t.device))
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class TimeConditionedRatioMLP(nn.Module):
    """s_theta(x, t) -> scalar"""
    def __init__(self, input_dim: int, hidden: int = 256, depth: int = 2, t_dim: int = 64):
        super().__init__()
        self.t_dim = t_dim
        layers, d = [], input_dim + t_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32: x = x.float()
        if t.dtype != torch.float32: t = t.float()
        te = sinusoidal_time_embedding(t, self.t_dim)
        return self.net(torch.cat([x, te], dim=-1)).squeeze(-1)

def vp_alpha(t: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    # α(t) = exp(-0.5 * beta * t)
    return torch.exp(-0.5 * beta * t)

def ctsm_gaussian_time_score(x: torch.Tensor, z: torch.Tensor, t: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Closed-form ∂_t log N(x ; mean=α(t) z, cov=(1-α(t)^2) I).
    x, z: [B, D]; t: [B] in [0,1].
    """
    a = vp_alpha(t, beta=beta)        # [B]
    a2 = a * a                        # [B]
    v = 1.0 - a2                      # [B]
    a_prime = -0.5 * beta * a         # [B]
    v_prime = beta * a2               # [B]

    r2 = (x - a[:, None] * z).pow(2).sum(dim=1)   # [B]
    x_dot_z = (x * z).sum(dim=1)                  # [B]
    z2 = (z * z).sum(dim=1)                       # [B]
    d = x.shape[1]

    term1 = -0.5 * d * (v_prime / v)
    term2 = 0.5 * v_prime * (r2 / (v * v))
    term3 = (a_prime / v) * (x_dot_z - a * z2)
    return term1 + term2 + term3

@torch.no_grad()
def sample_x_t(z: torch.Tensor, t: torch.Tensor, beta: float) -> torch.Tensor:
    a = vp_alpha(t, beta)
    v = 1.0 - a * a
    eps = torch.randn_like(z)
    return a[:, None] * z + torch.sqrt(v)[:, None] * eps

def train_ctsm(args):
    device = get_device()
    print("CTSM mode — device:", device)
    os.makedirs(args.workdir, exist_ok=True)
    save_config(args, args.workdir)
    set_seed(args.seed)

    # Reuse cache builder to get embeddings (MPNN preferred)
    feat_tag = "mpnn" if (args.prefer_mpnn and CHEMPROP_OK) else "morgan"
    cache_dir = os.path.join(args.data_dir, f"cache_{feat_tag}" + ("_ckpt" if args.chemprop_ckpt else ""))
    npzs = ensure_npz_cache(
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        batch=args.batch,
        chemprop_ckpt=args.chemprop_ckpt if args.chemprop_ckpt else None,
        prefer_mpnn=args.prefer_mpnn
    )

    # Use TRAIN embeddings to define p1 in embedding space
    train_npz = np.load(npzs["train"])
    Z = torch.from_numpy(train_npz["X"]).float().to(device)  # [N, D]
    N, D = Z.shape
    print(f"[CTSM] embedding dim D={D}, N={N}")

    model = TimeConditionedRatioMLP(input_dim=D, hidden=args.hidden, depth=args.depth, t_dim=args.time_embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for step in range(1, args.ctsm_steps + 1):
        idx = torch.randint(0, N, (args.ctsm_batch,), device=device)
        z = Z.index_select(0, idx)                      # [B, D]
        t = torch.rand(args.ctsm_batch, device=device)  # [B] ~ U(0,1)
        x_t = sample_x_t(z, t, beta=args.ctsm_beta)     # [B, D]
        target = ctsm_gaussian_time_score(x_t, z, t, beta=args.ctsm_beta)  # [B]

        pred = model(x_t, t)                            # [B]
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and math.isfinite(args.grad_clip):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"[ctsm/train] step {step:6d} | mse {loss.item():.5f}")
        if step % args.ckpt_every == 0:
            ckpt = os.path.join(args.workdir, f"ctsm_{step}.pt")
            torch.save({"model": model.state_dict(), "step": step, "dim": D}, ckpt)
            print("Saved:", ckpt)

    final_ckpt = os.path.join(args.workdir, "ctsm_final.pt")
    torch.save({"model": model.state_dict(), "step": args.ctsm_steps, "dim": D}, final_ckpt)
    print("Saved CTSM model to:", final_ckpt)


# =========================
# main / CLI
# =========================
def build_argparser():
    ap = argparse.ArgumentParser(description="Train on Tox21: DRE (default) or CTSM (time-conditioned).")
    # I/O
    ap.add_argument("--data_dir", type=str, required=True, help="Dir with train.csv, valid.csv, test.csv")
    ap.add_argument("--workdir", type=str, required=True, help="Where to save checkpoints/results")

    # DRE training config
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)

    # RatioMLP sizes
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)

    # feature control
    ap.add_argument("--prefer_mpnn", action="store_true", help="Try Chemprop MPNN embeddings first; fallback to Morgan.")
    ap.add_argument("--chemprop_ckpt", type=str, default="", help="Path to a Chemprop checkpoint (optional).")

    # CTSM switches
    ap.add_argument("--ctsm", action="store_true", help="Train time-conditioned CTSM in embedding space.")
    ap.add_argument("--ctsm_beta", type=float, default=1.0, help="Beta for VP path.")
    ap.add_argument("--time_embed_dim", type=int, default=64, help="Time embedding dim for CTSM head.")
    ap.add_argument("--ctsm_steps", type=int, default=20000, help="Training steps for CTSM mode.")
    ap.add_argument("--ctsm_batch", type=int, default=512, help="Batch size for CTSM mode.")
    ap.add_argument("--integrate_K", type=int, default=64, help="# time points for integral at eval (if you add eval).")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()
    if args.ctsm:
        train_ctsm(args)
    else:
        train_dre(args)

if __name__ == "__main__":
    main()
