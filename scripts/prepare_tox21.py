# prepare_tox21.py
import os, json, argparse
import numpy as np
import pandas as pd

# RDKit for optional Morgan baseline files
from rdkit import Chem
from rdkit.Chem import AllChem

# -------------------- helpers --------------------
def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _standardize_split_df(df: pd.DataFrame, assay: str) -> pd.DataFrame:
    """Return exactly: smiles,label (binary 0/1). Handles common TDC/DC column names."""
    # find SMILES
    smiles_col = next((c for c in ["smiles","SMILES","Drug","drug","mol","Mol","smile","SMILE"]
                       if c in df.columns), None)
    if smiles_col is None:
        raise ValueError(f"Could not find SMILES column in: {list(df.columns)}")
    # find label
    label_col = None
    candidates = [assay, "label","Label","y","Y"]
    for c in candidates:
        if c in df.columns:
            label_col = c; break
    if label_col is None:
        # fallback: single non-smiles column
        non_smiles = [c for c in df.columns if c != smiles_col]
        if len(non_smiles) == 1:
            label_col = non_smiles[0]
        else:
            raise ValueError(f"Could not infer label column from: {list(df.columns)}")

    out = pd.DataFrame({
        "smiles": df[smiles_col].astype(str),
        "label": pd.to_numeric(df[label_col], errors="coerce")
    })
    out = out.dropna(subset=["smiles","label"]).copy()
    # ensure binary in {0,1}
    # Some sources use {-1,1} or floats; map >0 to 1, else 0
    out["label"] = (out["label"] > 0).astype(int)
    return out

def _morgan_fp(smiles: str, nbits: int, radius: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

def _save_npz_from_csv(csv_path: str, out_npz: str, nbits: int, radius: int):
    df = pd.read_csv(csv_path)
    assert {"smiles","label"} <= set(df.columns)
    X = np.vstack([_morgan_fp(s, nbits, radius) for s in df["smiles"].tolist()])
    y = df["label"].astype(np.float32).values
    np.savez_compressed(out_npz, X=X, y=y)

# -------------------- loaders (try in order) --------------------
def load_tox21_splits_from_tdc_single_pred(assay: str, seed: int, frac):
    """Old PyTDC API: tdc.single_pred.Tox21"""
    from tdc.single_pred import Tox21
    ds = Tox21(name=assay)
    return ds.get_split(method="scaffold", seed=seed, frac=frac)  # dict of dfs

def load_tox21_splits_from_tdc_admet_group(assay: str, _seed: int, _frac):
    from tdc.benchmark_group import admet_group
    from tdc import utils

    bg = admet_group(path="data/tdc_cache")

    # Find a matching benchmark name
    # Try canonical list first
    try:
        names = utils.retrieve_benchmark_names('ADMET_Group')
    except Exception:
        names = []
    candidates = {assay, f"Tox21_{assay}", f"Tox21-{assay}", f"Tox21 {assay}"}
    name = next((n for n in names if n in candidates), None) or next(iter(candidates))

    # Try different API styles seen across versions:
    # 1) Direct get_train_valid_test(name)
    if hasattr(bg, "get_train_valid_test"):
        try:
            return bg.get_train_valid_test(name)  # dict of dfs
        except Exception:
            pass

    # 2) Two-step: get(...) → object with split getters
    if hasattr(bg, "get"):
        bench = bg.get(name)
        # common in other groups: get_train_valid_split() / get_test()
        if hasattr(bench, "get_train_valid_split") and hasattr(bench, "get_test"):
            tr, va = bench.get_train_valid_split()
            te = bench.get_test()
            return {"train": tr, "valid": va, "test": te}

        # sometimes: get_data() returns (train, valid, test)
        if hasattr(bench, "get_data"):
            tr, va, te = bench.get_data()
            return {"train": tr, "valid": va, "test": te}

    raise RuntimeError("ADMET group present, but no compatible split API on this version.")


def load_tox21_splits_from_deepchem(assay: str, seed: int, _frac):
    """DeepChem MoleculeNet loader; uses scaffold splitter internally."""
    import deepchem as dc
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer="Raw", splitter="scaffold", seed=seed, reload=True)
    train, valid, test = datasets
    # DeepChem returns multi-task; pick column for requested assay
    if assay not in tasks:
        raise ValueError(f"Assay '{assay}' not in DeepChem tasks: {tasks}")
    tidx = list(tasks).index(assay)

    def ds_to_df(ds):
        # ds.ids are SMILES; ds.y shape [N, T]; take task index tidx
        return pd.DataFrame({"smiles": ds.ids, "label": ds.y[:, tidx].astype(float)})
    return {"train": ds_to_df(train), "valid": ds_to_df(valid), "test": ds_to_df(test)}

def load_any_tox21_splits(assay: str, seed: int, frac):
    """Try old TDC → new TDC ADMET → DeepChem. Return dict of dfs with 'smiles','label'."""
    # 1) TDC single_pred
    try:
        split = load_tox21_splits_from_tdc_single_pred(assay, seed, frac)
        return {k: _standardize_split_df(v, assay) for k, v in split.items()}
    except Exception as e1:
        print(f"[info] tdc.single_pred path failed: {e1}")
    # 2) TDC ADMET group
    try:
        split = load_tox21_splits_from_tdc_admet_group(assay, seed, frac)
        return {k: _standardize_split_df(v, assay) for k, v in split.items()}
    except Exception as e2:
        print(f"[info] tdc.admet_group path failed: {e2}")
    # 3) DeepChem
    try:
        split = load_tox21_splits_from_deepchem(assay, seed, frac)
        return {k: _standardize_split_df(v, assay) for k, v in split.items()}
    except Exception as e3:
        print(f"[info] deepchem path failed: {e3}")

    raise RuntimeError(
        "Could not load Tox21 via PyTDC (old/new) or DeepChem.\n"
        "Fix by either:\n"
        "  A) install a TDC version with Tox21 support (e.g., `pip install pytdc==0.3.7`), or\n"
        "  B) install DeepChem that supports load_tox21 (e.g., `pip install deepchem`), or\n"
        "  C) provide your own CSVs with --csv_train/--csv_valid/--csv_test."
    )

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Prepare Tox21 split and save CSVs (+ optional Morgan .npz).")
    ap.add_argument("--assay", type=str, required=True, help="One Tox21 assay (e.g., NR-AR, NR-AR-LBD, SR-ARE, ...)")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, nargs=3, default=[0.8,0.1,0.1], help="train/valid/test fractions")
    # optional direct CSV bypass (if you already have them)
    ap.add_argument("--csv_train", type=str, default="")
    ap.add_argument("--csv_valid", type=str, default="")
    ap.add_argument("--csv_test",  type=str, default="")
    # legacy Morgan outputs (optional)
    ap.add_argument("--make_morgan_npz", action="store_true")
    ap.add_argument("--fp_bits", type=int, default=2048)
    ap.add_argument("--fp_radius", type=int, default=2)
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    paths = {}
    if args.csv_train and args.csv_valid and args.csv_test:
        for key, p in [("train",args.csv_train),("valid",args.csv_valid),("test",args.csv_test)]:
            if not os.path.isfile(p): raise FileNotFoundError(p)
            # standardize & re-write into outdir
            df = pd.read_csv(p)
            df_std = _standardize_split_df(df, args.assay)
            op = os.path.join(args.outdir, f"{key}.csv")
            df_std.to_csv(op, index=False)
            paths[key] = op
            print(f"Saved {key}.csv → {op}  ({len(df_std)} rows)")
    else:
        # auto-load from TDC/DC
        splits = load_any_tox21_splits(args.assay, args.seed, args.frac)
        for key in ["train","valid","test"]:
            op = os.path.join(args.outdir, f"{key}.csv")
            splits[key].to_csv(op, index=False)
            paths[key] = op
            print(f"Saved {key}.csv → {op}  ({len(splits[key])} rows)")

    # Optional: also build Morgan baseline .npz files
    if args.make_morgan_npz:
        for key in ["train","valid","test"]:
            csv_p = paths[key]
            npz_p = os.path.join(args.outdir, f"{key}.npz")
            print(f"Building Morgan FPs for {key} → {npz_p} (bits={args.fp_bits}, radius={args.fp_radius})")
            _save_npz_from_csv(csv_p, npz_p, nbits=args.fp_bits, radius=args.fp_radius)

    # meta.json
    meta = {
        "assay": args.assay, "split": "scaffold", "frac": args.frac,
        "csvs": {k: os.path.basename(v) for k, v in paths.items()},
        "morgan_npz": bool(args.make_morgan_npz),
        "fp_bits": args.fp_bits if args.make_morgan_npz else None,
        "fp_radius": args.fp_radius if args.make_morgan_npz else None,
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote meta.json")

if __name__ == "__main__":
    main()
