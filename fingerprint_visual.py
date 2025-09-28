# viz_fingerprints.py
# Visualize cached features (MPNN or Morgan) with PCA → 2D scatter.
# Uses only numpy + matplotlib (no seaborn), saves a PNG + CSV.

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_npz(base="data"):
    # Prefer MPNN caches, then Morgan caches, then any train.npz
    for pattern in ["**/cache_mpnn*/train.npz", "**/cache_morgan*/train.npz", "**/train.npz"]:
        hits = glob.glob(os.path.join(base, pattern), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    raise FileNotFoundError("Could not find any train.npz under ./data/ — "
                            "make sure you ran prepare_tox21.py and run_tox21.py.")

def compute_pca_np(X, k=2):
    # PCA via SVD (no sklearn needed)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:k].T
    # explained variance ratio (approx)
    ev = (S**2) / (X.shape[0]-1)
    r = ev[:k] / ev.sum()
    return Z, r

def main():
    npz_path = find_npz("data")
    print("Using features:", npz_path)
    d = np.load(npz_path)
    X, y = d["X"], d["y"]
    print("X shape:", X.shape, "dtype:", X.dtype, "| y shape:", y.shape)

    # try to load SMILES (optional, for the CSV)
    cache_dir = os.path.dirname(npz_path)
    assay_dir = os.path.dirname(cache_dir)
    smiles = None
    train_csv = os.path.join(assay_dir, "train.csv")
    if os.path.exists(train_csv):
        df_csv = pd.read_csv(train_csv)
        if "smiles" in df_csv.columns and len(df_csv) >= len(y):
            smiles = df_csv["smiles"].astype(str).values[:len(y)]

    # PCA to 2D
    Z, ratio = compute_pca_np(X, k=2)

    # Plot (single scatter, default colors)
    plt.figure(figsize=(8, 8))
    plt.title("Fingerprints / Embeddings (PCA)")
    plt.xlabel(f"PCA1 ({ratio[0]*100:.1f}% var)")
    plt.ylabel(f"PCA2 ({ratio[1]*100:.1f}% var)")
    plt.scatter(Z[:, 0], Z[:, 1], s=6)   # no explicit colors/styles
    plt.tight_layout()

    # Save outputs
    out_png = "fingerprints_pca.png"
    out_csv = "fingerprints_pca_coords.csv"
    plt.savefig(out_png, dpi=150)
    print("Saved plot →", out_png)

    out_df = pd.DataFrame({"pca1": Z[:, 0], "pca2": Z[:, 1], "label": y})
    if smiles is not None:
        out_df.insert(0, "smiles", smiles)
    out_df.to_csv(out_csv, index=False)
    print("Saved coordinates →", out_csv)

    # Show the plot window (optional)
    plt.show()

if __name__ == "__main__":
    main()
