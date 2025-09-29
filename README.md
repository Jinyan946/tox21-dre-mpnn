# tox21-dre-mpnn
**Density-Ratio Estimation (DRE) for small molecules with Chemprop MPNN “graph fingerprints” (cached) and a lightweight PyTorch head.**

> Replace 2048-bit Morgan bits with **Chemprop MPNN embeddings**, cache them to `.npz`, and train a **density-ratio** head (RatioMLP + logistic NCE).

---

## ✨ What’s in here
- **Custom dataset pipeline (TDC/DeepChem):** Scaffold split (**80/10/10**, seed=42) → `train/valid/test.csv` with `smiles,label`.
- **Graph fingerprints with Chemprop:** Build/c**ache MPNN embeddings** (`.npz`) from SMILES.
- **DRE head:** Small MLP (“RatioMLP”) trained with **logistic NCE** (BCE-with-logits) for density-ratio estimation \( \( \log \frac{p_A(x)}{p_B(x)} \) \).
- **Viz:** Quick **PCA plot** of fingerprints/embeddings for sanity checking.

---

## 📦 Environment
- Python **3.11**
- RDKit (conda-forge), PyTorch, Chemprop (v2 API from GitHub), DeepChem (fallback loader)

```bash
# Create/activate an environment
conda create -n chemprop311 python=3.11 -y
conda activate chemprop311

# Core deps
conda install -c conda-forge rdkit -y
pip install torch torchvision torchaudio

# Chemprop v2 
pip install "git+https://github.com/chemprop/chemprop.git"

# Fallback (only if TDC loader doesn't work on your machine)
pip install "deepchem>=2.8.0"

# project extras
pip install numpy pandas matplotlib scikit-learn
```

## 📚 Data preparation

Generates scaffold splits and standardized CSVs (smiles,label).

```bash
python scripts/prepare_tox21.py \
  --assay NR-AR \
  --outdir data/tox21_NR-AR_fp2048 \
  --make_morgan_npz
```

Outputs:

data/tox21_NR-AR_fp2048/
  train.csv  valid.csv  test.csv
  train.npz  valid.npz  test.npz         # optional Morgan baseline
  meta.json


## 🧠 Training (DRE head)

Builds and caches features automatically:

Trains the RatioMLP head on cached features (not end-to-end).
```bash
python scripts/run_tox21.py \
  --data_dir data/tox21_NR-AR_fp2048 \
  --workdir results/nr-ar_mpnn \
  --prefer_mpnn \
  --batch 256 --steps 20000 --lr 0.002 --hidden 256 --depth 2
```

Feature caches:

data/<assay>/cache_mpnn[/_ckpt]/{train,valid,test}.npz  # MPNN embeddings
data/<assay>/cache_morgan/{train,valid,test}.npz        # Morgan fallback


You’ll see:

Detected input dimension: <D>  (feature type: MPNN|Morgan)
[train] step ... | loss ...
[valid] step ... | loss ...

## 👀 Visualize fingerprints
```bash
python scripts/viz_fingerprints.py
```
Outputs:

fingerprints_pca.png — 2D scatter of PCA(Embeddings)

fingerprints_pca_coords.csv — coordinates (and labels/smiles if available)
