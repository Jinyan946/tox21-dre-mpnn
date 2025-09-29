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

