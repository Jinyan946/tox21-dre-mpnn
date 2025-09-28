# mpnn_embed_tox21.py  (new script)
from chemprop import data, featurizers, models
import numpy as np
import pandas as pd
import torch

# 1) Load your splits (CSV with at least: smiles, label)
def load_split(csv_path):
    df = pd.read_csv(csv_path)
    smiles = df["smiles"].tolist()
    y = df["label"].values.astype(float)  # shape [N] or [N, T]
    return smiles, y

for split in ["train","valid","test"]:
    smiles, y = load_split(f"data/tox21_NR-AR/{split}.csv")

    # 2) Build Chemprop datapoints & dataset
    dps = [data.MoleculeDatapoint.from_smi(s) for s in smiles]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(dps, featurizer=featurizer)
    loader = data.build_dataloader(dset, shuffle=False, batch_size=256, num_workers=0)

    # 3) Create an encoder MPNN (no task heads). Match hidden/depth to mentor’s example.
    mpnn = models.MPNN(hidden_size=256, depth=2)  # adjust if they gave other settings
    mpnn.eval()

    # 4) Run forward to capture latent encodings
    embs = []
    with torch.no_grad():
        for batch in loader:
            # Most chemprop examples expose a way to get the latent graph embedding.
            # If your file uses mpnn.encoding(batch): use that; otherwise mpnn.forward(batch, return_latent=True)
            z = mpnn.encoding(batch)  # <-- if your provided file uses a different method, call that
            embs.append(z.cpu().numpy())

    X = np.concatenate(embs, axis=0)   # shape [N, D], D=hidden_size or readout dim
    np.savez_compressed(f"data/tox21_NR-AR_mpnn/{split}.npz", X=X, y=y)

