from chemprop import data, featurizers, models
import numpy as np
import pandas as pd
import torch

def load_split(csv_path):
    df = pd.read_csv(csv_path)
    smiles = df["smiles"].tolist()
    y = df["label"].values.astype(float)  # shape [N] or [N, T]
    return smiles, y

for split in ["train","valid","test"]:
    smiles, y = load_split(f"data/tox21_NR-AR/{split}.csv")


    dps = [data.MoleculeDatapoint.from_smi(s) for s in smiles]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(dps, featurizer=featurizer)
    loader = data.build_dataloader(dset, shuffle=False, batch_size=256, num_workers=0)

    
    mpnn = models.MPNN(hidden_size=256, depth=2) 
    mpnn.eval()

  
    embs = []
    with torch.no_grad():
        for batch in loader:
            z = mpnn.encoding(batch) 
            embs.append(z.cpu().numpy())

    X = np.concatenate(embs, axis=0)   # shape [N, D], D=hidden_size or readout dim
    np.savez_compressed(f"data/tox21_NR-AR_mpnn/{split}.npz", X=X, y=y)

