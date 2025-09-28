import pandas as pd
import torch
from pathlib import Path
from chemprop import data, featurizers, models

# paths
csv_path = "data/tox21_binary.csv"           # your SMILES+label CSV
smiles_col = "smiles"
label_col = "label"
checkpoint_path = Path("ckpt_tox21_nr_ar/fold_0/model_0/model.pt")  # <-- adjust to your file

# load chemprop MPNN
mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
mpnn.eval()

# build dataset/loader
df = pd.read_csv(csv_path)
smis = df[smiles_col].tolist()

datapoints = [data.MoleculeDatapoint.from_smi(s) for s in smis]
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
dset = data.MoleculeDataset(datapoints, featurizer=featurizer)
loader = data.build_dataloader(dset, shuffle=False)

# helper to run encoding for i=0 (fingerprints) or i=1 (encodings)
def encode(i: int):
    outs = []
    with torch.no_grad():
        for batch in loader:
            # batch.bmg (batched molecular graph), batch.V_d (node feat), batch.X_d (edge feat)
            z = mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=i)
            outs.append(z)
    return torch.cat(outs, dim=0).cpu()

fingerprints = encode(i=0)   # [N, D_f]
encodings   = encode(i=1)    # [N, D_e]

# save to disk aligned with the original CSV row order
torch.save({"fingerprints": fingerprints, "encodings": encodings}, "data/tox21_mpnn_feats.pt")
print("Saved features to data/tox21_mpnn_feats.pt", fingerprints.shape, encodings.shape)

