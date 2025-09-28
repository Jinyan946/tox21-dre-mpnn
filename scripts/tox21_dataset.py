# tox21_dataset.py
import numpy as np, torch

class TwoClassArrayDataset:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        X, y = d["X"], d["y"]
        # enforce float32 for MPNN embeddings
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.X_pos = torch.from_numpy(X[y == 1])
        self.X_neg = torch.from_numpy(X[y == 0])
 
    @property
    def dim(self):
        return self.X_pos.shape[1]

    def sample_p(self, batch):
        idx = torch.randint(0, self.X_pos.shape[0], (batch,))
        return self.X_pos[idx]

    def sample_q(self, batch):
        idx = torch.randint(0, self.X_neg.shape[0], (batch,))
        return self.X_neg[idx]