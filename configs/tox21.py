from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.seed = 42  # ← move seed to top level

    config.dataset = ConfigDict()
    config.dataset.name = "tox21_binary"
    config.dataset.path = "data/tox21_binary.csv"
    config.dataset.smiles_col = "smiles"
    config.dataset.label_col = "label"
    config.dataset.task_type = "classification"

    config.model = ConfigDict()
    config.model.z_dim = 128

    config.training = ConfigDict()
    config.training.batch_size = 128
    config.training.n_iters = 10000
    config.training.lr = 0.001
    config.training.reweight = 'none'
    config.training.unit_factor = False

    config.optim = ConfigDict()
    config.optim.lr = 0.001

    return config
