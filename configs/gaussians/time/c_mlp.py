from configs.default_toy_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.joint = False
    training.reweight = "obj_var"
    training.conditional = True
    training.prob_path = "OneVP"
    training.batch_size = 256
    training.unit_factor = True

    # data
    data = config.data
    data.dataset = "Gaussians"
    data.eps1 = 0.0
    data.eps2 = 1e-5
    data.dim = 2
    data.centered = False

    # model
    model = config.model
    model.type = "time"
    model.param = False
    model.name = "toy_time_scorenet"
    model.nf = 64
    model.z_dim = 256

    # optimization
    optim = config.optim
    optim.lr = 1e-3

    return config
