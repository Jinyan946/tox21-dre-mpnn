from configs.default_toy_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.joint = False
    training.reweight = "path_var"
    training.conditional = False
    training.prob_path = "TwoSB"
    training.batch_size = 256
    training.unit_factor = True

    # for TwoSB
    training.two_sb_var = 2.0
    training.use_two_sb = True

    # data
    data = config.data
    data.dataset = "GMMs"
    data.eps1 = 1e-5
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

    return config
