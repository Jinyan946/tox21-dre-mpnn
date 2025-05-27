import torch
import torch.nn as nn
import numpy as np
from . import utils, layers, layerspp, normalization
import math
import functools


# to save us from importing lots of things for mnist
class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embed_dim=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.dim = embed_dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class Dense_act(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim, latent=32):
        super().__init__()
        # self.dense = nn.Linear(input_dim, output_dim)
        self.dense = nn.Sequential(
            nn.Linear(input_dim, latent),
            nn.Tanh(),  # added
            nn.Linear(latent, latent),  # added
            nn.Tanh(),  # added
            nn.Linear(latent, output_dim),
        )

    def forward(self, x):
        return self.dense(x)[..., None, None]


# TODO: renamed from ncsnunet_t_deeper_v4_activation
@utils.register_model(name="ncsnunet_t")
class NCSNUNet_t(nn.Module):
    """U-Net architecture with fc layer at the very end
    Starting from 64 channels instead of 32, and no "additional" layer (just the
    usual, but going from 256->512 instead of 128 -> 256.
    """

    def __init__(self, config, channels=[64, 128, 256, 512], embed_dim=256):
        """Initialize a time-dependent score-based network.
        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        self.config = config
        if config.model.embedding_type == "positional":
            print("using positional embeddings!")
            self.embed = nn.Sequential(
                SinusoidalPosEmb(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim)
            )
            # self.embed = SinusoidalPosEmb(embed_dim=embed_dim)
        elif config.model.embedding_type == "linear":
            print("using linear embeddings!")
            self.embed = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.Tanh(),  # added
                nn.Linear(embed_dim, embed_dim),  # added
                nn.Tanh(),  # added
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            print("using Gaussian random feature embeddings!")
            self.embed = nn.Sequential(
                GaussianFourierProjection(embed_dim=embed_dim // 2),
                nn.Linear(embed_dim, embed_dim),
            )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense_act(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense_act(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense_act(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense_act(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.tdense4 = Dense_act(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.tdense3 = Dense_act(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.tdense2 = Dense_act(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # self.out_conv = nn.Conv2d(channels[0], 1, kernel_size=1)
        img_size = (
            config.data.image_size * config.data.image_size * config.data.num_channels
        )
        self.out_fc = nn.Linear(img_size, 1)

    # The swish activation function
    # modified for pickle
    def act(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x, t):
        n = x.size(0)

        # print('in switched score model!')
        if len(t.size()) == 0:
            t = torch.ones(x.size(0), device=x.device) * t
        # Obtain the Gaussian random feature embedding for t
        t = t.squeeze()
        # TODO: should we take the log of t if doing fourier? note that originally it was the log stdev of marginal
        if self.config.model.embedding_type == "fourier":
            # embed = self.act(self.embed(torch.log(t)))
            embed = self.act(self.embed(t))
        elif self.config.model.embedding_type == "linear":
            embed = self.act(self.embed(t.view(-1, 1)))
        else:
            embed = self.act(self.embed(t))

        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 = h1 + self.dense1(embed)

        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)

        # this is the middle of the unet, incorporating information from t
        temb = self.dense4(embed)
        h4 = h4 + temb
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)  # (64, 256, 2, 2)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.tdense4(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.tdense3(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.tdense2(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # do we need some activations here?
        h = h.view(n, -1)
        out = self.out_fc(h)

        return out


# TODO: renamed from ncsnunet_t_deeper_v4_activation
@utils.register_model(name="c_ncsnunet_t")
class C_NCSNUNet_t(nn.Module):
    """U-Net architecture with fc layer at the very end
    Starting from 64 channels instead of 32, and no "additional" layer (just the
    usual, but going from 256->512 instead of 128 -> 256.
    """

    def __init__(self, config, channels=[64, 128, 256, 512], embed_dim=256):
        """Initialize a time-dependent score-based network.
        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        self.config = config
        if config.model.embedding_type == "positional":
            print("using positional embeddings!")
            self.embed = nn.Sequential(
                SinusoidalPosEmb(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim)
            )
            # self.embed = SinusoidalPosEmb(embed_dim=embed_dim)
        elif config.model.embedding_type == "linear":
            print("using linear embeddings!")
            self.embed = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.Tanh(),  # added
                nn.Linear(embed_dim, embed_dim),  # added
                nn.Tanh(),  # added
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            print("using Gaussian random feature embeddings!")
            self.embed = nn.Sequential(
                GaussianFourierProjection(embed_dim=embed_dim // 2),
                nn.Linear(embed_dim, embed_dim),
            )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense_act(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense_act(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense_act(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense_act(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.tdense4 = Dense_act(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.tdense3 = Dense_act(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.tdense2 = Dense_act(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # self.out_conv = nn.Conv2d(channels[0], 1, kernel_size=1)
        # img_size = (
        #     config.data.image_size * config.data.image_size * config.data.num_channels
        # )
        # self.out_fc = nn.Linear(img_size, 1)

    # The swish activation function
    # modified for pickle
    def act(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x, t):
        n = x.size(0)

        # print('in switched score model!')
        if len(t.size()) == 0:
            t = torch.ones(x.size(0), device=x.device) * t
        # Obtain the Gaussian random feature embedding for t
        t = t.squeeze()
        # TODO: should we take the log of t if doing fourier? note that originally it was the log stdev of marginal
        if self.config.model.embedding_type == "fourier":
            # embed = self.act(self.embed(torch.log(t)))
            embed = self.act(self.embed(t))
        elif self.config.model.embedding_type == "linear":
            embed = self.act(self.embed(t.view(-1, 1)))
        else:
            embed = self.act(self.embed(t))

        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 = h1 + self.dense1(embed)

        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)

        # this is the middle of the unet, incorporating information from t
        temb = self.dense4(embed)
        h4 = h4 + temb
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)  # (64, 256, 2, 2)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.tdense4(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.tdense3(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.tdense2(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        return h


@utils.register_model(name="ncsnunet_x")
class NCSNUNet_x(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, config, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        self.config = config
        if config.model.embedding_type == "positional":
            print("using positional embeddings!")
            self.embed = nn.Sequential(
                SinusoidalPosEmb(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim)
            )
        else:
            print("using Gaussian random feature embeddings!")
            self.embed = nn.Sequential(
                GaussianFourierProjection(embed_dim=embed_dim // 2),
                nn.Linear(embed_dim, embed_dim),
            )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        t = t.squeeze()
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 = h1 + self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)

        temb = self.dense4(embed)
        h4 = h4 + temb
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h = h + self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = h + self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = h + self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        out = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output (maybe doesn't matter too much?)
        if self.config.model.scale_by_sigma:
            out = out / t[:, None, None, None]

        return out


ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
ResnetBlockBigGAN_noskip = layerspp.ResnetBlockBigGANpp_noskip
AttnBlock_noskip = layerspp.AttnBlockpp_noskip


@utils.register_model(name="ncsnpp_t")
class NCSNpp_t(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            config.data.image_size // (2**i) for i in range(num_resolutions)
        ]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = progressive_input = (
            config.model.progressive_input.lower()
        )
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional", "linear"]
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        print("using embedding type: {}".format(embedding_type))
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            assert (
                config.training.continuous
            ), "Fourier features are only used for continuous training."

            modules.append(
                layerspp.GaussianFourierProjection(
                    embedding_size=nf, scale=config.model.fourier_scale
                )
            )
            embed_dim = nf * 2

        elif embedding_type == "positional":
            embed_dim = nf
        elif embedding_type == "linear":
            embed_dim = nf
            modules.append(nn.Linear(1, embed_dim))
            modules[0].weight.data = default_initializer()(modules[0].weight.shape)
            nn.init.zeros_(modules[0].bias)
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block

        channels = config.data.num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        self.time_fc = nn.Linear(784, 1)
        self.time_fc.weight.data = default_initializer()(self.time_fc.weight.shape)
        nn.init.zeros_(self.time_fc.bias)

    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            # used_sigmas = time_cond
            # temb = modules[m_idx](torch.log(used_sigmas))
            # temb = modules[m_idx](used_sigmas)
            temb = modules[m_idx](time_cond)
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            # used_sigmas = self.sigmas[time_cond.long()]  # not using this atm
            if self.config.model.embedding_scale:
                temb = layers.get_timestep_embedding((timesteps * 999), self.nf)
            else:
                temb = layers.get_timestep_embedding(timesteps, self.nf)
        elif self.embedding_type == "linear":
            temb = modules[m_idx](time_cond.view(-1, 1))
            m_idx += 1
        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        # finally, get the time-only scores
        h = h.view(x.size(0), -1)
        h = self.time_fc(h)

        # if self.config.model.scale_by_sigma:
        #   used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
        #   h = h / used_sigmas

        return h


@utils.register_model(name="ncsnpp_t_v2")
class NCSNpp_t_v2(nn.Module):
    """NCSN++ model.
    This one has an updated linear embedding, and is not able to use {positional,fourier} embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            config.data.image_size // (2**i) for i in range(num_resolutions)
        ]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = progressive_input = (
            config.model.progressive_input.lower()
        )
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type == "linear"
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        embed_dim = nf

        # TODO: we can merge the linear embedding + the conditional transformation into one block
        # TODO: check if you need to apply default_initializer() here
        # TODO: also check if we want to use tanh()
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, nf * 4),
            nn.Tanh(),
            nn.Linear(nf * 4, nf * 4),
        )

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block

        channels = config.data.num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        self.time_fc = nn.Linear(784, 1)
        self.time_fc.weight.data = default_initializer()(self.time_fc.weight.shape)
        nn.init.zeros_(self.time_fc.bias)

    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        # apply linear embedding, where self.conditional has been subsumed inside
        # note that m_idx will still start at 0 bc it's not part of self.all_modules
        temb = self.embed(time_cond.view(-1, 1))

        if not self.config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        # finally, get the time-only scores
        h = h.view(x.size(0), -1)
        h = self.time_fc(h)

        # if self.config.model.scale_by_sigma:
        #   used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
        #   h = h / used_sigmas

        return h


@utils.register_model(name="c_ncsnpp_t")
class C_NCSNpp_t(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            config.data.image_size // (2**i) for i in range(num_resolutions)
        ]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = progressive_input = (
            config.model.progressive_input.lower()
        )
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional", "linear"]
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        print("using embedding type: {}".format(embedding_type))
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            assert (
                config.training.continuous
            ), "Fourier features are only used for continuous training."

            modules.append(
                layerspp.GaussianFourierProjection(
                    embedding_size=nf, scale=config.model.fourier_scale
                )
            )
            embed_dim = nf * 2

        elif embedding_type == "positional":
            embed_dim = nf
        elif embedding_type == "linear":
            embed_dim = nf
            modules.append(nn.Linear(1, embed_dim))
            modules[0].weight.data = default_initializer()(modules[0].weight.shape)
            nn.init.zeros_(modules[0].bias)
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block

        channels = config.data.num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        # self.time_fc = nn.Linear(784, 1)
        # self.time_fc.weight.data = default_initializer()(self.time_fc.weight.shape)
        # nn.init.zeros_(self.time_fc.bias)

    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            # used_sigmas = time_cond
            # temb = modules[m_idx](torch.log(used_sigmas))
            # temb = modules[m_idx](used_sigmas)
            temb = modules[m_idx](time_cond)
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            # used_sigmas = self.sigmas[time_cond.long()]  # not using this atm
            if self.config.model.embedding_scale:
                temb = layers.get_timestep_embedding((timesteps * 999), self.nf)
            else:
                temb = layers.get_timestep_embedding(timesteps, self.nf)
        elif self.embedding_type == "linear":
            temb = modules[m_idx](time_cond.view(-1, 1))
            m_idx += 1
        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        # finally, get the time-only scores
        # h = h.view(x.size(0), -1)
        # h = self.time_fc(h)

        # if self.config.model.scale_by_sigma:
        #   used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
        #   h = h / used_sigmas

        return h
