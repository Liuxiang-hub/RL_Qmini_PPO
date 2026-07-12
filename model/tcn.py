from torch import nn
from typing import Union
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from rl.module.common import Net
from torch.distributions import Normal


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 4, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=1, padding=0),
            nn.ReLU(True)
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(8 * 8 * 11, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 4 * 4 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 6, 6))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, output_padding=0)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class RSBlock(nn.Module):
    """Deep Residual Shrinkage Network"""

    def __init__(self, channel):
        super(RSBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, int(channel / 16)),
            # nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / 16), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_abs = torch.abs(x)
        tau = torch.flatten(F.adaptive_avg_pool1d(x_abs, 1), 1)
        tau = torch.mul(tau, self.net(tau)).unsqueeze(2)
        x_abs = torch.max(x_abs - tau, torch.zeros_like(tau, requires_grad=False))  # soft thresholding
        return torch.mul(torch.sign(x), x_abs)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Networks"""

    def __init__(self, channel):
        super(SEBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, int(channel / 16)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / 16), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = torch.flatten(F.adaptive_avg_pool1d(x, 1), 1)
        w = self.net(w).unsqueeze(2)
        return torch.mul(w, x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, dilation, padding, dropout=0.2, rsnet=False, senet=True, **kwargs):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(input_channel, output_channel, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(output_channel, output_channel, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(input_channel, output_channel, 1) if input_channel != output_channel else None
        self.rsnet = RSBlock(output_channel) if rsnet else None
        self.senet = SEBlock(output_channel) if senet else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.rsnet is not None:
            out = self.rsnet(out)
        if self.senet is not None:
            out = torch.mul(self.senet(out), out)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, **kwargs):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout, **kwargs)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNEncoder(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, output_dim: int, **kwargs):
        super(TCNEncoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        num_channels = (input_dim, )

        # self.pre_net = nn.Sequential(
        #     nn.Linear(input_dim, num_channels[0]), nn.ReLU(),  # 128 todo
        #     nn.Linear(num_channels[0], num_channels[0]), nn.ReLU())  # 128, 128 todo
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=3, dropout=0., **kwargs)  # 128 todo
        # self.post_net = nn.Sequential(nn.Linear(num_channels[-1], output_dim), nn.Tanh())

    def forward(self, s: torch.Tensor):
        # z = self.pre_net(s)
        z = self.tcn(s.transpose(1, 2))
        return z[..., -1]  # self.post_net(z[..., -1])


class Actor(nn.Module):
    def __init__(self,
                 num_observations=256,
                 num_actions=12,
                 seq_len=3,
                 hidden_layers=(128, 64),
                 activation='relu',
                 device='cpu',
                 deploy=False,
                 **kwargs
                 ):
        super().__init__()

        self.model = TCNEncoder(seq_len=seq_len, input_dim=num_observations, output_dim=num_actions, device=device)
        self.net = Net(num_observations, num_actions, hidden_layers, output_activation=activation)  # (128, 64) todo
        self.deploy = deploy
        self.output_dim = num_actions
        self.std = nn.Parameter(torch.ones(num_actions) * 1)

    def forward(self, s):
        s = s.type(torch.float32)
        g_o = self.model(s)
        mu = torch.tanh(self.net(g_o))
        if self.deploy:
            return mu
        logits = mu, self.std
        if self.training:
            dist = Normal(*logits)
            return {'logits': logits, 'dist': dist, 'act': dist.sample()}
        else:
            return {'logits': logits, 'act': mu}


class Critic(nn.Module):
    def __init__(
            self,
            num_critic_obs=256,
            seq_len=256,
            hidden_layers=(128, 64),
            activation='relu',
            device="cpu",
            **kwargs
    ) -> None:
        super().__init__()
        self.model = TCNEncoder(seq_len=seq_len, input_dim=num_critic_obs, output_dim=hidden_layers[-1], device=device)

        self.preprocess = Net(num_critic_obs, 0, hidden_layers, output_activation=activation)  # (128, 64) todo
        self.value = nn.Linear(hidden_layers[-1], 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s1 = self.model(s)
        return self.value(self.preprocess(s1))
