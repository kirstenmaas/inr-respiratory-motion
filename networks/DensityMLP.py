import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import networks.encoders as encoders
import networks.hash_encoders as hash_encoders

import torch
import math

class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max()

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)
    
class DensityMLP(nn.Module):
    def __init__(self, kwargs):
        super().__init__()

        self.model_definition = kwargs

        self.in_features = kwargs['in_features']
        self.hidden_channels = kwargs['hidden_channels']
        self.num_layers = kwargs['num_layers']
        
        self.device = kwargs['device']
        
        self.pos_enc = kwargs['pos_enc']
        self.freq_basis = kwargs['freq_basis']

        self.num_obsv = kwargs['num_obsv']
        self.latent_dim = kwargs['latent_dim']

        self.frame_rate = kwargs['frame_rate']
        self.min_heart_rate = 30
        self.max_heart_rate = 150
        self.resp_cycle_time = 5 # in seconds

        self.cardiac_latent = nn.Parameter(torch.rand((self.num_obsv, 1)))
        self.respiratory_latent = nn.Parameter(torch.rand((self.num_obsv, 1)))
        self.contrast_latent = nn.Parameter(torch.linspace(0.5, 1., self.num_obsv).reshape((-1, 1)))

        self.min_hr_frames = (self.min_heart_rate / 60) * self.frame_rate
        self.max_hr_frames = (self.max_heart_rate / 60) * self.frame_rate

        self.ecg_period = nn.Parameter(torch.tensor([1/4]))

        self.encoder = encoders.NoEncoding(num_input=self.in_features)
        if self.pos_enc == 'fourier':
            self.freq_sigma = kwargs['freq_sigma']
            self.freq_gaussian = kwargs['freq_gaussian']
            self.encoder = encoders.FourierEncoding(num_input=self.in_features, basis=self.freq_basis, sigma=self.freq_sigma, device=self.device, gaussian=self.freq_gaussian)
        elif self.pos_enc == 'free':
            self.window_start = kwargs['window_start']
            self.encoder = encoders.FreeEncoding(num_input=self.in_features, basis=self.freq_basis, window_start=self.window_start, device=self.device)
        elif self.pos_enc == 'hash':
            self.encoder = hash_encoders.MultiResHashGrid(num_input=self.in_features)
        elif self.pos_enc == 'anhash':
            self.encoder = hash_encoders.AnnealedMultiResHashGrid(num_input=self.in_features, encoding_config={})

        self.cardiac_dim = 2
        encoded_dim = self.encoder.encoding_size + self.latent_dim

        self.layers = torch.nn.ModuleList()
        dim = encoded_dim
        for ii in range(self.num_layers - 1):
            self.layers.append(LipschitzLinear(dim, self.hidden_channels))
            dim = self.hidden_channels

        self.layer_output = LipschitzLinear(dim, 1)
        self.relu = torch.nn.ReLU()

    def forward_net(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        return self.layer_output(x)

    def forward(self, x, id):
        encoded_x = self.encoder(x)

        latents = torch.cat(
            (self.cardiac_latent, self.respiratory_latent, self.contrast_latent), 
            dim=-1
        )
        encoded_l = torch.clamp(latents[id], min=-1, max=1.)

        values = torch.cat([encoded_x, encoded_l], dim=-1)

        raw_density = self.forward_net(values)  # (..., 1)
        return torch.nn.Softplus()(raw_density)
    
    def forward_ind(self, x, id1, id2, id3):
        encoded_x = self.encoder(x)
        encoded_1 = torch.clamp(self.cardiac_latent[id1], min=-1., max=1.)
        encoded_2 = torch.clamp(self.respiratory_latent[id2], min=-1., max=1.)
        encoded_3 = torch.clamp(self.contrast_latent[id3], min=-1., max=1.)

        values = torch.cat([encoded_x, encoded_1, encoded_2, encoded_3], dim=-1)

        raw_density = self.forward_net(values)  # (..., 1)
        return torch.nn.Softplus()(raw_density)
    
    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        loss_lipc = loss_lipc * self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_weights(self):
        for name, param in self.named_parameters():
            if 'layer' in name or 'ecg_period' in name:
                param.requires_grad = False

    @torch.no_grad()
    def save(self, filename: str, training_information: dict) -> None:
        save_parameters = {
            'model': self.state_dict(),
            'parameters': self.model_definition,
        }

        torch.save(
            save_parameters,
            f=filename)