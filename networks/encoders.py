import torch
import numpy as np

class NoEncoding(torch.nn.Module):
    def __init__(self, num_input):
        super().__init__()

        self.encoding_size = num_input

    def forward(self, coords):
        return coords

class FourierEncoding(torch.nn.Module):
    def __init__(self,
                 device,
                 num_input=2,
                 basis=6,
                 sigma=1,
                 gaussian=None):
        super().__init__()

        self.device = device
        self.num_input = num_input
        self.basis = basis
        self.gaussian = gaussian
        self.coefficients = (sigma * self.gaussian).to(self.device)
        self.encoding_size = self.num_input * 2 * self.basis

    def forward(self, coords):
        basis_values = torch.cat(self.basis * [coords], dim=-1)
        value = 2 * torch.pi * basis_values * self.coefficients
        values = torch.cat([torch.sin(value), torch.cos(value)], dim=-1)
        return values
    
class FreeEncoding(torch.nn.Module):
    def __init__(self,
                 num_input,
                 basis,
                 window_start,
                 device):
        super().__init__()

        self.num_input = num_input
        self.basis = basis
        self.window_start = window_start
        self.device = device

        self.encoding_size = self.num_input + self.num_input * 2 * self.basis
        self.scales = 2.0 ** torch.arange(0, basis).to(self.device)
        self.alpha = torch.ones(self.basis).float()
        self.ptr = self.window_start

    def update_alpha(self, current_iter, max_iter):
        # based on https://github.com/Jiawei-Yang/FreeNeRF/blob/main/internal/math.py#L277
        if current_iter < max_iter:
            freq_mask = np.zeros(self.basis)
            ptr = ((self.basis-self.window_start) * current_iter) / max_iter + self.window_start
            self.ptr = ptr
            # ptr = ptr if ptr < pos_enc_basis / 3 else pos_enc_basis / 3
            int_ptr = int(ptr)

            freq_mask[: int_ptr + 1] = 1.0  # assign the integer part
            freq_mask[int_ptr : int_ptr + 1] = (ptr - int_ptr)  # assign the fractional part

            self.alpha = torch.clip(torch.from_numpy(freq_mask), 1e-8, 1-1e-8).float() # for numerical stability
        else:
            self.ptr = self.basis
            self.alpha = torch.ones(self.basis).float()

    def forward(self, coords):
        xb = coords[..., None, :] * self.scales[:, None]
        four_feat = torch.sin(torch.stack([xb, xb + 0.5 * torch.pi], axis=-2))

        window = self.alpha.to(self.device)
        four_feat = window[..., None, None] * four_feat

        four_feat = four_feat.reshape((*coords.shape[:-1], -1))
        fin_values = torch.cat([coords, four_feat], dim=-1)
        return fin_values