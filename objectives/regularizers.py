import torch
import torch.nn.functional as F

def compute_periodic_loss(signal, period):
    period_floor = int(torch.floor(period))
    period_ceil = period_floor + 1
    alpha = period - period_floor

    shifted_floor = torch.roll(signal, -period_floor, dims=0)
    shifted_ceil = torch.roll(signal, -period_ceil, dims=0)

    shifted = torch.lerp(shifted_floor, shifted_ceil, alpha)
    loss = torch.nn.functional.mse_loss(signal, shifted)

    return loss

def compute_smoothness_loss_resp(resp_signal, weights, step_size=1):
    diff = resp_signal[:-step_size] - resp_signal[step_size:]
    loss = (weights[:-step_size] * diff.pow(2)).mean()
    return loss

def compute_smoothness_loss_contr(contr_signal, step_size=1):
    loss = torch.nn.functional.mse_loss(contr_signal[:-step_size], contr_signal[step_size:])
    return loss

def compute_second_order_smoothness_loss(contr_signal, step_size=1):
    diff1 = contr_signal[step_size:] - contr_signal[:-step_size]
    diff2 = diff1[step_size:] - diff1[:-step_size]
    loss = diff2.pow(2).mean()
    return loss