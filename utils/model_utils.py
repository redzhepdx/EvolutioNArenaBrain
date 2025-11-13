import numpy as np
import torch.nn as nn


# Layer Initialization
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


def weights_init(model):
    if isinstance(model, nn.Linear):
        print(model)
        model.weight.data.uniform_(*hidden_init(model))


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def hard_copy_weights(target, source):
    # Copy weights from source to target network (part of initialization)
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)