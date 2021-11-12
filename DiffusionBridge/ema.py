"""
A module with some utility functions to implement exponential moving average update of neural network parameters.
"""

def ema_register(score_net):
    ema_parameters = {}
    for name, param in score_net.named_parameters():
        ema_parameters[name] = param.data.clone()
    return ema_parameters

def ema_update(ema_parameters, score_net, ema_momentum):
    for name, param in score_net.named_parameters():
        ema_parameters[name].data = ema_momentum * ema_parameters[name].data + (1.0 - ema_momentum) * param.data

def ema_copy(ema_parameters, score_net):
    for name, param in score_net.named_parameters():
        param.data.copy_(ema_parameters[name].data) 


    