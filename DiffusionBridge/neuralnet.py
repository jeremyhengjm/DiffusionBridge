"""
A module to approximate score functions with neural networks.
"""

import torch
import torch.nn.functional as F
from torch import nn
import math

def get_timestep_embedding(timesteps, embedding_dim = 128):
    """
    From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py

    Parameters
    ----------
    timesteps : state (N, 1)

    embedding_dim : int specifying dimension of time embedding 
                        
    Returns
    -------    
    emb : time embedding (N, embedding_dim)
    """
    scaling_factor = 100.0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)
    emb = scaling_factor * timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn = F.relu):
        """
        Parameters
        ----------    
        input_dim : int specifying dimension of input 

        layer_widths : list specifying width of each layer 
            (len is the number of layers, and last element is the output dimension)

        activate_final : bool specifying if activation function is applied in the final layer

        activation_fn : activation function for each layer        
        """
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x

class ScoreNetwork(torch.nn.Module):

    def __init__(self, dimension, encoder_layers = [16], pos_dim = 16, decoder_layers = [128,128]):
        """
        Parameters
        ----------    
        dimension : int specifying dimension of state variable (same as output of network)

        encoder_layers : list specifying width of each encoder layer 

        pos_dim : int specifying dimension of time embedding

        decoder_layers : list specifying width of each decoder layer 
        """
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, dimension]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths = decoder_layers + [dimension],
                       activate_final = False,
                       activation_fn = torch.nn.LeakyReLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths = encoder_layers + [t_enc_dim],
                             activate_final = False,
                             activation_fn = torch.nn.LeakyReLU())

        self.x_encoder = MLP(dimension,
                             layer_widths = encoder_layers + [t_enc_dim],
                             activate_final = False,
                             activation_fn = torch.nn.LeakyReLU())

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : time step (N, 1)

        x : state (N, dimension)
                        
        Returns
        -------    
        out :  score (N, dimension)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim) # size (N, temb_dim)
        temb = self.t_encoder(temb) # size (N, t_enc_dim)
        xemb = self.x_encoder(x) # size (N, t_enc_dim)
        h = torch.cat([xemb, temb], -1) # size (N, 2 * t_enc_dim)
        out = self.net(h) # size (N, dimension)
        return out
