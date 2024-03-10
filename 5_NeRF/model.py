import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class NeRF(nn.Module):
    def __init__(self,  d_input=3, n_layers=8, hidden_dim=256, skips=[4], d_viewdirs=None):
        """ 
        """
        super(NeRF, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.d_input = d_input
        self.skips = skips
        self.d_viewdirs = d_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(d_input, hidden_dim)] + 
            [nn.Linear(hidden_dim, hidden_dim) if i not in self.skips else nn.Linear(hidden_dim + d_input, hidden_dim) for i in range(n_layers-1)]
        )
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        if d_viewdirs is not None:
            self.views_linears = nn.ModuleList([nn.Linear(d_viewdirs + hidden_dim, hidden_dim//2)])
        
        if d_viewdirs is not None:
            self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
            self.alpha_linear = nn.Linear(hidden_dim, 1)
            self.rgb_linear = nn.Linear(hidden_dim//2, 3)
        else:
            self.output_linear = nn.Linear(hidden_dim, 4)


    def forward(self, x, viewdirs=None):
        # TASK 3: Implement the NeRF forward.
        # Make sure you understand the __init__ function first
        # NOTE: do not change the names in the init function

        input_pts = x
        input_views = viewdirs

        h = input_pts
        # for each layer with index i
        for i, l in enumerate(self.pts_linears):
            h = ... # HINT: feed h to the layer i and rewrite to h
            h = ... # HINT: use relu
            if i in self.skips:
                h = ... # implement skip with torch.cat

        if self.d_viewdirs is not None:
            alpha = ... # HINT: feed h to alpha linear
            feature = ... # HINT: feed h to feature linear
            h = ... # HINT: concat feature and input_views to create the input for the views_linreas
        
            for i, l in enumerate(self.views_linears):
                h = ... # HINT: forward for views_linears of i
                h = ... # HINT: Use relu

            rgb = ... # HINT: calculate rgb values with rgb_layer
            outputs = ... # HINT: concat rgb and alpha
        else:
            outputs = self.output_linear(h)

        return outputs    



class Embedder(nn.Module):
    """
    Sine-cosine positional encoder for input points.
    """
    def __init__(self, d_input, n_freqs, log_space=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # TASK 2: Complete the implementation of the Embedder
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: ...) # HINT: use torch.sin
            self.embed_fns.append(lambda x, freq=freq: ...) # HINT: use torch.cos

    def forward(self, x):
        """
        Apply positional encoding to input.
        """
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)
