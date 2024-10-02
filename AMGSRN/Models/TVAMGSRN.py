import torch
import torch.nn as nn
from Models.layers import ReLULayer
from Models.AMGSRN import AMGSRN, weights_init
from AMG_Encoder import create_transformation_matrices, encode, feature_density
from typing import List, Optional
import math

class TVAMGSRN(nn.Module):
    def __init__(self, opt):
        super(TVAMGSRN, self).__init__()
        
        self.n_timesteps = opt['n_timesteps']
        self.default_timestep = 0
        self.full_shape = opt['full_shape']


        # Create a list of AMGSRN models, one for each timestep
        self.models = nn.ModuleList([AMGSRN(opt) for _ in range(self.n_timesteps)])

        self.register_buffer(
            "volume_min",
            torch.tensor([opt['data_min']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([opt['data_max']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )

        self.last_timestep_init = opt['last_timestep_init']

    def min(self):
        return self.volume_min

    def max(self):
        return self.volume_max

    def set_default_timestep(self, t: int):
        if(t != self.default_timestep):
            self.default_timestep = t
    def get_default_timestep(self):
        return self.default_timestep

    @property
    def feature_grids(self):
        return self.models[self.default_timestep].feature_grids
    
    def transformation_matrices(self, t: int = None):
        if t is None:
            t = self.default_timestep
        return self.models[t].transformation_matrices

    def get_volume_extents(self):
        return self.full_shape
    
    def unload_timestep(self, timestep:int):
        self.models[timestep].to('cpu')

    def get_model_parameters(self):
        return self.models[self.default_timestep].get_model_parameters()

    def get_transform_parameters(self):
        return self.models[self.default_timestep].get_transform_parameters()

    def reset_parameters(self):
        for model in self.models:
            model.reset_parameters()

    def randomize_grids(self, n_timesteps: int, n_grids: int, n_features: int,
                        feat_grid_shape: List[int], n_dims: int):
        for model in self.models:
            model.randomize_grids(1, n_grids, n_features, feat_grid_shape, n_dims)

    def prepare_timestep(self, t: int):
        if t == 0:
            return
        elif t == self.n_timesteps:
            print("Error: t cannot be equal to n_timesteps")
            quit(1)
        else:
            with torch.no_grad():
                # Clone feature grids and transformation parameters
                if(self.last_timestep_init): 
                    self.models[t].feature_grids = nn.Parameter(self.models[t-1].feature_grids.clone().detach())
                    self.models[t].translations = nn.Parameter(self.models[t-1].translations.clone().detach())
                    self.models[t]._rotations = nn.Parameter(self.models[t-1]._rotations.clone().detach())
                    self.models[t]._scales = nn.Parameter(
                        self.models[t].inv_scale_activation(0.75 * self.models[t-1].scales.clone().detach())
                    )                
                    self.models[t].decoder.load_state_dict(self.models[t-1].decoder.state_dict())

    def transform(self, x: torch.Tensor, t: int = None) -> torch.Tensor:
        if t is None:
            t = self.default_timestep
        return self.models[t].transform(x)

    def inverse_transform(self, x: torch.Tensor, t: int = None) -> torch.Tensor:
        if t is None:
            t = self.default_timestep
        return self.models[t].inverse_transform(x)

    def feature_density(self, x: torch.Tensor, t: int = None) -> torch.Tensor:
        if t is None:
            t = self.default_timestep
        return self.models[t].feature_density(x)

    def forward(self, x: torch.Tensor, t: int = None) -> torch.Tensor:
        if t is None:
            t = self.default_timestep
        y = self.models[t](x)
        y = y.float() * (self.volume_max - self.volume_min) + self.volume_min
        return y
