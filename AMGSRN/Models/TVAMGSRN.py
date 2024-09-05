import torch
import torch.nn as nn
from AMGSRN import AMGSRN, weights_init
from AMG_Encoder import create_transformation_matrices, encode, feature_density
from typing import List, Optional
import math
   
class TVAMGSRN(AMGSRN):
    def __init__(self, opt):
        super(AMGSRN, self).__init__()
        
        self.n_grids : int = opt['n_grids']
        self.n_features : int = opt['n_features'] 
        self.feature_grid_shape : List[int] = [eval(i) for i in opt['feature_grid_shape'].split(',')]
        self.n_dims : int = opt['n_dims']
        self.n_outputs : int = opt['n_outputs'] 
        self.nodes_per_layer : int = opt['nodes_per_layer']
        self.n_layers : int = opt['n_layers'] 
        self.requires_padded_feats : bool = opt['requires_padded_feats']
        self.padding_size : int = 0
        self.full_shape = opt['full_shape']
        if(opt['requires_padded_feats']):
            self.padding_size : int = 16*int(math.ceil(max(1, (opt['n_grids']*opt['n_features'] )/16))) - \
                opt['n_grids']*opt['n_features'] 

        self.init_activations()
        self.randomize_grids(self.n_grids, self.n_features, 
                             self.feature_grid_shape, self.n_dims)
        
        def init_decoder_tcnn():
            import tinycudann as tcnn 
            input_size:int = opt['n_features']*opt['n_grids'] # + 6*3*2
            if(opt['requires_padded_feats']):
                input_size = opt['n_features']*opt['n_grids'] + self.padding_size
                
            decoder = tcnn.Network(
                n_input_dims=input_size,
                n_output_dims=opt['n_outputs'] ,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": opt['nodes_per_layer'],
                    "n_hidden_layers": opt['n_layers'] ,
                }
            )
                    
            return decoder
        
        def init_decoder_pytorch():
            decoder = nn.ModuleList()   
            
            first_layer_input_size:int = opt['n_features']*opt['n_grids'] # + 6*3*2
            if(opt['requires_padded_feats']):
                first_layer_input_size = opt['n_features']*opt['n_grids'] + self.padding_size
                                           
            layer = ReLULayer(first_layer_input_size, 
                opt['nodes_per_layer'], bias=opt['use_bias'], dtype=torch.float32)
            decoder.append(layer)
            
            for i in range(opt['n_layers'] ):
                if i == opt['n_layers']  - 1:
                    layer = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'] , 
                        bias=opt['use_bias'], dtype=torch.float32)
                    decoder.append(layer)
                else:
                    layer = ReLULayer(opt['nodes_per_layer'], opt['nodes_per_layer'], 
                        bias=opt['use_bias'], dtype=torch.float32)
                    decoder.append(layer)
            decoder = torch.nn.Sequential(*decoder)
            return decoder
        
        self.decoders = nn.ModuleList()
        for _ in range(opt['n_timesteps']):
            if(opt['use_tcnn_if_available']):
                try:
                    self.decoders.append(init_decoder_tcnn())
                except ImportError:
                    #print(f"Tried to use TinyCUDANN but found it was not installed - reverting to PyTorch layers.")
                    self.decoders.append(init_decoder_pytorch())
            else:                
                self.decoders.append(init_decoder_pytorch())
        
        self.reset_parameters()
    
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
        self.default_timestep = 0
    
    def set_default_timestep(self, t:int):
        self.default_timestep = t

    @property
    def transformation_matrices(self, t:int=None):
        if t is None:
            t = self.default_timestep
        return create_transformation_matrices(self.rotations[t], self.scales[t], self.translations[t])

    def get_model_parameters(self):
        return [
            {"params": [self.feature_grids]},
            {"params": self.decoders.parameters()}
        ]
    
    def reset_parameters(self):
        with torch.no_grad():
            feat_grid_shape = self.feature_grid_shape
            self.feature_grids =  torch.nn.parameter.Parameter(
                torch.ones(
                    [self.n_grids, self.n_features] + feat_grid_shape,
                    device = self.feature_grids.device
                ).uniform_(-0.0001, 0.0001),
                requires_grad=True
            )
            for decoder in self.decoders:
                decoder.apply(weights_init)   

    def randomize_grids(self, n_timesteps:int, n_grids:int, n_features:int,
                 feat_grid_shape:List[int], n_dims:int):  
        with torch.no_grad():     
            d = "cpu"
            s = torch.ones([n_timesteps, n_grids, n_dims], dtype=torch.float32, device = d)
            s += torch.rand_like(s)*0.05
            s = self.inv_scale_activation(s)
            r = torch.zeros([n_timesteps, n_grids, 4], dtype=torch.float32, device = d)
            r[:,-1] = 1.0
            t = torch.zeros([n_timesteps,n_grids, n_dims], dtype=torch.float32, device = d)
            t += torch.rand_like(t)*0.05
            
        self._scales = torch.nn.Parameter(s,requires_grad=True)   
        self._rotations = torch.nn.Parameter(r,requires_grad=True)   
        self.translations = torch.nn.Parameter(t,requires_grad=True)   

        self.feature_grids =  torch.nn.parameter.Parameter(
            torch.ones(
                [n_timesteps, n_grids, n_features] + feat_grid_shape,
                dtype=torch.float32
            ).uniform_(-0.0001, 0.0001),
            requires_grad=True
        )
    
    def prepare_timestep(self, t:int):
        if t == 0:
            return
        elif t == self.opt['n_timesteps']:
            print("Error: t cannot be equal to n_timesteps")
            quit(1)
        else:
            with torch.no_grad():
                self.feature_grids[t] = self.feature_grids[t-1]
                self.translations[t] = self.translations[t-1]
                self._rotations[t] = self._rotations[t-1]
                self._scales[t] = self._scales[t-1]
    
    def transform(self, x : torch.Tensor, t:int=None) -> torch.Tensor:
        if t is None:
            t = self.default_timestep
        '''
        Transforms global coordinates x to local coordinates within
        each feature grid, where feature grids are assumed to be on
        the boundary of [-1, 1]^n_dims in their local coordinate system.
        Scales the grid by a factor to match the gaussian shape
        (see feature_density_gaussian()). Assumes S*R*T order
        
        x: Input coordinates with shape [batch, n_dims]
        returns: local coordinates in a shape [n_grids, batch, n_dims]
        '''
                        
        batch : int = x.shape[0]
        dims : int = x.shape[1]
        ones = torch.ones([batch, 1], 
            device=x.device,
            dtype=torch.float32)
            
        x = torch.cat([x, ones], dim=1)
        transformed_points = torch.matmul(self.transformation_matrices(t), 
                            x.transpose(0, 1)).transpose(1, 2)
        transformed_points = transformed_points[...,0:dims]
        
        return transformed_points
   
    def inverse_transform(self, x : torch.Tensor, t:int=None) -> torch.Tensor:
        '''
        Transforms local coordinates within each feature grid x to 
        global coordinates. Scales local coordinates by a factor
        so as to be consistent with the transform() method, which
        attempts to align feature grids with the guassian density 
        calculated in feature_density_gaussian().Assumes S*R*T order,
        so inverse is T^(-1)*R^T*(1/S)
        
        x: Input coordinates with shape [batch, n_dims]
        returns: local coordinates in a shape [n_grids, batch, n_dims]
        '''
        if t is None:
            t = self.default_timestep
        local_to_global_matrices = torch.linalg.inv(self.transformation_matrices(t))
       
        batch : int = x.shape[0]
        dims : int = x.shape[1]
        ones = torch.ones([batch, 1], 
            device=x.device,
            dtype=torch.float32)
        
        x = torch.cat([x, ones], dim=1)
        
        transformed_points = torch.matmul(local_to_global_matrices,
            x.transpose(0,1)).transpose(1, 2)
        transformed_points = transformed_points[...,0:dims]

        return transformed_points
    
    def feature_density(self, x : torch.Tensor, t:int=None) -> torch.Tensor:
        if t is None:
            t = self.default_timestep
        return feature_density(x, self.rotations[t], self.scales[t], self.translations[t])  
    
    def forward(self, x : torch.Tensor, t:int=None) -> torch.Tensor:
        if t is None:
            t = self.default_timestep
        feats = encode(x, self.rotations[t], self.scales[t], self.translations[t], self.feature_grids[t])  
        y = self.decoder(feats)
        y = y.float() * (self.volume_max - self.volume_min) + self.volume_min   
        return y

