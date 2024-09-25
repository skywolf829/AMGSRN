import torch
import torch.nn as nn
from Models.layers import ReLULayer
from Other.utility_functions import make_coord_grid
from typing import List, Optional
import math
from AMG_Encoder import create_transformation_matrices, encode, feature_density

def weights_init(m):
    classname = m.__class__.__name__
    if classname.lower().find('linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if(m.bias is not None):
            torch.nn.init.normal_(m.bias, 0, 0.001) 
   
class AMGSRN(nn.Module):
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
        
        if(opt['use_tcnn_if_available']):
            try:
                self.decoder = init_decoder_tcnn()
            except ImportError:
                print(f"Tried to use TinyCUDANN but found it was not installed - reverting to PyTorch layers.")
                self.decoder = init_decoder_pytorch()
        else:                
            self.decoder = init_decoder_pytorch()
        
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

    def init_activations(self):
        self.scale_activation = torch.exp
        self.inv_scale_activation = torch.log
        self.rotation_activation = lambda x: torch.nn.functional.normalize(x, p=2, dim=-1)
    
    def randomize_grids(self, n_grids:int, n_features:int,
                 feat_grid_shape:List[int], n_dims:int):  
        with torch.no_grad():     
            d = "cpu"
            s = torch.ones([n_grids, n_dims], dtype=torch.float32, device = d)
            s -= torch.rand_like(s)*0.05
            s = self.inv_scale_activation(s)
            r = torch.zeros([n_grids, 4], dtype=torch.float32, device = d)
            r[:,-1] = 1.0
            t = torch.zeros([n_grids, n_dims], dtype=torch.float32, device = d)
            t += torch.rand_like(t)*0.1-0.05
            
        self._scales = torch.nn.Parameter(s,requires_grad=True)   
        self._rotations = torch.nn.Parameter(r,requires_grad=True)   
        self.translations = torch.nn.Parameter(t,requires_grad=True)   

        self.feature_grids =  torch.nn.parameter.Parameter(
            torch.ones(
                [n_grids, n_features] + feat_grid_shape,
                dtype=torch.float32
            ).uniform_(-0.0001, 0.0001),
            requires_grad=True
        )

    @property
    def scales(self):
        return self.scale_activation(self._scales)
    
    @property
    def rotations(self):
        return self.rotation_activation(self._rotations)

    @property
    def transformation_matrices(self):
        return create_transformation_matrices(self.rotations, self.scales, self.translations)

    def set_default_timestep(self, timestep:int):
        pass

    def get_default_timestep(self):
        return 0

    def prepare_timestep(self, timestep:int):
        pass

    def unload_timestep(self, timestep:int):
        self.to('cpu')

    def get_transform_parameters(self):
        return [{"params": self._rotations},
                {"params": self._scales},
                {"params": self.translations}]
    
    def get_model_parameters(self):
        return [
            {"params": [self.feature_grids]},
            {"params": self.decoder.parameters()}
        ]
    
    def get_volume_extents(self) -> List[int]:
        return self.full_shape
    
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
            self.decoder.apply(weights_init)   

    def check_grid_influence(self, data):
        delta_errors = []
        with torch.no_grad():
            # create a [-1, 1]^3 grid
            x = make_coord_grid([64, 64, 64], self.feature_grids.device, flatten=True, 
                        align_corners=True)
            # Get the global position for each grid's extents
            # [n_grids, 64*64*64, 3]
            x = self.inverse_transform(x)

            # Iterate over each grid to find the effect of removing the grid
            for i in range(self.n_grids):
                # Sample points within the grid
                points = x[i]
                # Find true data values
                vals = data.sample_values(points)
                # obtain feature values
                feats = encode(points, self.rotations, self.scales, self.translations, self.feature_grids)
                # find original network output (no change)
                y = self.decoder(feats).float() * (self.volume_max - self.volume_min) + self.volume_min
                # 0 the feature for this grid and see the output change
                feats[:,i] = 0  
                y_grid_removed = self.decoder(feats).float() * (self.volume_max - self.volume_min) + self.volume_min

                # compute error (RMSE) for both
                error_no_change = ((vals - y)**2).mean() ** 0.5
                error_grid_removed = ((vals - y_grid_removed)**2).mean() ** 0.5

                # keep the delta error. Higher means the grid has large influence. Lower means it is not as important.
                delta_error = error_grid_removed - error_no_change
                delta_errors.append(delta_error.item())
        # return the per-grid delta errors
        return delta_errors
    
    def transform(self, x : torch.Tensor) -> torch.Tensor:
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
        transformed_points = torch.matmul(self.transformation_matrices, 
                            x.transpose(0, 1)).transpose(1, 2)
        transformed_points = transformed_points[...,0:dims]
        
        return transformed_points
   
    def inverse_transform(self, x : torch.Tensor) -> torch.Tensor:
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

        local_to_global_matrices = torch.linalg.inv(self.transformation_matrices)
       
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
    
    def feature_density(self, x : torch.Tensor) -> torch.Tensor:
        return feature_density(x, self.rotations, self.scales, self.translations)  
    
    def grad_at(self, x : torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        y = self(x)

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y),]

        grad_x = torch.autograd.grad([y], [x],
            grad_outputs=grad_outputs)[0]
        return grad_x

    def min(self) -> torch.Tensor:
        return self.volume_min

    def max(self) -> torch.Tensor:
        return self.volume_max

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        feats = encode(x, self.rotations, self.scales, self.translations, self.feature_grids)  
        y = self.decoder(feats)
        y = y.float() * (self.volume_max - self.volume_min) + self.volume_min   
        return y

