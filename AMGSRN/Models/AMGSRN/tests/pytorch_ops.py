import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False

def composite_transformation(rotations:torch.Tensor, scales:torch.Tensor, translations:torch.Tensor, 
                             transformation_matrix:torch.Tensor=None) -> torch.Tensor:
    if(transformation_matrix is None):
        transformation_matrix = torch.zeros([rotations.shape[0], 4, 4],
                    device='cuda', dtype=torch.float32)
        
    # grab useful quaternion values for later
    x, y, z, w = rotations.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # Create the transformation matrix
    transformation_matrix[:, 0, 0] = scales[:, 0] * (1. - 2. * (yy + zz))     #0
    transformation_matrix[:, 0, 1] = scales[:, 1] * (2. * (xy - wz))         #1
    transformation_matrix[:, 0, 2] = scales[:, 2] * (2. * (xz + wy))         #2
    transformation_matrix[:, 1, 0] = scales[:, 0] * (2. * (xy + wz))         #4
    transformation_matrix[:, 1, 1] = scales[:, 1] * (1. - 2. * (xx + zz))     #5
    transformation_matrix[:, 1, 2] = scales[:, 2] * (2. * (yz - wx))         #6
    transformation_matrix[:, 2, 0] = scales[:, 0] * (2. * (xz - wy))         #8
    transformation_matrix[:, 2, 1] = scales[:, 1] * (2. * (yz + wx))         #9
    transformation_matrix[:, 2, 2] = scales[:, 2] * (1. - 2. * (xx + yy))     #10
    transformation_matrix[:, :3, 3] = translations
    transformation_matrix[:, 3, 3] = 1.0
    return transformation_matrix

def transform_points(transformation_matrices:torch.Tensor, points:torch.Tensor) -> torch.Tensor:
    batch : int = points.shape[0]
    dims : int = points.shape[1]
    ones = torch.ones([batch, 1], 
        device=points.device,
        dtype=torch.float32)
        
    x = torch.cat([points, ones], dim=1)
    # [n_transforms, 4, 4] x [4, N]
    transformed_points = torch.matmul(transformation_matrices, 
                        x.transpose(0, 1)).transpose(1, 2)
    transformed_points = transformed_points[...,0:dims]
    
    # return [n_grids,batch,n_dims]
    return transformed_points

def encode_pytorch(feature_grids:torch.Tensor, rotations:torch.Tensor, scales:torch.Tensor, 
                   translations:torch.Tensor, query_positions:torch.Tensor) -> torch.tensor:

    matrices = composite_transformation(rotations, scales, translations)
    x = transform_points(matrices, query_positions)
    grids : int = x.shape[0]
    batch : int = x.shape[1]
    dims : int = x.shape[2]        
    x = x.reshape(grids, 1, 1, batch, dims)
    
    
    # Sample the grids at the batch of transformed point locations
    # Uses zero padding, so any point outside of [-1,1]^n_dims will be a 0 feature vector
    feats = F.grid_sample(feature_grids, x,
        mode='bilinear', align_corners=True,
        padding_mode="zeros").flatten(0, dims).permute(1,0)
    
    return feats

def feature_density_pytorch(query_points:torch.Tensor, rotations:torch.Tensor, scales:torch.Tensor, translations:torch.Tensor) -> torch.Tensor:
    mats = composite_transformation(rotations, scales, translations)
    transformed_points = transform_points(mats, query_points)
    
    # get the coeffs of shape [n_grids], then unsqueeze to [1,n_grids] for broadcasting
    coeffs = torch.linalg.det(mats[:,0:-1,0:-1])[None,...]
    
    # sum the exp part to [batch,n_grids]
    exps = torch.exp(-1 * \
        torch.sum(
            transformed_points.transpose(0,1)**20, 
        dim=-1))
    
    result = torch.sum(coeffs * exps, dim=-1, keepdim=True)
    return result  

