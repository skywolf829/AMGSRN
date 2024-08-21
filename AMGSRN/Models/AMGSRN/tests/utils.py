import torch
import numpy as np

def random_quaternion(batch_size=1, device="cuda", requires_grad=True):
    torch.random.manual_seed(0)
    np.random.seed(0)
    u = np.random.rand(batch_size)
    v = np.random.rand(batch_size)
    w = np.random.rand(batch_size)
    r = torch.tensor(np.array([((1-u)**0.5)*np.sin(2*np.pi*v),
                        ((1-u)**0.5)*np.cos(2*np.pi*v),
                        (u**0.5)*np.sin(2*np.pi*w),
                        (u**0.5)*np.cos(2*np.pi*w)]).T, 
                        dtype=torch.float32, device=device)
    
    r = torch.nn.functional.normalize(r)
    r.requires_grad_(requires_grad)
    return r

def get_random_rst(batch_size=1, device="cuda", requires_grad=True):
    torch.random.manual_seed(0)
    np.random.seed(0)
    r = random_quaternion(batch_size=batch_size, device=device)
    s = torch.rand([batch_size,3], device=device, dtype=torch.float32) + 0.1
    s.requires_grad_(requires_grad)
    t = torch.rand([batch_size,3], device=device, dtype=torch.float32, requires_grad=requires_grad)
    return r,s,t

def get_random_feature_grids(num_grids:int, num_features:int, dims:list[int], 
        device:str='cuda', requires_grad:bool=True) -> torch.Tensor:
    torch.random.manual_seed(0)
    np.random.seed(0)
    feats = torch.rand([num_grids, num_features, *dims], device=device, dtype=torch.float32)
    feats *= 2
    feats -= 1
    feats.requires_grad_(requires_grad)
    return feats

def get_random_points(batch_size:int=1, device='cuda'):
    torch.random.manual_seed(0)
    np.random.seed(0)
    points = torch.rand([batch_size, 3], device=device, dtype=torch.float32)*2-1
    return points