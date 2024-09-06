import os
import torch
from Other.utility_functions import make_coord_grid, nc_to_tensor, curl
import torch.nn.functional as F
import time

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.full_coord_grid = None
        
        self.file_names = []
        self.loaded_timesteps = []
        self.data = {}
        self.default_timestep = 0
        # check if the given data is a folder or a file
        if os.path.isdir(os.path.join(data_folder, self.opt['data'])):
            # iterate over all timesteps (entries in)
            self.file_names = sorted(os.listdir(os.path.join(data_folder, self.opt['data'])))    
            # remove anything that isn't a .h5 file
            self.file_names = [os.path.join(data_folder, self.opt['data'], f) for f in self.file_names if f.endswith('.h5')]
            self.n_timesteps = len(self.file_names)
            d, full_shape = nc_to_tensor(self.file_names[0], self.opt)
            d = d.to(self.opt['data_device'])
            self.data[0] = d
            opt['full_shape'] = d.shape[2:]
            opt['n_timesteps'] = len(self.file_names)
            self.loaded_timesteps.append(0)
        else:
            # load file
            self.file_names = [os.path.join(data_folder, self.opt['data'])]
            self.n_timesteps = 1
            d, full_shape = nc_to_tensor(self.file_names[0], self.opt)
            d = d.to(self.opt['data_device'])
            self.data[0] = d
            opt['full_shape'] = d.shape[2:]
            opt['n_timesteps'] = 1
            self.loaded_timesteps.append(0)

    def set_default_timestep(self, t:int):
        self.default_timestep = t

    def load_timestep(self, t):
        if t in self.loaded_timesteps:
            return
        folder_to_load = os.path.join(data_folder, self.opt['data'], self.file_names[t])
        d, full_shape = nc_to_tensor(folder_to_load, self.opt)
        d = d.to(self.opt['data_device'])
        self.data[t] = d
        self.loaded_timesteps.append(t)

    def unload_timestep(self, t):
        if t not in self.loaded_timesteps:
            return
        del self.data[t]
        self.loaded_timesteps.remove(t)

    def unload_all_timesteps(self):
        for t in self.loaded_timesteps:
            del self.data[t]
        torch.cuda.empty_cache()

    def min(self):
        if self.min_ is not None:
            return self.min_
        else:
            self.min_ = float('inf')
            for t in range(len(self.file_names)):
                self.load_timestep(t)
                self.min_ = min(self.min_, self.data[t].min())
                self.unload_timestep(t)
            return self.min_
        
    def mean(self):
        if self.mean_ is not None:
            return self.mean_
        else:
            self.mean_ = 0
            for t in range(len(self.file_names)):
                self.load_timestep(t)
                self.mean_ += self.data[t].mean()
                self.unload_timestep(t)
            self.mean_ /= len(self.file_names)
            return self.mean_
        
    def max(self):
        if self.max_ is not None:
            return self.max_
        else:
            self.max_ = float('-inf')
            for t in range(len(self.file_names)):
                self.load_timestep(t)
                self.max_ = max(self.max_, self.data[t].max())
                self.unload_timestep(t)
            return self.max_

    def get_full_coord_grid(self):
        if self.full_coord_grid is None:
            self.full_coord_grid = make_coord_grid(self.data.shape[2:], 
                    self.opt['data_device'], flatten=True, 
                    align_corners=self.opt['align_corners'])
        return self.full_coord_grid
    
    def sample_values(self, points, t:int=None):
        x = points[None,None,None,...].to(self.opt['data_device'])
        y = F.grid_sample(self.data[t],
            x, mode='bilinear', 
            align_corners=self.opt['align_corners'])
        
        x = x.squeeze()
        y = y.squeeze()
        if(len(y.shape) == 1):
            y = y.unsqueeze(0)    
        
        y = y.permute(1,0)
        return y
    
    def get_random_points(self, n_points, t:int=None):
        if t is None:
            t = self.default_timestep
        self.load_timestep(t)
        x = torch.rand([n_points, self.opt['n_dims']], 
                device=self.opt['data_device']) * 2 - 1
        y = self.sample_values(x, t)
        return x, y

    def __len__(self):
        return self.opt['iterations']
    
    def __getitem__(self, idx, t:int=None):
        return self.get_random_points(
            self.opt['points_per_iteration'],
            t)