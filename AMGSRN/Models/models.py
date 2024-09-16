from __future__ import absolute_import, division, print_function
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import os
from math import pi
from Models.options import *
from Other.utility_functions import create_folder, make_coord_grid
import math
import zlib
import zipfile
import base64
import io
import numpy as np
import subprocess

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def next_highest_multiple(num:int, base:int):
    return base*int(math.ceil(max(1, num/base)))

def convert_tcnn_to_pytorch(ckpt, opt):
    base = 16
    requires_padding = False
    if(opt['model'] == "fVSRN"):
        input_dim = opt['n_features']+opt['num_positional_encoding_terms']*opt['n_dims']*2
        input_dim_padded = next_highest_multiple(input_dim, base)
        if(input_dim != input_dim_padded):
            requires_padding = True
        
        output_dim = opt['n_outputs']
        output_dim_padded = next_highest_multiple(output_dim, base)
        
    elif(opt['model'] == "APMGSRN" or opt['model'] == "AMGSRN"):
        input_dim = opt['n_features']*opt['n_grids']
        input_dim_padded = next_highest_multiple(input_dim, base)
        if(input_dim != input_dim_padded):
            requires_padding = True
        
        output_dim = opt['n_outputs']
        output_dim_padded = next_highest_multiple(output_dim, base)
    
    else:
        #print(f"Currently we do not support converting model type {opt['model']} to pure PyTorch.")
        quit()
    opt['requires_padded_feats'] = requires_padding  
        
    layer_weight_shape = []
    
    first_layer_shape = [opt['nodes_per_layer'], input_dim_padded]
    layer_weight_shape.append(first_layer_shape)
    
    for i in range(opt['n_layers']-1):
        layer_shape = [opt['nodes_per_layer'], opt['nodes_per_layer']]
        layer_weight_shape.append(layer_shape)
        
    last_layer_shape = [output_dim_padded,opt['nodes_per_layer']]
    layer_weight_shape.append(last_layer_shape)
    
    weights = ckpt['state_dict']['decoder.params']
    new_weights = []
    current_weight_index = 0
    for i in range(len(layer_weight_shape)):
        #print(f"Layer {i}: {layer_weight_shape[i]}")
        
        weight_shape_this_layer = layer_weight_shape[i]        
        num_weights_this_layer = weight_shape_this_layer[0]*weight_shape_this_layer[1]
        
        if(current_weight_index+num_weights_this_layer > weights.shape[0]):
            #print(f"Number of expected weights {current_weight_index+num_weights_this_layer} is larger than the number of weights saved {weights.shape}.")
            quit()
        this_layer_weights = weights[current_weight_index:current_weight_index+num_weights_this_layer].clone()
        this_layer_weights = this_layer_weights.reshape(weight_shape_this_layer)
        new_weights.append(this_layer_weights)
        current_weight_index += num_weights_this_layer
        
    del ckpt['state_dict']['decoder.params']
    for i in range(len(new_weights)):
        if(i == len(new_weights) - 1):
            # In the last layer, we can actually prune the unused weights when
            # moving back to PyTorch
            name = f"decoder.{i}.weight"
            ckpt['state_dict'][name] = new_weights[i][0:1]
        else:
            # The first layer is not pruned, even if it was padded, as TCNN learned
            # to use those padded weights as a bias essentially
            # All other layers are not changed            
            name = f"decoder.{i}.linear.weight"
            ckpt['state_dict'][name] = new_weights[i]
    
    return ckpt

def save_model(model,opt):
    with torch.no_grad():
        folder = create_folder(save_folder, opt["save_name"])
        path_to_save = os.path.join(save_folder, folder)
        
        if(opt['model'] == "TVAMGSRN" or not opt['save_with_compression']):
            state_dict = model.state_dict()
            torch.save({'state_dict': state_dict}, 
                os.path.join(path_to_save, "model.ckpt")
            )
        else: 
            state_dict = model.state_dict()
            # Create a temporary directory for compressed files
            temp_dir = os.path.join(path_to_save, "temp_compressed")
            os.makedirs(temp_dir, exist_ok=True)

            try:
                # Attempt to compress feature grids using SZ3
                feature_grids = state_dict['feature_grids'].cpu().numpy().astype(np.float32)
                for i in range(feature_grids.shape[1]): # iterate over each channel
                    grid = feature_grids[:, i]
                    grid_path = os.path.join(temp_dir, f"feature_grid_{i}.raw")
                    grid.tofile(grid_path)
                    
                    dims = ' '.join(str(d) for d in grid.shape)
                    dim_flag = f"-{len(grid.shape)} {dims}"
                    
                    compressed_path = os.path.join(temp_dir, f"feature_grid_{i}.sz")
                    command = f"sz3 -f -z {compressed_path} -i {grid_path} -M ABS {opt['save_with_compression_level']} {dim_flag}"
                    subprocess.run(command, check=True)
                    
                    # Remove the raw file
                    os.remove(grid_path)

                # Save the rest of the model losslessly
                for key, tensor in state_dict.items():
                    if key != 'feature_grids':
                        np.save(os.path.join(temp_dir, f"{key}.npy"), tensor.cpu().numpy())
                opt['compressor_used'] = "sz3"

            except Exception as e:
                print(f"Error during compression: {str(e)}. Saving all data losslessly.")
                # If compression fails, save everything losslessly
                for key, tensor in state_dict.items():
                    np.save(os.path.join(temp_dir, f"{key}.npy"), tensor.cpu().numpy())
                opt['compressor_used'] = "none"

            # Create a zip file containing all compressed and losslessly saved arrays
            with zipfile.ZipFile(os.path.join(path_to_save, "compressed_model.zip"), 'w') as zipf:
                for filename in os.listdir(temp_dir):
                    zipf.write(os.path.join(temp_dir, filename), filename)

            # Clean up temporary directory
            for filename in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, filename))
            os.rmdir(temp_dir)
        
        save_options(opt, path_to_save)

        # Saving javascript readable model
        # with zipfile.ZipFile(os.path.join(path_to_save, "web_model.zip"), 'w') as zipf:
        #     for key, tensor in state_dict.items():
        #         with io.BytesIO() as array_buffer:
        #             arr = tensor.cpu().numpy().astype(np.float32)
        #             #print(key)
        #             #print(arr)
        #             array_buffer.write(arr.tobytes())
        #             zipf.writestr(key+'.bin', array_buffer.getvalue())
        #     with io.StringIO() as json_buffer:
        #         json.dump(opt, json_buffer, sort_keys=True, indent=4)
        #         zipf.writestr('options.json', json_buffer.getvalue())

def load_model(opt, device):
    path_to_load = os.path.join(save_folder, opt["save_name"])
    
    try:
        import tinycudann
        tcnn_installed = True
    except ImportError:
        tcnn_installed = False

    if(not opt['ensemble']):
        model = create_model(opt)
        if opt['model'] == "TVAMGSRN" or not opt['save_with_compression']:
            model.load_state_dict(torch.load(os.path.join(path_to_load, "model.ckpt"))['state_dict'])
        else:
            # Load the compressed model
            with zipfile.ZipFile(os.path.join(path_to_load, "compressed_model.zip"), 'r') as zipf:
                state_dict = {}
                for filename in zipf.namelist():
                    if filename.endswith('.npy'):
                        with zipf.open(filename) as f:
                            state_dict[filename[:-4]] = torch.from_numpy(np.load(f))
                    elif filename.endswith('.sz'):
                        # Decompress SZ3 files if used
                        if opt['compressor_used'] == "sz3":
                            with zipf.open(filename) as f:
                                temp_file = os.path.abspath(os.path.join(path_to_load, filename))
                                with open(temp_file, 'wb') as tf:
                                    tf.write(f.read())
                            
                            # Decompress using SZ3
                            decompressed_file = temp_file + '.out'
                            dims = "-4 " + str(opt['n_grids']) + " " + ' '.join(str(d) for d in opt['feature_grid_shape'].split(","))
                            command = f"sz3 -x -f -z {temp_file} -o {decompressed_file} {dims}"
                            print(command)
                            try:
                                subprocess.run(command, check=True)
                            except subprocess.CalledProcessError as e:
                                print(f"Error during decompression: {str(e)}. Make sure SZ3 is installed.")
                                quit(4)
                            
                            # Load decompressed data
                            decompressed_data = np.fromfile(decompressed_file, dtype=np.float32).reshape(
                                [opt['n_grids'], 
                                *[int(i) for i in opt['feature_grid_shape'].split(",")]]
                            )
                            print(decompressed_data.shape)
                            os.remove(temp_file)
                            os.remove(decompressed_file)
                            
                            # Reshape and add to state_dict
                            grid_num = int(filename.split('_')[-1].split('.')[0])
                            if 'feature_grids' not in state_dict:
                                state_dict['feature_grids'] = torch.zeros(
                                    [opt['n_grids'], opt['n_features']] + 
                                    [int(i) for i in opt['feature_grid_shape'].split(",")], 
                                    dtype=torch.float32)
                            state_dict['feature_grids'][:, grid_num] = torch.from_numpy(decompressed_data)
            
            # Load the state dict
            model.load_state_dict(state_dict)
    else:
        model = create_model(opt)
    return model

def create_model(opt):

    if(opt['ensemble']):
        from Models.ensemble_SRN import Ensemble_SRN
        return Ensemble_SRN(opt)
    else:
        if(opt['model'] == "fVSRN"):
            from Models.fVSRN import fVSRN_NGP
            return fVSRN_NGP(opt)
        elif(opt['model'] == "APMGSRN" or opt['model'] == "AMGSRN" \
             or opt['model'] == "APMGSRN_cuda" or opt['model'] == "AMGSRN_cuda"):
            from Models.AMGSRN import AMGSRN
            return AMGSRN(opt)
        elif(opt['model'] == "APMGSRN_pytorch" or \
             opt['model'] == "AMGSRN_pytorch" or \
             opt['model'] == "AMGSRN_old" or \
             opt['model'] == "APMGSRN_old"):
            from AMGSRN.Models.AMGSRN_pytorch import AMGSRN_old
            return AMGSRN_old(opt)
        elif(opt['model'] == "NGP"):
            from Models.NGP import NGP
            return NGP(opt)
        elif(opt['model'] == "TVAMGSRN"):
            from Models.TVAMGSRN import TVAMGSRN
            return TVAMGSRN(opt)
   
def sample_grid(model, grid, align_corners:bool = False,
                device:str="cuda", data_device:str="cuda", max_points:int = 100000):
    coord_grid = make_coord_grid(grid, 
        data_device, flatten=False,
        align_corners=align_corners)
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    vals = forward_maxpoints(model, coord_grid, 
                             max_points = max_points,
                             data_device=data_device,
                             device=device
                             )
    coord_grid_shape[-1] = -1
    vals = vals.reshape(coord_grid_shape)
    return vals

def forward_maxpoints(model, coords, out_dim=1, max_points=100000, 
                      data_device="cuda", device="cuda"):
    output_shape = list(coords.shape)
    output_shape[-1] = out_dim
    output = torch.empty(output_shape, 
        dtype=torch.float32, 
        device=data_device)
    
    for start in range(0, coords.shape[0], max_points):
        output[start:min(start+max_points, coords.shape[0])] = \
            model(coords[start:min(start+max_points, coords.shape[0])].to(device)).to(data_device)
    return output

