from __future__ import absolute_import, division, print_function
import argparse
from AMGSRN.Datasets.datasets import Dataset
import datetime
from AMGSRN.Other.utility_functions import str2bool
from AMGSRN.Models.models import load_model, create_model, save_model
import torch
import torch.optim as optim
import time
import os
from AMGSRN.Models.options import *
from torch.utils.tensorboard import SummaryWriter
from AMGSRN.Models.losses import *
import shutil
from AMGSRN.Other.utility_functions import make_coord_grid, create_path
from AMGSRN.Other.vis_io import get_vts, write_pvd, write_vtm
from vtk import vtkMultiBlockDataSet
import numpy as np
from torch.utils.data import DataLoader
import glob
import matplotlib.pyplot as plt
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def log_to_writer(iteration, metrics, writer, opt):
    with torch.no_grad():   
        print_str = f"Iteration {iteration}/{opt['iterations']}, "
        for key in metrics.keys():
            if(metrics[key] is not None):    
                print_str = print_str + str(key) + f": {metrics[key] : 0.07f} " 
                writer.add_scalar(str(key), metrics[key], iteration)
        print(print_str)
        if("cuda" in opt['device']):
            GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
                / (1024**3))
            writer.add_scalar('GPU memory (GB)', GBytes, iteration)

def logging(writer, iteration, metrics, model, opt, grid_to_sample, dataset):
    if(opt['log_every'] > 0 and iteration % opt['log_every'] == 0):
        log_to_writer(iteration, metrics, writer, opt)
    if(opt['log_features_every'] > 0 and \
        iteration % opt['log_features_every'] == 0):
        log_feature_grids(model, dataset, opt, iteration)

def log_feature_grids(model, dataset, opt, iteration):
    with torch.no_grad():
        feat_grid_shape = [2, 2, 2]  # Fixed shape for simplicity
        
        global_points = make_coord_grid(feat_grid_shape, opt['device'], 
                        flatten=True, align_corners=True)
        transformed_points = model.inverse_transform(global_points)

        transformed_points += 1.0
        transformed_points *= 0.5 * (torch.tensor(opt['full_shape'], device=opt['device'])-1).flip(0)
        
        ids = torch.arange(transformed_points.shape[0], device=opt['device'])
        ids = ids.unsqueeze(1).unsqueeze(1)
        ids = ids.repeat([1, transformed_points.shape[1], 1])
        transformed_points = torch.cat((transformed_points, ids), dim=2)
        
        # use zyx point ordering for vtk files
        feat_grid_shape_zyx = np.flip(feat_grid_shape)

        # Create a single vtkMultiBlockDataSet to hold all grids
        vtm = vtkMultiBlockDataSet()
        vtm.SetNumberOfBlocks(opt['n_grids'])

        for i, grid in enumerate(transformed_points):
            grid_points = grid[:, :3].cpu().numpy()
            grid_ids = grid[:, -1].cpu().numpy()
            vts = get_vts(feat_grid_shape_zyx, grid_points, scalar_fields={"id": grid_ids})
            vtm.SetBlock(i, vts)

        # Write the vtkMultiBlockDataSet to a single .vtm file
        grid_dir = os.path.join(output_folder, "FeatureLocations", opt['save_name'], "ts_"+str(dataset.default_timestep))
        create_path(grid_dir)
        vtm_filename = f"grids_iter{iteration:05}.vtm"
        write_vtm(os.path.join(grid_dir, vtm_filename), vtm)

def combine_vtm_files(opt, t):
    grid_dir = os.path.join(output_folder, "FeatureLocations", opt['save_name'])
    vtm_files = sorted(glob.glob(os.path.join(grid_dir, f"ts_{t}", "grids_iter*.vtm")))
    
    # Create a PVD file to represent time-varying data
    pvd_filename = os.path.join(grid_dir, f"grids_over_iterations_{t}.pvd")
    
    # Extract iteration numbers from filenames to use as timesteps
    timesteps = [int(os.path.basename(f).split('iter')[1].split('.')[0]) for f in vtm_files]

    # Write the PVD file using the original VTM files
    write_pvd(vtm_files, pvd_filename, timesteps)

def train_step_APMGSRN(opt, iteration, batch, dataset, model, optimizer, scheduler, writer, scaler):
    prof_iter = -1
    
    if(iteration == prof_iter):
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        prof = profile(activities=activities, with_stack=True, record_shapes=True, profile_memory=True)
        prof = prof.__enter__()
    optimizer[0].zero_grad()                  
    x, y = batch
    
    x = x.to(opt['device'])
    y = y.to(opt['device'])
    
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt['use_amp']):
        with record_function("forward"):
            model_output = model.forward(x)

        with record_function("L_rec"):
            loss = F.mse_loss(model_output, y, reduction='none')
            loss = loss.sum(dim=1)

        if opt['tv_weight'] > 0:
            with record_function("L_tv"):
                # Calculate total variation loss for feature grids
                tv_loss = opt['tv_weight'] * (
                    torch.diff(model.feature_grids, dim=4).pow(2).mean() +
                    torch.diff(model.feature_grids, dim=3).pow(2).mean() +
                    torch.diff(model.feature_grids, dim=2).pow(2).mean() +
                    torch.diff(model.feature_grids, dim=0).pow(2).mean()
                )
        else:
            tv_loss = None

        if opt['model'] == "TVAMGSRN" and opt['grid_diff_weight'] > 0 and opt['save_grid_diffs'] and model.default_timestep > 0:
            grid_diff_loss = opt['grid_diff_weight'] * \
                torch.abs(model.models[model.default_timestep-1].feature_grids.detach() - \
                        model.feature_grids).mean() 
        else:
            grid_diff_loss = None
        
        if opt['l1_regularization'] > 0:
            with record_function("L1 reg"):
                l1_loss = opt['l1_regularization'] * model.feature_grids.abs().mean()
        else:
            l1_loss = None

    with record_function("L_rec backward"):
        scaler.scale(loss.mean()).backward()

    if tv_loss is not None:
        with record_function("L_tv backward"):
            scaler.scale(tv_loss.mean()).backward()

    if grid_diff_loss is not None:
        with record_function("L_grid_diff backward"):
            scaler.scale(grid_diff_loss.mean()).backward()

    if l1_loss is not None:
        with record_function("L1 backward"):
            scaler.scale(l1_loss.mean()).backward()

    if iteration > 500 and optimizer[1].param_groups[0]['lr'] > 1e-8:
        optimizer[1].zero_grad() 
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt['use_amp']):
            with record_function("density"):
                density = model.feature_density(x)
                density /= density.sum().detach()
            with record_function("target density"):
                target = torch.exp(torch.log(density+1e-16) * \
                    (loss.mean()/(loss+1e-16)))
                target /= target.sum()
            with record_function("density_loss"):
                density_loss = F.kl_div(
                    torch.log(density+1e-16), 
                    torch.log(target.detach()+1e-16), reduction='none', 
                    log_target=True)
                # density_loss = F.mse_loss(density, target.detach())


        with record_function("density_loss backward"):
            scaler.scale(density_loss.mean()).backward()
        
        with record_function("transform params step"):
            scaler.step(optimizer[1])
            scheduler[1].step()   

    else:
        density_loss = None

    with record_function("model params step"):
        scaler.step(optimizer[0])
        scaler.update()
        scheduler[0].step()   
    
    if opt['log_every'] != 0:
        logging(writer, iteration, 
            {"Fitting loss": loss.mean().detach().item(), 
            "Grid loss": density_loss.mean().detach().item() if density_loss is not None else None,
            "TV loss": tv_loss.mean().detach().item() if tv_loss is not None else None,
            "Grid diff loss": grid_diff_loss.mean().detach().item() if grid_diff_loss is not None else None,
            "L1 loss": l1_loss.mean().detach().item() if l1_loss is not None else None,
            "Learning rate": optimizer[0].param_groups[0]['lr']}, 
            model, opt, opt['full_shape'], dataset)
    
    if(iteration == prof_iter):
        # Print profiler results
        prof.__exit__(None, None, None)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=100))        
        # Optionally, you can save the profiler results to a file
        prof.export_chrome_trace(os.path.join(output_folder, opt['save_name']+"_trace.json"))
        print(f'Saved profile trace to {os.path.join(output_folder, opt["save_name"],"_trace.json")}')

def train_step_vanilla(opt, iteration, batch, dataset, model, optimizer, scheduler, writer, scaler):
    opt['iteration_number'] = iteration
    optimizer.zero_grad()
       
    x, y = batch
    x = x.to(opt['device'])
    y = y.to(opt['device'])
    
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt['use_amp']):
        model_output = model(x)
        loss = F.mse_loss(model_output, y, reduction='none')
    
    scaler.scale(loss.mean()).backward()                   

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()   
    
    logging(writer, iteration, 
        {"Fitting loss": loss.mean()}, 
        model, opt, opt['full_shape'], dataset)

def train_model(model, dataset, opt):
    model = model.to(opt['device'])
    
    print("Training on %s" % (opt["device"]), 
        os.path.join(save_folder, opt["save_name"]))
    

    if(opt['log_every'] > 0):
        writer = SummaryWriter(os.path.join('tensorboard', opt['save_name'], "ts_"+str(dataset.default_timestep)))
    else: 
        writer = None
    dataloader = DataLoader(dataset, 
                            batch_size=None, 
                            num_workers=4 if ("cpu" in opt['data_device'] and "cuda" in opt['device']) else 0,
                            pin_memory=True if ("cpu" in opt['data_device'] and "cuda" in opt['device']) else False,
                            pin_memory_device=opt['device'] if ("cpu" in opt['data_device'] and "cuda" in opt['device']) else "")
    
    model.train(True)

    # choose the specific training iteration function based on the model
    
    if("APMGSRN" in opt['model'] or "AMGSRN" in opt['model']):
        train_step = train_step_APMGSRN
        optimizer = [
            optim.Adam(
                model.get_model_parameters(), 
                lr=opt["lr"],
                betas=[opt['beta_1'], opt['beta_2']], 
                eps = 1e-15
                ),
            optim.Adam(
                model.get_transform_parameters(), 
                eps = 1e-15,
                #betas=[opt['beta_1'], opt['beta_2']], 
                #momentum=0.9
                )
        ]        
        scheduler = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=opt['iterations']),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[1], T_max=int(opt['iterations']/5))
        ] 
    else:
        train_step = train_step_vanilla
        optimizer = optim.Adam(model.parameters(), lr=opt["lr"], 
            betas=[opt['beta_1'], opt['beta_2']]) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt['iterations'])
        #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=opt['iterations'])
    
    scaler = torch.amp.GradScaler()
    
    start_time = time.time()
    for (iteration, batch) in enumerate(dataloader):
        train_step(opt,
                iteration,
                batch,
                dataset,
                model,
                optimizer,
                scheduler,
                writer,
                scaler)
    end_time = time.time()
    sec_passed = end_time-start_time
    mins = sec_passed / 60
    
    if writer is not None and ("APMGSRN" in opt['model'] or "AMGSRN" in opt['model']):
        transform_params = model.get_transform_parameters()
        all_params = torch.cat([p['params'].flatten() for p in transform_params])
        fig, ax = plt.subplots()
        ax.hist(all_params.detach().cpu().numpy(), bins=50)
        ax.set_title('Histogram of Transform Parameters')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Frequency')
        
        writer.add_figure('Transform Parameters Histogram', fig, global_step=iteration)
        plt.close(fig)
    #print(f"Model completed training after {int(mins)}m {sec_passed%60:0.02f}s")

    
    #writer.add_graph(model, torch.zeros([1, 3], device=opt['device'], dtype=torch.float32))
    if(opt['log_every'] > 0):
        writer.close()

def train(model, dataset, opt):
    if(os.path.exists(os.path.join(project_folder_path, "tensorboard", opt['save_name']))):
        shutil.rmtree(os.path.join(project_folder_path, "tensorboard", opt['save_name']))
    
    if(os.path.exists(os.path.join(output_folder, "FeatureLocations", opt['save_name']))):
        shutil.rmtree(os.path.join(output_folder, "FeatureLocations", opt['save_name']))

    for t in range(dataset.n_timesteps):
        model.set_default_timestep(t)
        dataset.set_default_timestep(t)
        model.prepare_timestep(t)
        dataset.load_timestep(t)
        train_model(model, dataset, opt)
        dataset.unload_timestep(t)
        
        opt['iteration_number'] = 0
        # Create histogram of transform parameters
        if(opt['log_features_every'] > 0):
            combine_vtm_files(opt, t)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains an implicit model on data.')

    parser.add_argument('--n_dims',default=None,type=int,
        help='Number of dimensions in the data')
    parser.add_argument('--n_outputs',default=None,type=int,
        help='Number of output channels for the data (ex. 1 for scalar field, 3 for image or vector field)')
    parser.add_argument('--feature_grid_shape',default=None,type=str,
        help='Resolution for feature grid')
    parser.add_argument('--n_features',default=None,type=int,
        help='Number of features in the feature grid')       
    parser.add_argument('--n_grids',default=None,type=int,
        help='Number of grids for APMGSRN')
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int,
        help='Number of positional encoding terms')   
    parser.add_argument('--extents',default=None,type=str,
        help='Spatial extents to use for this model from the data')   
    parser.add_argument('--bias',default=None,type=str2bool,
        help='Use bias in linear layers or not')
    parser.add_argument('--use_global_position',default=None,type=str2bool,
        help='For the fourier featuers, whether to use the global position or local.')
    parser.add_argument('--use_tcnn_if_available',default=None,type=str2bool,
        help='Whether to use TCNN if available on the machine training.')
    
    # Hash Grid (NGP model) hyperparameters
    parser.add_argument('--hash_log2_size',default=None,type=int,
        help='Size of hash table')
    parser.add_argument('--hash_base_resolution',default=None,type=int,
        help='Minimum resolution of a single dimension')
    parser.add_argument('--hash_max_resolution',default=None,type=int,
        help='Maximum resolution of a single dimension') 
    

    parser.add_argument('--data',default=None,type=str,
        help='Data file name')
    parser.add_argument('--model',default=None,type=str,
        help='The model architecture to use')
    parser.add_argument('--grid_initialization',default=None,type=str,
        help='How to initialize APMGSRN grids. choices: default, large, small')
    parser.add_argument('--save_name',default=None,type=str,
        help='Save name for the model')
    parser.add_argument('--align_corners',default=None,type=str2bool,
        help='Aligns corners in implicit model.')      
    parser.add_argument('--n_layers',default=None,type=int,
        help='Number of layers in the model')
    parser.add_argument('--nodes_per_layer',default=None,type=int,
        help='Nodes per layer in the model')    
    parser.add_argument('--interpolate',default=None,type=str2bool,
        help='Whether or not to use interpolation during training')  
    parser.add_argument('--requires_padded_feats',default=None,type=str2bool,
        help='Pads features to next multiple of 16 for TCNN.')      
    parser.add_argument('--grid_index',default=None,type=str,
        help='Index for this network in an ensemble of networks')      
    
    
    parser.add_argument('--iters_to_train_new_layer',default=None,type=int,
        help='Number of iterations to fine tune a new layer')    
    parser.add_argument('--iters_since_new_layer',default=None,type=int,
        help='To track the number of iterations since a new layer was added')    
    
    
    parser.add_argument('--device',default=None,type=str,
        help='Which device to train on')
    parser.add_argument('--data_device',default=None,type=str,
        help='Which device to keep the data on')

    parser.add_argument('--iterations',default=None, type=int,
        help='Number of iterations to train')
    parser.add_argument('--points_per_iteration',default=None, type=int,
        help='Number of points to sample per training loop update')
    parser.add_argument('--lr',default=None, type=float,
        help='Learning rate for the adam optimizer')
    parser.add_argument('--transform_lr',default=None, type=float,
        help='Learning rate for the transform parameters')
    parser.add_argument('--tv_weight',default=None, type=float,
        help='Weight for the TV loss')
    parser.add_argument('--grid_diff_weight',default=None, type=float,
        help='Weight for the grid diff loss')
    parser.add_argument('--l1_regularization',default=None, type=float,
        help='Weight for the L1 regularization')
    parser.add_argument('--beta_1',default=None, type=float,
        help='Beta1 for the adam optimizer')
    parser.add_argument('--beta_2',default=None, type=float,
        help='Beta2 for the adam optimizer')

    parser.add_argument('--iteration_number',default=None, type=int,
        help="Not used.")
    parser.add_argument('--save_every',default=None, type=int,
        help='How often to save the model')
    parser.add_argument('--log_every',default=None, type=int,
        help='How often to log the loss')
    parser.add_argument('--log_features_every',default=None, type=int,
        help='How often to log the feature positions')
    parser.add_argument('--log_image_every',default=None, type=int,
        help='How often to log the image')
    parser.add_argument('--load_from',default=None, type=str,
        help='Model to load to start training from')
    parser.add_argument('--log_image',default=None, type=str2bool,
        help='Whether or not to log an image. Slows down training.')
    parser.add_argument('--use_amp',default=None, type=str2bool,
        help='Whether or not to use automatic mixed precision.')
    parser.add_argument('--profile',default=None, type=str2bool,
        help='Whether or not to profile the training.')
    parser.add_argument('--save_with_compression',default=None, type=str2bool,
        help='Use compression?')
    parser.add_argument('--save_with_compression_level',default=None, type=float,
        help='Compression level for SZ3, ABS mode only.')
    parser.add_argument('--save_grid_diffs',default=None, type=str2bool,
        help='Save the difference between grids to improve compression.')
    parser.add_argument('--last_timestep_init',default=None, type=str2bool,
        help='Initialize the last timestep with the previous timestep.')
    parser.add_argument('--error_volume',default=None, type=str2bool,
        help='The volume being traind is an error volume.')


    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    if(args['load_from'] is None):
        # Init models
        model = None
        opt = Options.get_default()

        # Read arguments and update our options
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]

        dataset = Dataset(opt)
        opt['data_min'] = dataset.min().item()
        opt['data_max'] = dataset.max().item()
        
        #opt['data_min'] = max(dataset.min(), dataset.data.mean() - dataset.data.std()*3).item()
        #opt['data_max'] = min(dataset.max(), dataset.data.mean() + dataset.data.std()*3).item()
        #opt['data_min'] = dataset.data.mean().item()
        #opt['data_max'] = max(dataset.data.mean()-dataset.data.min(), dataset.data.max() -dataset.data.mean()).item()
        model = create_model(opt)
        model = model.to(opt['device'])
    else:        
        opt = load_options(os.path.join(save_folder, args["load_from"]))
        opt["device"] = args["device"]
        opt["save_name"] = args["load_from"]
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
        dataset = Dataset(opt)
        model = load_model(opt, opt['device'])

    now = datetime.datetime.now()
    start_time = time.time()
    
    train(model, dataset, opt)
    del dataset
    save_model(model, opt)
    

