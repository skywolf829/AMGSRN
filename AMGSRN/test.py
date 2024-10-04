from __future__ import absolute_import, division, print_function
import argparse
import os
from AMGSRN.Other.utility_functions import PSNR, tensor_to_cdf, create_path, make_coord_grid
from AMGSRN.Models.models import load_model, sample_grid, forward_maxpoints, save_model
from AMGSRN.Models.options import load_options
from AMGSRN.Datasets.datasets import Dataset
import torch
import numpy as np
import torch.nn.functional as F
import time
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import subprocess
from matplotlib import pyplot as plt

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def model_reconstruction(model, opt):
    
    # Load the reference data
    with torch.no_grad():
        result = sample_grid(model, opt['full_shape'], max_points=1000000,
                             align_corners=opt['align_corners'],
                             device=opt['device'],
                             data_device=opt['data_device'])
    result = result.to(opt['data_device'])
    result = result.permute(3, 0, 1, 2).unsqueeze(0)
    create_path(os.path.join(output_folder, "Reconstruction"))
    tensor_to_cdf(result, 
        os.path.join(output_folder, 
        "Reconstruction", opt['save_name']+".nc"))

def model_reconstruction_chunked(model, opt, timestep):
    
    chunk_size = 512
    full_shape = opt['full_shape']
    
    output = torch.empty(opt['full_shape'], 
        dtype=torch.float32, 
        device=opt['data_device']).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        for z_ind in range(0, full_shape[0], chunk_size):
            z_ind_end = min(full_shape[0], z_ind+chunk_size)
            z_range = z_ind_end-z_ind
            for y_ind in range(0, full_shape[1], chunk_size):
                y_ind_end = min(full_shape[1], y_ind+chunk_size)
                y_range = y_ind_end-y_ind            
                for x_ind in range(0, full_shape[2], chunk_size):
                    x_ind_end = min(full_shape[2], x_ind+chunk_size)
                    x_range = x_ind_end-x_ind
                    
                    opt['extents'] = f"{z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}"
                    #print(f"Extents: {z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}")
                                                                
                    grid = [z_range, y_range, x_range]
                    coord_grid = make_coord_grid(grid, 
                        opt['data_device'], flatten=True,
                        align_corners=opt['align_corners'],
                        use_half=False)
                    
                    coord_grid += 1.0
                    coord_grid /= 2.0
                    
                    coord_grid[:,0] *= (x_range-1) / (full_shape[2]-1)
                    coord_grid[:,1] *= (y_range-1) / (full_shape[1]-1)
                    coord_grid[:,2] *= (z_range-1) / (full_shape[0]-1)
                    
                    coord_grid[:,0] += x_ind / (full_shape[2]-1)
                    coord_grid[:,1] += y_ind / (full_shape[1]-1)
                    coord_grid[:,2] += z_ind / (full_shape[0]-1)
                    
                    coord_grid *= 2.0
                    coord_grid -= 1.0
                    
                    out_tmp = forward_maxpoints(model, 
                                                coord_grid, max_points=2**20, 
                                                data_device=opt['data_device'],
                                                device=opt['device'])
                    out_tmp = out_tmp.permute(1,0)
                    out_tmp = out_tmp.view([out_tmp.shape[0]] + grid)
                    output[0,:,z_ind:z_ind_end,y_ind:y_ind_end,x_ind:x_ind_end] = out_tmp

                    print(f"Chunk {z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}")
    return output

def test_psnr(model, dataset, opt):
    
    # Load the reference data
    data = dataset.data[dataset.default_timestep]
    
    grid = list(data.shape[2:])
    
    data = data[0].flatten(1,-1).permute(1,0)
    data_max = data.max()
    data_min = data.min()
        
    with torch.no_grad():
        coord_grid = make_coord_grid(grid, 
        opt['data_device'], flatten=True,
        align_corners=opt['align_corners'],
        use_half=False)
        
        for start in range(0, coord_grid.shape[0], 2**20):
            end_ind = min(coord_grid.shape[0], start+2**20)
            output = model(coord_grid[start:end_ind].to(opt['device']).float()).to(opt['data_device'])
            data[start:end_ind] -= output
        
        data **= 2
        SSE : torch.Tensor = data.sum()
        MSE = SSE / data.numel()
        y = 10*torch.log10(MSE)
        y = 20.0 * torch.log10(data_max-data_min) - y
    
    #print(f"PSNR: {y : 0.03f}")
    return y, SSE, MSE, data.numel()

def test_grid_influence(model, opt):
    data = Dataset(opt)
    torch.cuda.synchronize()
    t0 = time.time()
    grid_influences = model.check_grid_influence(data)
    torch.cuda.synchronize()
    t1 = time.time()
    print("Grid influences on MSE. Higher means the grid has more positive impact on quality.")
    print(f"Test took {t1-t0:0.02f} seconds")
    print(grid_influences)

    worst = np.argmin(grid_influences)
    print(f"Weakest influence - grid {worst}: {grid_influences[worst] :0.08f} RMSE")

def test_psnr_chunked(model, opt):
    
    data_max = None
    data_min = None
    
    SSE = torch.tensor([0.0], dtype=torch.float32, device=opt['data_device'])
    
    chunk_size = 768
    full_shape = opt['full_shape']
    init_extents = opt['extents']
    with torch.no_grad():
        for z_ind in range(0, full_shape[0], chunk_size):
            z_ind_end = min(full_shape[0], z_ind+chunk_size)
            z_range = z_ind_end-z_ind
            for y_ind in range(0, full_shape[1], chunk_size):
                y_ind_end = min(full_shape[1], y_ind+chunk_size)
                y_range = y_ind_end-y_ind            
                for x_ind in range(0, full_shape[2], chunk_size):
                    x_ind_end = min(full_shape[2], x_ind+chunk_size)
                    x_range = x_ind_end-x_ind
                    
                    opt['extents'] = f"{z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}"
                    #print(f"Extents: {z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}")
                    dataset = Dataset(opt)
                    dataset.set_default_timestep(model.get_default_timestep())
                    dataset.load_timestep(model.get_default_timestep())
                    data = dataset.data[dataset.default_timestep]
                    data = data[0].flatten(1,-1).permute(1,0)
                    
                    if(data_max is None):
                        data_max = data.max()
                    else:
                        data_max = max(data.max(), data_max)
                    if(data_min is None):
                        data_min = data.min()
                    else:
                        data_min = min(data.min(), data_min)
                        
                    grid = [z_range, y_range, x_range]
                    coord_grid = make_coord_grid(grid, 
                        opt['data_device'], flatten=True,
                        align_corners=opt['align_corners'],
                        use_half=False)
                    
                    coord_grid += 1.0
                    coord_grid /= 2.0
                    
                    coord_grid[:,0] *= (x_range-1) / (full_shape[2]-1)
                    coord_grid[:,1] *= (y_range-1) / (full_shape[1]-1)
                    coord_grid[:,2] *= (z_range-1) / (full_shape[0]-1)
                    
                    coord_grid[:,0] += x_ind / (full_shape[2]-1)
                    coord_grid[:,1] += y_ind / (full_shape[1]-1)
                    coord_grid[:,2] += z_ind / (full_shape[0]-1)
                    
                    coord_grid *= 2.0
                    coord_grid -= 1.0
                    
                    for start in range(0, coord_grid.shape[0], 2**20):
                        end_ind = min(coord_grid.shape[0], start+2**20)
                        output = model(coord_grid[start:end_ind].to(opt['device']).float()).to(opt['data_device'])
                        data[start:end_ind] -= output
                        torch.cuda.empty_cache()

                    data **= 2
                    SSE += data.sum()
                    #print(f"Chunk {z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end} SSE: {data.sum()}")
                    del coord_grid, output, data
                    torch.cuda.empty_cache()
        
        MSE = SSE / (full_shape[0]*full_shape[1]*full_shape[2])
        #print(f"MSE: {MSE}, shape {full_shape}")
        y = 10 * torch.log10(MSE)
        y = 20.0 * torch.log10(data_max-data_min) - y
    #print(f"Data min/max: {data_min}/{data_max}")
    #print(f"PSNR: {y.item() : 0.03f}")
    opt['extents'] = init_extents
    opt['full_shape'] = full_shape
    return y.item()

def error_volume(model, dataset, opt):
    
    
    grid = list(dataset.data.shape[2:])
    
    
    with torch.no_grad():
        result = sample_grid(model, grid, max_points=1000000,
                             device=opt['device'],
                             data_device=opt['data_device'])
    result = result.to(opt['data_device'])
    result = result.permute(3, 0, 1, 2).unsqueeze(0)
    create_path(os.path.join(output_folder, "ErrorVolume"))
    
    result -= dataset.data
    result **= 2
    tensor_to_cdf(result, 
        os.path.join(output_folder, "ErrorVolume",
        opt['save_name'] + "_error.nc"))

def data_hist(model, opt):
    grid = list(opt['full_shape'])
    with torch.no_grad():
        result = sample_grid(model, grid, max_points=1000000,
            align_corners=opt['align_corners'],
            device=opt['device'],
            data_device=opt['data_device'])
        result = result.cpu().numpy().flatten()
    import matplotlib.pyplot as plt
    plt.hist(result, bins=100)
    plt.show()
    
def scale_distribution(model, opt):
    import matplotlib.pyplot as plt
    grid_scales = torch.diagonal(model.get_transformation_matrices()[:,], 0, 1, 2)[0:3]
    x_scales = grid_scales[:,0].detach().cpu().numpy()
    y_scales = grid_scales[:,1].detach().cpu().numpy()
    z_scales = grid_scales[:,2].detach().cpu().numpy()
    plt.hist(x_scales, alpha=0.4, bins=20, label="X scales")
    plt.hist(y_scales, alpha=0.4, bins=20, label="Y scales")
    plt.hist(z_scales, alpha=0.4, bins=20, label="Z scales")
    plt.legend(loc='upper right')
    plt.title("Scale distributions")
    create_path(os.path.join(output_folder, "ScaleDistributions"))
    plt.savefig(os.path.join(output_folder, "ScaleDistributions", opt['save_name']+'.png'))

def test_throughput(model, opt):
    import torch.profiler

    batch = 2**23
    num_forward = 100

    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        input_data :torch.Tensor = torch.rand([batch, 3], device=opt['device'], dtype=torch.float32)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(num_forward):
            if i == num_forward // 2:  # Profile the middle iteration
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                    model(input_data)
            else:
                model(input_data)
            torch.cuda.synchronize()
        t1 = time.time()
    passed_time = t1 - t0
    points_queried = batch * num_forward
    print(f"Time for {num_forward} passes with batch size {batch}: {passed_time}")
    print(f"Throughput: {points_queried/passed_time} points per second")
    GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
                / (1024**3))
    print(f"{GBytes : 0.02f}GB of memory used during test.")
    
    # Save the trace
    prof.export_chrome_trace(os.path.join(output_folder, opt['save_name'] + "_forward_trace.json"))

    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()

    input_data :torch.Tensor = torch.rand([batch, 3], device=opt['device'], dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(num_forward):
            if i == num_forward // 2:  # Profile the middle iteration
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                    model(input_data)
            else:
                model(input_data)
            torch.cuda.synchronize()
        t1 = time.time()
    passed_time = t1 - t0
    points_queried = batch * num_forward
    print(f"Time for {num_forward} passes with batch size {batch} and autocast: {passed_time}")
    print(f"Throughput: {points_queried/passed_time} points per second")
    GBytes = (torch.cuda.max_memory_allocated(device=opt['device']) \
                / (1024**3))
    print(f"{GBytes : 0.02f}GB of memory used during test.")
    
    # Save the trace for AMP forward
    prof.export_chrome_trace(os.path.join(output_folder, opt['save_name'] + "_amp_forward_trace.json"))

def feature_density(model, opt):
    
    # Load the reference data
    dataset = Dataset(opt)
    
    create_path(os.path.join(output_folder, "FeatureDensity"))
    
    data_shape = list(dataset.data.shape[2:])
    grid = make_coord_grid(data_shape, opt['device'], 
                           flatten=True, align_corners=opt['align_corners'])
    with torch.no_grad():
        print(grid.device)
        
        density = forward_maxpoints(model.feature_density, grid, 
                                    data_device=opt['data_device'], 
                                    device=opt['device'],
                                    max_points=1000000)
        density = density.reshape(data_shape)
        density = density.unsqueeze(0).unsqueeze(0)
        density = density / density.sum()
        
        tensor_to_cdf(density, 
            os.path.join(output_folder, 
            "FeatureDensity", opt['save_name']+"_density.nc"))
        
        result = sample_grid(model, list(dataset.data.shape[2:]), 
                             max_points=1000000,
                             device=opt['device'],
                             data_device=opt['data_device'])
        result = result.to(opt['data_device'])
        result = result.permute(3, 0, 1, 2).unsqueeze(0)
        result -= dataset.data
        result **= 2
        result /= result.mean()
        result = torch.exp(torch.log(density+1e-16) / torch.exp(result))
        result /= result.sum()
        tensor_to_cdf(result, 
            os.path.join(output_folder, 
            "FeatureDensity", opt['save_name']+"_targetdensity.nc"))     
        
        result = F.kl_div(torch.log(density+1e-16), 
                          torch.log(result+1e-16), 
                               reduction="none", 
                               log_target=True)           
        tensor_to_cdf(result, 
            os.path.join(output_folder, 
            "FeatureDensity", opt['save_name']+"_kl.nc"))    
        
def feature_locations(model, opt):
    if(opt['model'] == "afVSRN"):
        feat_locations = model.feature_locations.detach().cpu().numpy()
        np.savetxt(os.path.join(output_folder, "feature_locations", opt['save_name']+".csv"),
                feat_locations, delimiter=",")
    elif(opt['model'] == "AMRSRN"):
        feat_grid_shape = opt['feature_grid_shape'].split(',')
        feat_grid_shape = [eval(i) for i in feat_grid_shape]
        with torch.no_grad():
            global_points = make_coord_grid(feat_grid_shape, opt['device'], 
                                flatten=True, align_corners=opt['align_corners'])
            
            transformed_points = model.transform(global_points)
            ids = torch.arange(transformed_points.shape[0])
            ids = ids.unsqueeze(1).unsqueeze(1)
            ids = ids.repeat([1, transformed_points.shape[1], 1])
            transformed_points = torch.cat((transformed_points, ids), dim=2)
            transformed_points = transformed_points.flatten(0,1).numpy()
        
        create_path(os.path.join(output_folder, "FeatureLocations"))

        np.savetxt(os.path.join(output_folder, "FeatureLocations", opt['save_name']+".csv"),
                transformed_points, delimiter=",", header="x,y,z,id")
        
        print(f"Largest/smallest transformed points: {transformed_points.min()} {transformed_points.max()}")
    
def quantize_model(model, dataset, opt):
    """ Fuse together convolutions/linear layers and ReLU """
    opt['data_device'] = "cpu"
    opt['device'] = "cpu"
    model = model.to("cpu")
    # list all modules in the model
    q_model = load_model(opt, "cpu")
    q_model = q_model.to("cpu")
    q_model.train(False)
    q_model.eval()
    q_model.init_quant()
    torch.quantization.fuse_modules(model, [['decoder.0.linear', 'decoder.0.relu'], 
                                    ['decoder.1.linear', 'decoder.1.relu']], inplace=True)
    # test psnr
    dataset = Dataset(opt)
    psnr, SSE, MSE, numel = test_psnr(q_model, dataset, opt)
    print(f"PSNR fused: {psnr : 0.03f}")
    # save and get model size
    torch.save(q_model.state_dict(), os.path.join(output_folder, "fused_model.pth"))
    model_size = os.path.getsize(os.path.join(output_folder, "fused_model.pth")) / (1024**2)
    print(f"Model size post fusing: {model_size : 0.02f} MB")
    
    q_model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(q_model, inplace=True)
    q_model(torch.rand([2**17, 3], dtype=torch.float32, device="cpu")*2-1)
    torch.quantization.convert(q_model, inplace=True)
    torch.save(q_model.state_dict(), os.path.join(output_folder, "quantized_model.pth"))
    model_size = os.path.getsize(os.path.join(output_folder, "quantized_model.pth")) / (1024**2)
    print(f"Model size post quantization: {model_size : 0.02f} MB")

    # post quantization psnr
    psnr, SSE, MSE, numel = test_psnr(q_model, dataset, opt)
    print(f"PSNR quantized: {psnr : 0.03f}")

    # print original model size
    torch.save(model.state_dict(), os.path.join(output_folder, "original_model.pth"))
    original_model_size = os.path.getsize(os.path.join(output_folder, "original_model.pth")) / (1024**2)
    print(f"Original model size: {original_model_size : 0.02f} MB")

def convert_to_trt(model, opt):
    """ Fuse together convolutions/linear layers and ReLU """
    torch.onnx.export(model, 
                      torch.rand([2**17, 3], 
                                 dtype=torch.float32, 
                                 device=opt['device']), 
                                 os.path.join(save_folder, opt['save_name'], "model.onnx"),
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'},
                                  'output' : {0 : 'batch_size'}},
                    opset_version=20)
    #Check onnx
    import onnx
    onnx_model = onnx.load(os.path.join(save_folder, opt['save_name'], "model.onnx"))
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(os.path.join(save_folder, opt['save_name'], "model.onnx"), 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse ONNX file")
    config.max_workspace_size = 1 << 32  # 1GB
    engine = builder.build_engine(network, config)
    if engine is None:
        raise ValueError("Failed to build TensorRT engine")
    with open(os.path.join(save_folder, opt['save_name'], "model.trt"), 'wb') as f:
        f.write(engine.serialize())

def convert_to_torchscript(model, opt):
    with torch.no_grad():
        input_tensor = torch.rand([2**21, 3], dtype=torch.float32, device="cuda")*2-1
        rand_tensor = torch.rand([2**21, 3], dtype=torch.float32, device="cuda")*2-1
        # traced_model = torch.jit.trace(model, input_tensor)
        
        # for i in range(10):
        #     traced_model(input_tensor)
        # g = torch.cuda.CUDAGraph()
        # with torch.cuda.graph(g):
        #     output = traced_model(input_tensor)
        
        # def run_inference(new_input):
        #     input_tensor.copy_(new_input)
        #     g.replay()
        #     return output.clone()
        
        # 
        # result = run_inference(rand_tensor)
        # result_og = model(rand_tensor)
        # print(f"Difference with traced graph: {torch.mean(torch.abs(result-result_og)) : 0.05f}")

        # #autocasted model difference
        # with torch.autocast(device_type='cuda', dtype=torch.float16):   
        #     result_autocast = model(rand_tensor)
        #     print(f"Difference with autocast float16: {torch.mean(torch.abs(result_autocast-result_og)) : 0.05f}")

        
        # # timing test for cuda graph
        # import time
        # times = []
        # for i in range(100):
        #     torch.cuda.synchronize()
        #     t0 = time.time()
        #     result = run_inference(rand_tensor)
        #     torch.cuda.synchronize()
        #     t1 = time.time()
        #     times.append(t1-t0)
        # print(f"Mean time for cuda graph: {sum(times) / len(times) : 0.05f} seconds")

        # # just the torchscript
        # times = []
        # for i in range(100):
        #     torch.cuda.synchronize()
        #     t0 = time.time()
        #     result = traced_model(rand_tensor)
        #     torch.cuda.synchronize()
        #     t1 = time.time()
        #     times.append(t1-t0)
        # print(f"Mean time for torchscript: {sum(times) / len(times) : 0.05f} seconds")


        # # just the torchscript
        # times = []
        # traced_model = torch.jit.optimize_for_inference(traced_model)
        # for i in range(100):
        #     torch.cuda.synchronize()
        #     t0 = time.time()
        #     result = traced_model(rand_tensor)
        #     torch.cuda.synchronize()
        #     t1 = time.time()
        #     times.append(t1-t0)
        # print(f"Mean time for torchscript optimize_for_inference: {sum(times) / len(times) : 0.05f} seconds")


        # # autocast model
        # times = []
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        #     for i in range(100):
        #         torch.cuda.synchronize()
        #         t0 = time.time()
        #         result = traced_model(rand_tensor)
        #         torch.cuda.synchronize()
        #         t1 = time.time()
        #         times.append(t1-t0)
        # print(f"Mean time for autocast torchscript: {sum(times) / len(times) : 0.05f} seconds")

        # now test model
        times = []
        for i in range(100):
            torch.cuda.synchronize()
            t0 = time.time()
            result = model(rand_tensor)
            torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1-t0)
        print(f"Mean time for model: {sum(times) / len(times) : 0.05f} seconds")

        # compiled model
        times = []
        with torch.autocast(device_type='cuda', dtype=torch.float16), torch.inference_mode():
            for i in range(100):
                torch.cuda.synchronize()
                t0 = time.time()
                result = model(rand_tensor)
                torch.cuda.synchronize()
                t1 = time.time()
                times.append(t1-t0)
        print(f"Mean time for amp+inference model: {sum(times) / len(times) : 0.05f} seconds")
    
def half_precision(model, opt):
    with torch.no_grad():
        input_tensor = torch.rand([2**21, 3], dtype=torch.float32, device="cuda")*2-1
        
        times = []
        for i in range(100):
            torch.cuda.synchronize()
            t0 = time.time()
            result = model(input_tensor)
            torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1-t0)
        print(f"Mean time for normal model: {sum(times) / len(times) : 0.05f} seconds")

        # amp model
        times = []
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i in range(100):
                torch.cuda.synchronize()
                t0 = time.time()
                result = model(input_tensor)
                torch.cuda.synchronize()
                t1 = time.time()
                times.append(t1-t0)
        print(f"Mean time for amp model: {sum(times) / len(times) : 0.05f} seconds")

        model = model.half()
        input_tensor = input_tensor.half()
        times = []
        for i in range(100):
            torch.cuda.synchronize()
            t0 = time.time()
            result = model(input_tensor)
            torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1-t0)
        print(f"Mean time for half precision model: {sum(times) / len(times) : 0.05f} seconds")

def compression_test(model, opt):
    font_size = 40
    marker_size = 10
    line_width = 5

    model = model.to("cuda").eval()
    original_psnr = test_psnr_chunked(model, opt)
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # Size in MB
    print(f"Original model PSNR: {original_psnr:.2f} dB")
    print(f"Original model size: {original_size:.2f} MB")

    # Define error bounds and precisions
    sz3_zfp_abs_error_bounds = [0.5, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.001, 0.0001]
    fpzip_precisions = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 28, 32]

    # Define parameter subsets
    param_subsets = {
        #"transform": ["_scales", "_rotations", "translations"],
        #"network": ["feature_grids", "decoder"],
        "feature_grids": ["feature_grids"],
        #"decoder": ["decoder"],
        #"all": ["_scales", "_rotations", "translations", "feature_grids", "decoder"]
    }

    results = {
        "sz3": {subset: [] for subset in param_subsets},
        "fpzip": {subset: [] for subset in param_subsets},
        "zfp": {subset: [] for subset in param_subsets}
    }

    with torch.no_grad():
        for subset_name, subset in param_subsets.items():
            print(f"Testing compression for {subset_name} parameters")
            
            # SZ3 ABS
            for bound in sz3_zfp_abs_error_bounds:
                test_model = load_model(opt, "cuda")
                test_model = test_model.to("cuda")
                test_model.train(False)
                test_model.eval()
                test_model_p = test_psnr_chunked(test_model, opt)
                print(f"PSNR of test_model: {test_model_p:.2f} dB")
                try:
                    compressed_size, psnr, bound = compress_and_test(model, test_model, opt, bound, "sz3", subset)
                    results["sz3"][subset_name].append((compressed_size, psnr, bound))
                except Exception as e:
                    print(f"Error in SZ3 ABS compression for {subset_name} with bound {bound}: {str(e)}")
            
            for bound in sz3_zfp_abs_error_bounds:
                test_model = load_model(opt, "cuda")
                test_model = test_model.to("cuda")
                test_model.train(False)
                test_model.eval()
                test_model_p = test_psnr_chunked(test_model, opt)
                print(f"PSNR of test_model: {test_model_p:.2f} dB")
                try:
                    compressed_size, psnr, bound = compress_and_test(model, test_model, opt, bound, "zfp", subset)
                    results["zfp"][subset_name].append((compressed_size, psnr, bound))
                except Exception as e:
                    print(f"Error in ZFP ABS compression for {subset_name} with bound {bound}: {str(e)}")

            # FPZIP
            for precision in fpzip_precisions:
                test_model = load_model(opt, "cuda")
                test_model = test_model.to("cuda")
                test_model.eval()
                try:
                    compressed_size, psnr, precision = compress_and_test(model, test_model, opt, precision, "fpzip", subset)
                    results["fpzip"][subset_name].append((compressed_size, psnr, precision))
                except Exception as e:
                    print(f"Error in FPZIP compression for {subset_name} with precision {precision}: {str(e)}")

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))

    # Find global min and max PSNR values
    all_psnrs = [psnr for subsets in results.values() for data in subsets.values() for _, psnr, _ in data]
    all_psnrs.append(original_psnr)
    min_psnr = 20
    max_psnr = min(50, np.max(all_psnrs)*1.1)

    # Find global min and max sizes
    all_sizes = [size for subsets in results.values() for data in subsets.values() for size, _, _ in data]
    all_sizes.append(original_size)
    min_size = 0
    max_size = original_size*1.1

    for subset_name, _ in param_subsets.items():
        for compressor, data in results.items():
            if data[subset_name]:
                sizes, psnrs, bounds = zip(*data[subset_name])
                ax.plot(sizes, psnrs, 'o-', label=f"{compressor.upper()}", markersize=marker_size, linewidth=line_width)
    
    ax.plot(original_size, original_psnr, 'r*', markersize=marker_size*3, label='Original Model')
    ax.set_xlabel("Compressed Size (MB)", fontsize=font_size)
    ax.set_ylabel("PSNR (dB)", fontsize=font_size)
    ax.set_title(f"{opt['save_name'].split('_')[0].split('.')[0].rstrip('0123456789').capitalize()}", fontsize=font_size)
    ax.legend(fontsize=font_size)
    ax.grid(True)
    ax.set_ylim(min_psnr, max_psnr)
    ax.set_xlim(min_size, max_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, opt['save_name'], "compression.png"))

    # Create a histogram for the model's feature grid values
    feature_grids = model.feature_grids.cpu().detach().numpy()
    flattened_grids = feature_grids.flatten()

    plt.figure(figsize=(12, 8))
    plt.hist(flattened_grids, bins=100, edgecolor='black')
    #plt.title("Histogram of Feature Grid Values", fontsize=font_size)
    plt.xlabel("Value", fontsize=font_size)
    plt.ylabel("Frequency", fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, opt['save_name'], "feature_grid_histogram.png"))
    plt.close()

    # Print some statistics about the feature grid values
    print("\nFeature Grid Statistics:")
    print(f"Min value: {np.min(flattened_grids):.4f}")
    print(f"Max value: {np.max(flattened_grids):.4f}")
    print(f"Mean value: {np.mean(flattened_grids):.4f}")
    print(f"Median value: {np.median(flattened_grids):.4f}")
    print(f"Standard deviation: {np.std(flattened_grids):.4f}")
    # Print results
    for compressor, subsets in results.items():
        print(f"\n{compressor.upper()} Results:")
        for subset_name, data in subsets.items():
            print(f"  {subset_name} parameters:")
            for size, psnr, bound in data:
                print(f"    Compressed Size: {size:.2f} MB, PSNR: {psnr:.2f} dB, Error Bound: {bound:.4f}")

def compress_and_test(model, test_model, opt, error_bound_or_precision, compressor, param_subset):
    params = {
        '_scales': model._scales.cpu().detach().numpy().astype(np.float32),
        '_rotations': model._rotations.cpu().detach().numpy().astype(np.float32),
        'translations': model.translations.cpu().detach().numpy().astype(np.float32),
        'feature_grids': model.feature_grids.cpu().detach().numpy().astype(np.float32),
        'decoder': [p.cpu().detach().numpy().astype(np.float32) for p in model.decoder.parameters()]
    }
    
    total_compressed_size = 0
    
    for name in params:
        if name in param_subset:
            continue
        if name in ['_scales', '_rotations', 'translations']:
            total_compressed_size += params[name].nbytes
        elif name == 'feature_grids':
            total_compressed_size += params['feature_grids'].nbytes
        elif name == 'decoder':
            total_compressed_size += sum(p.nbytes for p in params['decoder'])
    
    for name in param_subset:
        if name in ['_scales', '_rotations', 'translations']:
            param = params[name]
            param.tofile(f"temp_{name}.raw")
            dims = ' '.join(str(d) for d in param.shape)
            dim_flag = f"-{len(param.shape)} {dims}"
            
            if compressor == 'sz3':
                command = f"sz3 -f -z temp_{name}.sz -i temp_{name}.raw -o temp_{name}.sz.out -M ABS {error_bound_or_precision} {dim_flag}"
                output_file = f"temp_{name}.sz.out"
            elif compressor == 'fpzip':
                command = f"fpzip -t float -p {error_bound_or_precision} {dim_flag} -i temp_{name}.raw -o temp_{name}.fpz"
                output_file = f"temp_{name}.fpz"
            elif compressor == 'zfp':
                command = f"zfp -f -i temp_{name}.raw -z temp_{name}.zfp -a {error_bound_or_precision} {dim_flag}"
                output_file = f"temp_{name}.zfp"
            
            subprocess.run(command, check=True)
            
            compressed_size = os.path.getsize(output_file)
            total_compressed_size += compressed_size
            
            if compressor == 'fpzip':
                command = f"fpzip -d -t float -i temp_{name}.fpz -o temp_{name}.out"
                subprocess.run(command, check=True)
                output_file = f"temp_{name}.out"
            elif compressor == 'zfp':
                command = f"zfp -f -z temp_{name}.zfp -o temp_{name}.out {dim_flag}"
                subprocess.run(command, check=True)
                output_file = f"temp_{name}.out"
            
            decompressed = np.fromfile(output_file, dtype=np.float32).reshape(param.shape)
            setattr(test_model, name, torch.nn.Parameter(torch.from_numpy(decompressed).to(model.feature_grids.device)))
        
        elif name == 'feature_grids':
            for i in range(params['feature_grids'].shape[1]):
                param = params['feature_grids'][:, i]
                param.tofile(f"temp_feature_grids_{i}.raw")
                dims = ' '.join(str(d) for d in param.shape)
                dim_flag = f"-{len(param.shape)} {dims}"
                
                if compressor == 'sz3':
                    command = f"sz3 -f -z temp_feature_grids_{i}.sz -i temp_feature_grids_{i}.raw -o temp_feature_grids_{i}.sz.out -M ABS {error_bound_or_precision} {dim_flag}"
                    compressed_file = f"temp_feature_grids_{i}.sz"
                    output_file = f"temp_feature_grids_{i}.sz.out"
                elif compressor == 'fpzip':
                    command = f"fpzip -t float -p {error_bound_or_precision} {dim_flag} -i temp_feature_grids_{i}.raw -o temp_feature_grids_{i}.fpz"
                    compressed_file = f"temp_feature_grids_{i}.fpz"
                elif compressor == 'zfp':
                    command = f"zfp -f -i temp_feature_grids_{i}.raw -z temp_feature_grids_{i}.zfp -a {error_bound_or_precision} {dim_flag}"
                    compressed_file = f"temp_feature_grids_{i}.zfp"
                
                subprocess.run(command, check=True)
                
                compressed_size = os.path.getsize(compressed_file)
                total_compressed_size += compressed_size
                
                if compressor == 'fpzip':
                    command = f"fpzip -d -t float -i temp_feature_grids_{i}.fpz -o temp_feature_grids_{i}.out"
                    subprocess.run(command, check=True)
                    output_file = f"temp_feature_grids_{i}.out"
                elif compressor == 'zfp':
                    command = f"zfp -f -z temp_feature_grids_{i}.zfp -o temp_feature_grids_{i}.out -a {error_bound_or_precision} {dim_flag}"
                    subprocess.run(command, check=True)
                    output_file = f"temp_feature_grids_{i}.out"
                
                decompressed = np.fromfile(output_file, dtype=np.float32).reshape(param.shape)
                test_model.feature_grids[:, i] = torch.from_numpy(decompressed).to(model.feature_grids.device)
        
        elif name == 'decoder':
            for i, layer_param in enumerate(params['decoder']):
                layer_param.tofile(f"temp_decoder_{i}.raw")
                dims = ' '.join(str(d) for d in layer_param.shape)
                dim_flag = f"-{len(layer_param.shape)} {dims}"
                
                if compressor == 'sz3':
                    command = f"sz3 -f -z temp_decoder_{i}.sz -i temp_decoder_{i}.raw -o temp_decoder_{i}.sz.out -M ABS {error_bound_or_precision} {dim_flag}"
                    output_file = f"temp_decoder_{i}.sz.out"
                elif compressor == 'fpzip':
                    command = f"fpzip -t float -p {error_bound_or_precision} {dim_flag} -i temp_decoder_{i}.raw -o temp_decoder_{i}.fpz"
                    output_file = f"temp_decoder_{i}.fpz"
                elif compressor == 'zfp':
                    command = f"zfp -f -i temp_decoder_{i}.raw -z temp_decoder_{i}.zfp -a {error_bound_or_precision} {dim_flag}"
                    output_file = f"temp_decoder_{i}.zfp"
                
                subprocess.run(command, check=True)
                
                compressed_size = os.path.getsize(output_file)
                total_compressed_size += compressed_size
                
                if compressor == 'fpzip':
                    command = f"fpzip -d -t float -i temp_decoder_{i}.fpz -o temp_decoder_{i}.out"
                    subprocess.run(command, check=True)
                    output_file = f"temp_decoder_{i}.out"
                elif compressor == 'zfp':
                    command = f"zfp -f -z temp_decoder_{i}.zfp -o temp_decoder_{i}.out"
                    subprocess.run(command, check=True)
                    output_file = f"temp_decoder_{i}.out"
                
                decompressed = np.fromfile(output_file, dtype=np.float32).reshape(layer_param.shape)
                list(test_model.decoder.parameters())[i].data = torch.from_numpy(decompressed).to(model.feature_grids.device)

    # Clean up temporary files
    for name in param_subset:
        if name in ['_scales', '_rotations', 'translations']:
            for ext in ['raw', 'sz', 'fpz', 'zfp', 'out']:
                file_path = f"temp_{name}.{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
        elif name == 'feature_grids':
            for i in range(model.feature_grids.shape[1]):
                for ext in ['raw', 'sz', 'fpz', 'zfp', 'out']:
                    file_path = f"temp_feature_grids_{i}.{ext}"
                    if os.path.exists(file_path):
                        os.remove(file_path)
        elif name == 'decoder':
            for i in range(len(params['decoder'])):
                for ext in ['raw', 'sz', 'fpz', 'zfp', 'out']:
                    file_path = f"temp_decoder_{i}.{ext}"
                    if os.path.exists(file_path):
                        os.remove(file_path)

    # Test PSNR using test_psnr_chunked
    psnr = test_psnr_chunked(test_model, opt)
    
    return total_compressed_size / (1024 * 1024), psnr, error_bound_or_precision  # Return size in MB

def comp_report(model, opt):
    # Compute original model size and PSNR
    original_size = opt['full_shape'][0] * opt['full_shape'][1] * opt['full_shape'][2] * 4 * opt['n_timesteps']
    original_psnr = test_psnr_chunked(model, opt)
    # read the model file's size in bytes
    model_size = os.path.getsize(os.path.join(save_folder, opt['save_name'], "compressed_model.zip"))

    # Print results in CSV format
    print("Compressed Size (bytes),Compression Ratio,PSNR")
    print(f"{model_size},{original_size / model_size:.2f},{original_psnr:.3f}")

def perform_tests(model, tests, opt, timestep):
    if("reconstruction" in tests):
        output = model_reconstruction_chunked(model, opt, timestep)
        create_path(os.path.join(output_folder, "Reconstruction"))
        tensor_to_cdf(output, 
            os.path.join(output_folder, 
            "Reconstruction", opt['save_name']+"_timestep_"+str(timestep)+".nc"))
    if("feature_locations" in tests):
        feature_locations(model, opt)
    if("error_volume" in tests):
        error_volume(model, opt)
    if("scale_distribution" in tests):
        scale_distribution(model, opt)
    if("feature_density" in tests):
        feature_density(model, opt)
    if("psnr" in tests):
        p = test_psnr_chunked(model, opt)
    if("histogram" in tests):
        data_hist(model, opt)
    if("throughput" in tests):
        test_throughput(model, opt)
    if("grid_influence" in tests):
        test_grid_influence(model, opt)
    if("quantize" in tests):
        quantize_model(model, opt)
    if("torchscript" in tests):
        convert_to_torchscript(model, opt)
    if("trt" in tests):
        convert_to_trt(model, opt)  
    if("half_precision" in tests):
        half_precision(model, opt)
    if("compression" in tests):
        compression_test(model, opt)
    if("comp_report" in tests):
        comp_report(model, opt)

    if("psnr" in tests):
        return p
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--tests_to_run',default=None,type=str,
                        help="A set of tests to run, separated by commas. Options are psnr, reconstruction, error_volume, histogram, throughput, and feature_locations.")
    parser.add_argument('--device',default=None,type=str,
                        help="Device to load model to")
    parser.add_argument('--data_device',default=None,type=str,
                        help="Device to load data to")
    args = vars(parser.parse_args())
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    tests_to_run = args['tests_to_run'].split(',')
    
    # Load the model
    opt = load_options(os.path.join(save_folder, args['load_from']))
    opt['device'] = args['device']
    opt['data_device'] = args['data_device']
    model = load_model(opt, args['device'])
    model = model.to(opt['device'])
    model.train(False)
    model.eval()
    
    # Perform tests
    psnrs = []
    print(opt['n_timesteps'])
    for t in range(opt['n_timesteps']):
        if opt['n_timesteps'] > 1:
            print(f"========= Timestep {t} ==========")
        model.set_default_timestep(t)
        p = perform_tests(model, tests_to_run, opt, timestep=t)
        if(p is not None):
            psnrs.append(p)
        model.unload_timestep(t)
        print()
    
    if(len(psnrs) > 0):
        print(f"PSNRs: {psnrs}")
        print(f"Mean PSNR: {sum(psnrs) / len(psnrs) : 0.03f}")
        

    
        



        

