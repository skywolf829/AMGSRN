import torch
import torch.nn.functional as F
from AMG_Encoder import create_transformation_matrices, encode, feature_density
from utils import get_random_rst, get_random_feature_grids, get_random_points, random_quaternion
import tinycudann as tcnn 
from pytorch_ops import feature_density_pytorch, encode_pytorch
from torch.profiler import profile, record_function, ProfilerActivity


def timing_test_forward_pass_CUDA(points, model, n_points, num_cycles):
    print("======================================================================")
    print("===================CUDA extension time/memory test====================")
    print("======================================================================")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.max_memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        torch.cuda.synchronize()
        start.record()
        with record_function("CUDA_inference"):
            for _ in range(num_cycles):
                model.forward_cuda(points)
        end.record()
        torch.cuda.synchronize()
    time_passed = start.elapsed_time(end)
    end_mem = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"Static memory in use: {(start_mem)/(1024**2):0.02f} MB")
    print(f"Memory used by operation: {(end_mem-start_mem)/(1024**2):0.02f} MB")
    print(f"Time passed: {time_passed:0.02f} ms")
    print(f"Throughput: {n_points*num_cycles / time_passed * 1000 :0.02f} points/s")
    print()
    print()

def timing_test_density_CUDA(points, model, n_points, num_cycles):
    print("======================================================================")
    print("===================CUDA extension time/memory test====================")
    print("======================================================================")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.max_memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        torch.cuda.synchronize()
        start.record()
        with record_function("CUDA_feature_density"):
            for _ in range(num_cycles):
                model.feature_density_cuda(points)
        end.record()
        torch.cuda.synchronize()
    time_passed = start.elapsed_time(end)
    end_mem = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"Static memory in use: {(start_mem)/(1024**2):0.02f} MB")
    print(f"Memory used by operation: {(end_mem-start_mem)/(1024**2):0.02f} MB")
    print(f"Time passed: {time_passed:0.02f} ms")
    print(f"Throughput: {n_points*num_cycles / time_passed * 1000 :0.02f} points/s")
    print()
    print()

def timing_test_forward_pass_PyTorch(points, model, n_points, num_cycles):
    print("======================================================================")
    print("=======================PyTorch time/memory test=======================")
    print("======================================================================")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()

    start_mem = torch.cuda.max_memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        torch.cuda.synchronize()
        start.record()
        with record_function("PyTorch_inference"):
            for _ in range(num_cycles):
                model.forward_pytorch(points)
        end.record()
        torch.cuda.synchronize()
    time_passed = start.elapsed_time(end)
    end_mem = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()

    print(f"Static memory in use: {(start_mem)/(1024**2):0.02f} MB")
    print(f"Memory used by operation: {(end_mem-start_mem)/(1024**2):0.02f} MB")
    print(f"Time passed: {time_passed:0.02f} ms")
    print(f"Throughput: {n_points*num_cycles / time_passed * 1000 :0.02f} points/s")
    print()
    print()

def timing_test_density_PyTorch(points, model, n_points, num_cycles):
    print("======================================================================")
    print("=======================PyTorch time/memory test=======================")
    print("======================================================================")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()

    start_mem = torch.cuda.max_memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        torch.cuda.synchronize()
        start.record()
        with record_function("PyTorch_feature_density"):
            for _ in range(num_cycles):
                model.feature_density_pytorch(points)
        end.record()
        torch.cuda.synchronize()
    time_passed = start.elapsed_time(end)
    end_mem = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()

    print(f"Static memory in use: {(start_mem)/(1024**2):0.02f} MB")
    print(f"Memory used by operation: {(end_mem-start_mem)/(1024**2):0.02f} MB")
    print(f"Time passed: {time_passed:0.02f} ms")
    print(f"Throughput: {n_points*num_cycles / time_passed * 1000 :0.02f} points/s")
    print()
    print()


def nsight_profile(model, points):
    
    model_out = model(points)
    err = ((model_out)**2)
    err.mean().backward()
    
    '''
    density = model.feature_density_cuda(points)
    err_density = density**2
    err_density.mean().backward()
    '''
    

class AMGSRN(torch.nn.Module):
    def __init__(self, n_grids, feats_per_grid, feat_grid_dim):
        super(AMGSRN, self).__init__()
        r, s, t = get_random_rst(n_grids)
        features = get_random_feature_grids(n_grids, feats_per_grid, feat_grid_dim)
        self.decoder = tcnn.Network(
            n_input_dims=feats_per_grid*n_grids,
            n_output_dims=1 ,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )
        self.r = torch.nn.Parameter(r,requires_grad=True)
        self.s = torch.nn.Parameter(s,requires_grad=True)
        self.t = torch.nn.Parameter(t,requires_grad=True)
        self.features = torch.nn.Parameter(features,requires_grad=True)
    
    def feature_density_cuda(self, points):
        return feature_density(points, self.r, self.s, self.t) 
    
    def feature_density_pytorch(self, points):
        return feature_density_pytorch(points, self.r, self.s, self.t) 
    
    def forward_cuda(self, x):
        return self.decoder(encode(x, self.r, self.s, self.t, self.features)).float()
    
    def forward_pytorch(self, x):
        return self.decoder(encode_pytorch(self.features, self.r, self.s, self.t, x)).float()
    
    def forward(self, x):
        return self.forward_cuda(x)

def profile_efficiency_forward(points, model, n_points, num_cycles):

    '''with profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True) as prof:
    '''
    timing_test_forward_pass_PyTorch(points, model, n_points, num_cycles)
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    timing_test_forward_pass_CUDA(points, model, n_points, num_cycles)
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    '''
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_memory_timeline(f"profile.html", device="cuda:0")
    prof.export_chrome_trace("trace.json")
    '''

def profile_efficiency_density(points, model, n_points, num_cycles):

    '''with profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True) as prof:
    '''
    timing_test_density_PyTorch(points, model, n_points, num_cycles)
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    timing_test_density_CUDA(points, model, n_points, num_cycles)
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    '''
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_memory_timeline(f"profile.html", device="cuda:0")
    prof.export_chrome_trace("trace.json")
    '''

if __name__ == "__main__":
    
    # Testing params
    num_cycles = 10

    # Hyperparams
    n_grids = 32
    feats_per_grid = 2
    feat_grid_dim = [32, 32, 32]
    n_points = 2**23

   
    # Setup model
    points = get_random_points(n_points)
    model = AMGSRN(n_grids, feats_per_grid, feat_grid_dim)
    
    #nsight_profile(model, points)
    profile_efficiency_forward(points, model, n_points, num_cycles)