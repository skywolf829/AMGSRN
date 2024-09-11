from AMG_Encoder import encode, create_transformation_matrices, feature_density
from AMGSRN.Models.AMGSRN_old import AMG_encoder_old, AMGSRN_old
from AMGSRN.Models.AMGSRN import AMGSRN
from AMGSRN.Models.options import Options
import torch
from time import time
    
def randomize_grids( n_grids:int, n_features:int,
                 feat_grid_shape:list[int], n_dims:int):  
    with torch.no_grad():     
        d = "cuda"
        s = torch.ones([n_grids, n_dims], dtype=torch.float32, device = d)
        s += torch.rand_like(s)*0.05
        s = torch.log(s)
        r = torch.zeros([n_grids, 4], dtype=torch.float32, device = d)
        r[:,-1] = 1.0
        t = torch.zeros([n_grids, n_dims], dtype=torch.float32, device = d)
        t += torch.rand_like(t)*0.05
        
    _scales = torch.nn.Parameter(s,requires_grad=True)   
    _rotations = torch.nn.Parameter(r,requires_grad=True)   
    translations = torch.nn.Parameter(t,requires_grad=True)   

    feature_grids =  torch.nn.parameter.Parameter(
        
        torch.ones(
            [n_grids, n_features] + feat_grid_shape,
            dtype=torch.float32,
            device=d
        ).uniform_(-0.0001, 0.0001),
        requires_grad=True
    )
    return feature_grids, translations, _scales, _rotations

def forward_encode_test():

    print("======Forward encoding test======")
    feature_grids, translations, scales, rotations = randomize_grids(32, 2, [64, 64, 64], 3)

    torch.cuda.reset_peak_memory_stats()
    x = torch.rand([2**23, 3], dtype=torch.float32, device="cuda")
    s = torch.exp(scales)
    r = torch.nn.functional.normalize(rotations)
    encode(x, r, s, translations, feature_grids)
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        feats = encode(x, r, s, translations, feature_grids)
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"New encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")


    old_encoder = AMG_encoder_old(32, 2, [64, 64, 64], 3, "default")
    old_encoder.feature_grids = torch.nn.Parameter(feature_grids.clone(), requires_grad=True)
    transformation_matrices = create_transformation_matrices(torch.nn.functional.normalize(rotations), torch.exp(scales), translations)
    del rotations, scales, translations, feats, feature_grids
    torch.cuda.empty_cache()
    old_encoder.transformation_matrices = torch.nn.parameter.Parameter(transformation_matrices, requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    old_encoder(x)
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        feats = old_encoder.forward(x)
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"Old encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")

    # cleanup all
    del old_encoder, x
    torch.cuda.empty_cache()

def backward_encode_test():

    torch.cuda.empty_cache()
    print("======Backward encoding test======")
    feature_grids, translations, scales, rotations = randomize_grids(32, 2, [64, 64, 64], 3)
    x = torch.rand([2**22, 3], dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        feats = encode(x, rotations, scales, translations, feature_grids)
        feats.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"New encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")


    old_encoder = AMG_encoder_old(32, 2, [64, 64, 64], 3, "default")
    old_encoder.feature_grids = torch.nn.Parameter(feature_grids.clone(), requires_grad=True)
    transformation_matrices = create_transformation_matrices(rotations, scales, translations)
    old_encoder.transformation_matrices = torch.nn.parameter.Parameter(transformation_matrices.clone(), requires_grad=True)
    del rotations, scales, translations, feature_grids, transformation_matrices
    torch.cuda.empty_cache()    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        feats = old_encoder(x)
        feats.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"Old encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")

def forward_density_test():

    print("======Forward density test======")
    feature_grids, translations, scales, rotations = randomize_grids(32, 2, [64, 64, 64], 3)

    x = torch.rand([2**23, 3], dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        den = feature_density(x, rotations, scales, translations)  
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"New encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")


    old_encoder = AMG_encoder_old(32, 2, [64, 64, 64], 3, "default")
    old_encoder.feature_grids = torch.nn.parameter.Parameter(feature_grids.clone(), requires_grad=True)
    transformation_matrices = create_transformation_matrices(rotations, scales, translations)
    old_encoder.transformation_matrices = torch.nn.parameter.Parameter(transformation_matrices.clone(), requires_grad=True)
    del rotations, scales, translations, feature_grids, transformation_matrices

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        den = old_encoder.feature_density(x)
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"Old encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")

def backward_density_test():

    print("======Backward density test======")
    feature_grids, translations, scales, rotations = randomize_grids(32, 2, [64, 64, 64], 3)

    x = torch.rand([2**23, 3], dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        den = feature_density(x, rotations, scales, translations)  
        den.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"New encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")


    old_encoder = AMG_encoder_old(32, 2, [64, 64, 64], 3, "default")
    old_encoder.feature_grids = torch.nn.parameter.Parameter(feature_grids.clone(), requires_grad=True)
    transformation_matrices = create_transformation_matrices(rotations, scales, translations)
    old_encoder.transformation_matrices = torch.nn.parameter.Parameter(transformation_matrices.clone(), requires_grad=True)
    del rotations, scales, translations, feature_grids, transformation_matrices
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        den = old_encoder.feature_density(x)
        den.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"Old encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")

def forward_full_test():

    print("======Forward model test======")
    opt = Options.get_default()
    opt['n_grids'] = 32
    opt['n_features'] = 2
    opt['feature_grid_shape'] = "64,64,64"
    opt['data_min']=0.0
    opt['data_max'] = 1.0
    opt['device'] = "cuda"
    
    model_new = AMGSRN(opt).to(opt['device'])
    
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand([2**22, 3], dtype=torch.float32, device="cuda")
    model_new(x)
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        o = model_new(x)
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"New model code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")

    del model_new
    torch.cuda.empty_cache()

    old_model : AMGSRN_old = AMGSRN_old(opt).to(opt['device'])
    torch.cuda.reset_peak_memory_stats()
    old_model(x)
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        o = old_model(x)
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"Old model code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")

def backward_full_test():

    print("======Backward model test======")
    opt = Options.get_default()
    opt['n_grids'] = 32
    opt['n_features'] = 2
    opt['feature_grid_shape'] = "64,64,64"
    opt['data_min']=0.0
    opt['data_max'] = 1.0
    opt['device'] = "cuda"
    
    model_new = AMGSRN(opt).to(opt['device'])
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand([2**23, 3], dtype=torch.float32, device="cuda")
    model_new(x)
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        o = model_new(x)
        o.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"New model code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")

    del model_new
    torch.cuda.empty_cache()
    old_model = AMGSRN_old(opt).to(opt['device'])
    torch.cuda.reset_peak_memory_stats()
    old_model(x)
    torch.cuda.synchronize()
    t0 = time()
    for i in range(10):
        o = old_model(x)
        o.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    print(f"Old model code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")

#forward_encode_test()
#backward_encode_test()
#forward_density_test()
#backward_density_test()
forward_full_test()
#backward_full_test()

