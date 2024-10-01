from AMG_Encoder import encode, create_transformation_matrices, feature_density
from AMGSRN.Models.AMGSRN_pytorch import AMG_encoder_old, AMGSRN_old
from AMGSRN.Models.AMGSRN import AMGSRN
from AMGSRN.Models.options import Options
import torch
from time import time
from torch.profiler import profile, record_function, ProfilerActivity


use_amp = False  # Global variable for controlling automatic mixed precision
batch_size = 2**23  # Global variable for batch size

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
    feature_grids, translations, scales, rotations = randomize_grids(64, 2, [64, 64, 64], 3)

    torch.cuda.reset_peak_memory_stats()
    x = torch.rand([batch_size, 3], dtype=torch.float32, device="cuda")
    s = torch.exp(scales)
    r = torch.nn.functional.normalize(rotations)
    torch.cuda.synchronize()
    
    with torch.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16), torch.no_grad():
        # Warmup
        for _ in range(5):
            encode(x, r, s, translations, feature_grids)
        torch.cuda.synchronize()
        
        t0 = time()
        for _ in range(10):
            feats = encode(x, r, s, translations, feature_grids)
        torch.cuda.synchronize()
        t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"New encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"New encoder throughput: {throughput:.2f} points/sec")

    old_encoder = AMG_encoder_old(32, 2, [64, 64, 64], 3, "default")
    old_encoder.feature_grids = torch.nn.Parameter(feature_grids.clone(), requires_grad=True)
    transformation_matrices = create_transformation_matrices(torch.nn.functional.normalize(rotations), torch.exp(scales), translations)
    del rotations, scales, translations, feats, feature_grids
    torch.cuda.empty_cache()
    old_encoder.transformation_matrices = torch.nn.parameter.Parameter(transformation_matrices, requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    old_encoder(x)
    torch.cuda.synchronize()
    
    with torch.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16), torch.no_grad():
        # Warmup
        for _ in range(5):
            old_encoder.forward(x)
        torch.cuda.synchronize()
        
        t0 = time()
        for _ in range(10):
            feats = old_encoder.forward(x)
        torch.cuda.synchronize()
        t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"Old encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"Old encoder throughput: {throughput:.2f} points/sec")

    # cleanup all
    del old_encoder, x, feats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def backward_encode_test():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("======Backward encoding test======")
    scaler = torch.amp.GradScaler()
    feature_grids, translations, scales, rotations = randomize_grids(32, 2, [32, 32, 32], 3)
    x = torch.rand([batch_size, 3], dtype=torch.float32, device="cuda")
    
    # Warmup
    for _ in range(5):        
        with torch.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16):
            feats = encode(x, rotations, scales, translations, feature_grids)
        scaler.scale(feats).sum().backward()
    torch.cuda.synchronize()
        
    t0 = time()
    for _ in range(10):
        with torch.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16):
            feats = encode(x, rotations, scales, translations, feature_grids)
        scaler.scale(feats).sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"New encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"New encoder throughput: {throughput:.2f} points/sec")

    old_encoder = AMG_encoder_old(32, 2, [64, 64, 64], 3, "default")
    old_encoder.feature_grids = torch.nn.Parameter(feature_grids.clone(), requires_grad=True)
    transformation_matrices = create_transformation_matrices(rotations, scales, translations)
    old_encoder.transformation_matrices = torch.nn.parameter.Parameter(transformation_matrices.clone(), requires_grad=True)
    del rotations, scales, translations, feature_grids, transformation_matrices
    torch.cuda.empty_cache()    
    torch.cuda.reset_peak_memory_stats()
    
        # Warmup
    for _ in range(5):
        with torch.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16):
            feats = old_encoder(x)
        scaler.scale(feats).sum().backward()
    torch.cuda.synchronize()
        
    t0 = time()
    for _ in range(10):
        with torch.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16):
            feats = old_encoder(x)
        scaler.scale(feats).sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"Old encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"Old encoder throughput: {throughput:.2f} points/sec")

    # Cleanup
    del old_encoder, x, feats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def forward_density_test():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("======Forward density test======")
    feature_grids, translations, scales, rotations = randomize_grids(32, 2, [64, 64, 64], 3)

    x = torch.rand([batch_size, 3], dtype=torch.float32, device="cuda")
    
    with torch.autocast(device_type='cuda', enabled=use_amp), torch.no_grad():
        # Warmup
        for _ in range(5):
            feature_density(x, rotations, scales, translations)
        torch.cuda.synchronize()
        
        t0 = time()
        for _ in range(10):
            den = feature_density(x, rotations, scales, translations)  
        torch.cuda.synchronize()
        t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"New encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"New encoder throughput: {throughput:.2f} points/sec")

    old_encoder = AMG_encoder_old(32, 2, [64, 64, 64], 3, "default")
    old_encoder.feature_grids = torch.nn.parameter.Parameter(feature_grids.clone(), requires_grad=True)
    transformation_matrices = create_transformation_matrices(rotations, scales, translations)
    old_encoder.transformation_matrices = torch.nn.parameter.Parameter(transformation_matrices.clone(), requires_grad=True)
    del rotations, scales, translations, feature_grids, transformation_matrices

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.autocast(device_type='cuda', enabled=use_amp), torch.no_grad():
        # Warmup
        for _ in range(5):
            old_encoder.feature_density(x)
        torch.cuda.synchronize()
        
        t0 = time()
        for _ in range(10):
            den = old_encoder.feature_density(x)
        torch.cuda.synchronize()
        t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"Old encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"Old encoder throughput: {throughput:.2f} points/sec")

    # Cleanup
    del old_encoder, x, den
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def backward_density_test():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("======Backward density test======")
    feature_grids, translations, scales, rotations = randomize_grids(32, 2, [64, 64, 64], 3)

    x = torch.rand([batch_size, 3], dtype=torch.float32, device="cuda")
    
    # Warmup
    for _ in range(5):
        den = feature_density(x, rotations, scales, translations)  
        den.sum().backward()
    torch.cuda.synchronize()
    
    t0 = time()
    for _ in range(10):
        den = feature_density(x, rotations, scales, translations)  
        den.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"New encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"New encoder throughput: {throughput:.2f} points/sec")

    old_encoder = AMG_encoder_old(32, 2, [64, 64, 64], 3, "default")
    old_encoder.feature_grids = torch.nn.parameter.Parameter(feature_grids.clone(), requires_grad=True)
    transformation_matrices = create_transformation_matrices(rotations, scales, translations)
    old_encoder.transformation_matrices = torch.nn.parameter.Parameter(transformation_matrices.clone(), requires_grad=True)
    del rotations, scales, translations, feature_grids, transformation_matrices
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(5):
        den = old_encoder.feature_density(x)
        den.sum().backward()
    torch.cuda.synchronize()
    
    t0 = time()
    for _ in range(10):
        den = old_encoder.feature_density(x)
        den.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"Old encoder code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"Old encoder throughput: {throughput:.2f} points/sec")

    # Cleanup
    del old_encoder, x, den
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def forward_full_test():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("======Forward model test======")
    opt = Options.get_default()
    opt['n_grids'] = 32
    opt['n_features'] = 2
    opt['feature_grid_shape'] = "32,32,32"
    opt['data_min']=0.0
    opt['data_max'] = 1.0
    opt['device'] = "cuda"
    
    model_new = AMGSRN(opt).to(opt['device'])
    
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand([batch_size, 3], dtype=torch.float32, device="cuda")
    model_new(x)
    with torch.no_grad(), torch.autocast(device_type=opt['device'], dtype=torch.float16, enabled=use_amp):
        # Warmup
        for _ in range(5):
            model_new(x)
        torch.cuda.synchronize()
        
        t0 = time()
        for _ in range(100):
            o = model_new(x)
        torch.cuda.synchronize()
        t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 100) / (t1 - t0)
    print(f"New model code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"New model throughput: {throughput:.2f} points/sec")

    del model_new
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    old_model : AMGSRN_old = AMGSRN_old(opt).to(opt['device'])
    torch.cuda.reset_peak_memory_stats()
    old_model(x)
    with torch.no_grad(), torch.autocast(device_type=opt['device'], dtype=torch.float16, enabled=use_amp):
        # Warmup
        for _ in range(5):
            old_model(x)
        torch.cuda.synchronize()
        
        t0 = time()
        for _ in range(100):
            o = old_model(x)
        torch.cuda.synchronize()
        t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 100) / (t1 - t0)
    print(f"Old model code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"Old model throughput: {throughput:.2f} points/sec")

    # Cleanup
    del old_model, x, o
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def backward_full_test():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
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
    x = torch.rand([batch_size, 3], dtype=torch.float32, device="cuda")
    model_new(x)
    
    # Warmup
    for _ in range(5):
        o = model_new(x)
        o.sum().backward()
    torch.cuda.synchronize()
    
    t0 = time()
    for _ in range(10):
        o = model_new(x)
        o.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"New model code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"New model throughput: {throughput:.2f} points/sec")

    del model_new
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    old_model = AMGSRN_old(opt).to(opt['device'])
    torch.cuda.reset_peak_memory_stats()
    old_model(x)
    
    # Warmup
    for _ in range(5):
        o = old_model(x)
        o.sum().backward()
    torch.cuda.synchronize()
    
    t0 = time()
    for _ in range(10):
        o = old_model(x)
        o.sum().backward()
    torch.cuda.synchronize()
    t1 = time()
    
    gb = torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
    torch.cuda.reset_peak_memory_stats()
    throughput = (batch_size * 10) / (t1 - t0)
    print(f"Old model code took {t1-t0:0.02f}s with {gb:0.02f} GB VRAM")
    print(f"Old model throughput: {throughput:.2f} points/sec")

    # Cleanup
    del old_model, x, o
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def train_one_step_test():
    print("======Train one step test======")
    opt = Options.get_default()
    opt['n_grids'] = 32
    opt['n_features'] = 2
    opt['feature_grid_shape'] = "64,64,64"
    opt['data_min']=0.0
    opt['data_max'] = 1.0
    opt['device'] = "cuda"
    opt['use_amp'] = use_amp
    model = AMGSRN(opt).to(opt['device'])
    model.train()
    iteration = 601
    scaler = torch.amp.GradScaler()

    x = torch.rand([2**17, 3], dtype=torch.float32, device="cuda")
    y = torch.rand([2**17, 1], dtype=torch.float32, device="cuda")
    
    optimizer = [
        torch.optim.Adam(
            model.get_model_parameters(), 
            lr=opt["lr"],
            betas=[opt['beta_1'], opt['beta_2']], eps = 10e-15
            ),
        torch.optim.Adam(
            model.get_transform_parameters(), 
            lr=opt['transform_lr'],
            betas=[opt['beta_1'], opt['beta_2']], eps = 10e-15
            )
    ]        
    scheduler = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=opt['iterations']),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[1], T_max=int(opt['iterations']/5))
    ] 
    
    optimizer[0].zero_grad()   
    
    x = x.to(opt['device'])
    y = y.to(opt['device'])
    
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt['use_amp']):
        with record_function("forward"):
            model_output = model.forward(x)

        with record_function("L_rec"):
            loss = torch.nn.functional.mse_loss(model_output, y, reduction='none')
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
                density_loss = torch.nn.functional.kl_div(
                    torch.log(density+1e-16), 
                    torch.log(target.detach()+1e-16), reduction='none', 
                    log_target=True)


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
    
# forward_encode_test()
# backward_encode_test()
# forward_density_test()
# backward_density_test()
# forward_full_test()
# backward_full_test()
train_one_step_test()
