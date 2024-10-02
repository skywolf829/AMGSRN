import os
import argparse
from AMGSRN.Models.options import Options
from AMGSRN.Models.AMGSRN import AMGSRN
from AMGSRN.Datasets.datasets import Dataset
from AMGSRN.train import train_model
from AMGSRN.test import model_reconstruction_chunked, test_psnr_chunked
from AMGSRN.Models.models import save_model, load_model
import torch 
import time
from Other.utility_functions import PSNR, tensor_to_cdf, create_path

def parse_args():
    parser = argparse.ArgumentParser(description="AMGSRN Training and Reconstruction")
    parser.add_argument("-i", "--input", type=str, help="Path to input data file")
    parser.add_argument("-z", "--save", type=str, help="Path to save trained model")
    parser.add_argument("-o", "--output", type=str, help="Path to reconstruct data")
    parser.add_argument("-F", "--feature-grid", type=int, nargs=3, default=[8, 8, 8], help="Feature grid size")
    parser.add_argument("-M", "--num-grids", type=int, default=64, help="Number of grids")
    parser.add_argument("-E", "--ensemble", type=int, nargs=3, default=[1, 1, 1], help="Ensemble dimensions")
    parser.add_argument("-s", "--stats", action="store_true", help="Display statistics")
    return parser.parse_args()

def main():
    args = parse_args()
    
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load default options
    opt = Options.get_default()

    # Update options based on command line arguments
    if args.input:
        opt['data'] = args.input
    if args.save:
        opt['save_name'] = os.path.basename(args.save)
    opt['feature_grid_shape'] = ','.join(map(str, args.feature_grid))
    opt['n_grids'] = args.num_grids
    if args.ensemble != [1, 1, 1]:
        opt['ensemble'] = True
        opt['ensemble_grid'] = ','.join(map(str, args.ensemble))

    # Validate input combinations
    if not (args.input or args.save or args.output):
        raise ValueError("At least one of input, save, or output must be specified")
    if args.output and not (args.input or args.save):
        raise ValueError("output requires either input or save to be specified")

    # Train or load model
    if args.input:
        model = AMGSRN(opt).to(opt['device'])
        model.train()
        dataset = Dataset(opt)
        start_time = time.time()
        train_model(model, dataset, opt)
        end_time = time.time()
        training_time = end_time - start_time
        max_vram_used = torch.cuda.max_memory_allocated() / (1024 * 1024)
    if args.save:
        if args.input:
            save_model(model, opt)
        else:
            model = load_model(opt, opt['device'])
        model = model.to(opt['device'])
    
    if args.output:
        if args.stats and not args.input:
            raise ValueError("stats requires input to be specified")
        
        # Reconstruct data
        total_reconstruction_time = 0
        max_vram_used_reconstruction = 0
        for timestep in range(opt['n_timesteps']):
            start_time = time.time()
            with torch.no_grad(), torch.autocast(device=opt['device'], dtype=torch.float16):
                output = model_reconstruction_chunked(model, opt, timestep)
            total_reconstruction_time += time.time() - start_time
            max_vram_used_reconstruction = max(max_vram_used_reconstruction, torch.cuda.max_memory_allocated() / (1024 * 1024))
            create_path(os.path.join(args.output, "Reconstruction"))
            tensor_to_cdf(output, 
                os.path.join(args.output, 
                             opt['save_name']+"_timestep"+
                             (f"_{timestep}" if timestep > 0 else "") + ".nc"))

        if args.stats:
            if args.input:
                # Print training stats
                print(f"Training completed in {training_time:.2f} seconds")
                print(f"Max VRAM used during training: {max_vram_used:.2f} MB")
                # Print reconstruction quality stats
                for timestep in range(opt['n_timesteps']):
                    with torch.no_grad(), torch.autocast(device=opt['device'], dtype=torch.float16):
                        psnr = test_psnr_chunked(model, opt, timestep)
                    if(opt['n_timesteps'] > 1):
                        print(f"PSNR at timestep {timestep}: {psnr:.2f} dB")
                    else:
                        print(f"PSNR: {psnr:.2f} dB")
                
            # Print reconstruction stats
            print(f"Reconstruction completed in {total_reconstruction_time:.2f} seconds")
            print(f"Max VRAM used during reconstruction: {max_vram_used_reconstruction:.2f} MB")

if __name__ == "__main__":
    main()


