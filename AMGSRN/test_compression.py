
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from AMGSRN.Models.models import load_model
from AMGSRN.Models.options import load_options
import argparse
import os
import torch

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def compress_and_test(model, test_model, input_data, ground_truth, error_bounds, error_mode):
    results = []
    
    params = {
        '_scales': model._scales.cpu().numpy().astype(np.float32),
        '_rotations': model._rotations.cpu().numpy().astype(np.float32),
        'translations': model.translations.cpu().numpy().astype(np.float32),
        'feature_grids': model.feature_grids.cpu().numpy().astype(np.float32),
        'decoder': [p.cpu().numpy().astype(np.float32) for p in model.decoder.parameters()]
    }
    for bound in error_bounds:
        try:
            # Save all parameters to separate raw files
            
            total_original_size = 0
            total_compressed_size = 0
            
            for name, param in params.items():
                if name != 'decoder' and name != 'feature_grids':
                    param.tofile(f"temp_{name}.raw")
                    dims = ' '.join(str(d) for d in param.shape)
                    dim_flag = f"-{len(param.shape)} {dims}"
                    
                    # Compress using SZ3
                    command = f"sz3 -f -z temp_{name}.sz -i temp_{name}.raw -o temp_{name}.sz.out -M {error_mode} {bound} {dim_flag}"
                    print(command)
                    subprocess.run(command, check=True)
                    
                    # Get compression ratio
                    original_size = os.path.getsize(f"temp_{name}.raw")
                    compressed_size = os.path.getsize(f"temp_{name}.sz")
                    total_original_size += original_size
                    total_compressed_size += compressed_size
                elif name == 'feature_grids':
                    for i in range(param.shape[1]):
                        param[:,i].tofile(f"temp_feature_grids_{i}.raw")
                        dims = ' '.join(str(d) for d in param[:,i].shape)
                        dim_flag = f"-{len(param[:,i].shape)} {dims}"                
                        # Compress using SZ3
                        command = f"sz3 -f -z temp_feature_grids_{i}.sz -i temp_feature_grids_{i}.raw -o temp_feature_grids_{i}.sz.out -M {error_mode} {bound} {dim_flag}"   

                        print(command)
                        subprocess.run(command, check=True)
                elif name == 'decoder':
                    for i, layer_param in enumerate(param):
                        layer_param.tofile(f"temp_decoder_{i}.raw")
                        dims = ' '.join(str(d) for d in layer_param.shape)
                        dim_flag = f"-{len(layer_param.shape)} {dims}"
                        
                        # Compress using SZ3
                        command = f"sz3 -f -z temp_decoder_{i}.sz -i temp_decoder_{i}.raw -o temp_decoder_{i}.sz.out -M {error_mode} {bound} {dim_flag}"
                        print(command)
                        subprocess.run(command, check=True)
                        
                        # Get compression ratio
                        original_size = os.path.getsize(f"temp_decoder_{i}.raw")
                        compressed_size = os.path.getsize(f"temp_decoder_{i}.sz")
                        total_original_size += original_size
                        total_compressed_size += compressed_size
            
            compression_ratio = total_original_size / total_compressed_size
            
            # Decompress and load parameters
            for name, param in params.items():
                pass
                # if name != 'decoder' and name != 'feature_grids':
                #     decompressed = np.fromfile(f"temp_{name}.sz.out", dtype=np.float32).reshape(params[name].shape)
                #     setattr(test_model, name, torch.nn.Parameter(torch.from_numpy(decompressed).to(model.feature_grids.device)))
                if name == 'feature_grids':
                    for i in range(params['feature_grids'].shape[1]):
                        decompressed = np.fromfile(f"temp_feature_grids_{i}.sz.out", dtype=np.float32).reshape(params['feature_grids'][:,i].shape)
                        test_model.feature_grids[:,i] = torch.from_numpy(decompressed).to(model.feature_grids.device)
                    print(test_model.feature_grids[0,0,0])

                # else:
                #     decompressed_decoder = []
                #     for i, layer_param in enumerate(param):
                #         decompressed = np.fromfile(f"temp_decoder_{i}.sz.out", dtype=np.float32).reshape(params['decoder'][i].shape)
                #         decompressed_decoder.append(decompressed)
                    
                #     for param, decompressed in zip(test_model.decoder.parameters(), decompressed_decoder):
                #         param.data = torch.from_numpy(decompressed).to(test_model.feature_grids.device)
            
            # Test error
            with torch.no_grad():
                output = test_model(input_data)
            mse = torch.mean((output - ground_truth) ** 2).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            
            results.append((bound, compression_ratio, psnr))
            
            
        
        except subprocess.CalledProcessError as e:
            print(f"Compression failed for error bound {bound}. Skipping this bound.")
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for error bound {bound}. Skipping this bound.")
            print(f"Error: {e}")
            continue
    # Clean up temporary files
    for name in params.keys():
        try:
            if name != 'decoder' and name != "feature_grids":
                os.remove(f"temp_{name}.raw")
                os.remove(f"temp_{name}.sz")
                os.remove(f"temp_{name}.sz.out")
            elif name == 'feature_grids':  
                for i in range(model.feature_grids.shape[1]):
                    os.remove(f"temp_feature_grids_{i}.raw")
                    os.remove(f"temp_feature_grids_{i}.sz")
                    os.remove(f"temp_feature_grids_{i}.sz.out")
            else:
                for i in range(len(params['decoder'])):
                    os.remove(f"temp_decoder_{i}.raw")
                    os.remove(f"temp_decoder_{i}.sz")
                    os.remove(f"temp_decoder_{i}.sz.out")
        except:
            pass
    return results

def compress_and_test_fpzip(model, test_model, input_data, ground_truth, precisions):
    results = []
    
    params = {
        '_scales': model._scales.cpu().numpy().astype(np.float32),
        '_rotations': model._rotations.cpu().numpy().astype(np.float32),
        'translations': model.translations.cpu().numpy().astype(np.float32),
        'feature_grids': model.feature_grids.cpu().numpy().astype(np.float32),
        'decoder': [p.cpu().numpy().astype(np.float32) for p in model.decoder.parameters()]
    }
    for precision in precisions:
        try:
            total_original_size = 0
            total_compressed_size = 0
            
            for name, param in params.items():
                if name != 'decoder' and name != 'feature_grids':
                    param.tofile(f"temp_{name}.raw")
                    dims = ' '.join(str(d) for d in param.shape)
                    dim_flag = f"-{len(param.shape)} {dims}"
                    
                    # Compress using fpzip
                    command = f"fpzip -t float -p {precision} {dim_flag} -i temp_{name}.raw -o temp_{name}.fpz"
                    print(command)
                    subprocess.run(command, check=True)
                    
                    # Get compression ratio
                    original_size = os.path.getsize(f"temp_{name}.raw")
                    compressed_size = os.path.getsize(f"temp_{name}.fpz")
                    total_original_size += original_size
                    total_compressed_size += compressed_size
                elif name == 'feature_grids':
                    for i in range(param.shape[1]):
                        param[:,i].tofile(f"temp_feature_grids_{i}.raw")
                        dims = ' '.join(str(d) for d in param[:,i].shape)
                        dim_flag = f"-{len(param[:,i].shape)} {dims}"
                        command = f"fpzip -t float -p {precision} {dim_flag} -i temp_feature_grids_{i}.raw -o temp_feature_grids_{i}.fpz"
                        print(command)
                        subprocess.run(command, check=True)
                        
                        original_size = os.path.getsize(f"temp_feature_grids_{i}.raw")
                        compressed_size = os.path.getsize(f"temp_feature_grids_{i}.fpz")
                        total_original_size += original_size
                        total_compressed_size += compressed_size
                elif name == 'decoder':
                    for i, layer_param in enumerate(param):
                        layer_param.tofile(f"temp_decoder_{i}.raw")
                        dims = ' '.join(str(d) for d in layer_param.shape)
                        dim_flag = f"-{len(layer_param.shape)} {dims}"
                        
                        command = f"fpzip -t float -p {precision} {dim_flag} -i temp_decoder_{i}.raw -o temp_decoder_{i}.fpz"
                        print(command)
                        subprocess.run(command, check=True)
                        
                        original_size = os.path.getsize(f"temp_decoder_{i}.raw")
                        compressed_size = os.path.getsize(f"temp_decoder_{i}.fpz")
                        total_original_size += original_size
                        total_compressed_size += compressed_size
            
            compression_ratio = total_original_size / total_compressed_size
            
            # Decompress and load parameters
            for name, param in params.items():
                if name == 'feature_grids':
                    for i in range(params['feature_grids'].shape[1]):
                        command = f"fpzip -d -t float -i temp_feature_grids_{i}.fpz -o temp_feature_grids_{i}.out"
                        subprocess.run(command, check=True)
                        decompressed = np.fromfile(f"temp_feature_grids_{i}.out", dtype=np.float32).reshape(params['feature_grids'][:,i].shape)
                        test_model.feature_grids[:,i] = torch.from_numpy(decompressed).to(model.feature_grids.device)
                    print(test_model.feature_grids[0,0,0])
            
            # Test error
            with torch.no_grad():
                output = test_model(input_data)
            mse = torch.mean((output - ground_truth) ** 2).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            
            results.append((precision, compression_ratio, psnr))
        
        except subprocess.CalledProcessError as e:
            print(f"Compression failed for precision {precision}. Skipping this precision.")
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for precision {precision}. Skipping this precision.")
            print(f"Error: {e}")
            continue
    
    # Clean up temporary files
    for name in params.keys():
        try:
            if name != 'decoder' and name != "feature_grids":
                os.remove(f"temp_{name}.raw")
                os.remove(f"temp_{name}.fpz")
                os.remove(f"temp_{name}.out")
            elif name == 'feature_grids':  
                for i in range(model.feature_grids.shape[1]):
                    os.remove(f"temp_feature_grids_{i}.raw")
                    os.remove(f"temp_feature_grids_{i}.fpz")
                    os.remove(f"temp_feature_grids_{i}.out")
            else:
                print(len(param))
                for i in range(len(param)):
                    os.remove(f"temp_decoder_{i}.raw")
                    os.remove(f"temp_decoder_{i}.fpz")
                    os.remove(f"temp_decoder_{i}.out")
        except:
            pass
    return results



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test compression of AMGSRN model parameters')
    parser.add_argument('--model_name', type=str, default='supernova_large_new', help='Name of the model to load')
    args = parser.parse_args()

    # Load model
    opt = load_options(os.path.join(save_folder, args.model_name))
    model = load_model(opt, "cuda")
    model = model.to("cuda")
    model.eval()
    test_model = load_model(opt, "cuda")
    test_model = test_model.to("cuda")
    test_model.eval()

    print(model.feature_grids[0,0,0])
    # Generate random input points and ground truth
    input_data = torch.rand(1000000, 3, device="cuda")*2 -1 
    with torch.no_grad():
        ground_truth = model(input_data)

    # Define error bounds to test
    abs_error_bounds = [0.075, 0.05, 0.04, 0.02, 0.01]
    rel_error_bounds = []

    # Test compression for transform parameters with ABS error
    with torch.no_grad():
        abs_results = compress_and_test(model, test_model, input_data, ground_truth, abs_error_bounds, "ABS")

    # Test compression for transform parameters with REL error
    with torch.no_grad():
        rel_results = compress_and_test(model, test_model, input_data, ground_truth, rel_error_bounds, "REL")

    # Define precisions for fpzip
    fpzip_precisions = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

    # Test compression using fpzip
    with torch.no_grad():
        fpzip_results = compress_and_test_fpzip(model, test_model, input_data, ground_truth, fpzip_precisions)

    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.semilogx([r[1] for r in abs_results], [r[2] for r in abs_results], 'o-', label='ABS')
    plt.semilogx([r[1] for r in rel_results], [r[2] for r in rel_results], 'o-', label='REL')
    plt.plot([r[1] for r in fpzip_results], [r[2] for r in fpzip_results], 'o-', label='FPZIP')
    
    plt.xlabel("Compression Ratio")
    plt.ylabel("PSNR (dB)")
    plt.title("Compression Results")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("compression_results.png")

    # Print results
    print("ABS Error Results:")
    for bound, ratio, psnr in abs_results:
        print(f"Error Bound: {bound}, Compression Ratio: {ratio:.2f}, PSNR: {psnr:.2f}")

    print("\nREL Error Results:")
    for bound, ratio, psnr in rel_results:
        print(f"Error Bound: {bound}, Compression Ratio: {ratio:.2f}, PSNR: {psnr:.2f}")

    print("\nFPZIP Results:")
    for precision, ratio, psnr in fpzip_results:
        print(f"Precision: {precision}, Compression Ratio: {ratio:.2f}, PSNR: {psnr:.2f}")


if __name__ == "__main__":
    main()
