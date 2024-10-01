import argparse
import os
import subprocess
import numpy as np
import pandas as pd
from AMGSRN.Other.utility_functions import nc_to_np
import time
import multiprocessing
from glob import glob

def compress_decompress(compressor, input_file, output_file, error_bound, data_shape):
    dims_1 = f'-{len(data_shape)}'
    dims_2 = [str(i) for i in data_shape]
    if compressor == 'sz3':
        subprocess.run(['sz3', '-f', '-i', input_file, '-o', f'{output_file}.sz3.out', '-z', f'{output_file}.sz3',
                         '-M', 'ABS', '-A', f"{error_bound:.20f}", dims_1, *dims_2], check=True)
    elif compressor == 'tthresh':
        subprocess.run(['tthresh', '-i', input_file, '-c', f'{output_file}.tthresh', '-o', f'{output_file}.tthresh.out',
                        '-t', 'float','-e', f"{error_bound:.20f}", '-s', *dims_2], check=True)
    elif compressor == 'zfp':
        subprocess.run(['zfp', '-f', '-i', input_file, '-z', f'{output_file}.zfp', '-a', f"{error_bound:.20f}", dims_1, *dims_2], check=True)
        subprocess.run(['zfp', '-f', '-z', f'{output_file}.zfp', '-o', f'{output_file}.zfp.out', '-a', f"{error_bound:.20f}", dims_1, *dims_2], check=True)

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = np.max(original)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def process_compression(file_path, compressor, error_bound, data, data_shape, raw_file):
    output_file = f"{file_path}_{compressor}_{error_bound}"
    
    start_time = time.time()
    compress_decompress(compressor, raw_file, output_file, float(error_bound), data_shape)
    compression_time = time.time() - start_time
    
    original_size = os.path.getsize(raw_file)
    compressed_size = os.path.getsize(f"{output_file}.{compressor}")
    compression_ratio = original_size / compressed_size
    
    decompressed_data = np.fromfile(f"{output_file}.{compressor}.out", dtype=data.dtype).reshape(data.shape)
    psnr = calculate_psnr(data, decompressed_data)
    
    result = {
        'Timestep': os.path.basename(file_path),
        'Compressor': compressor,
        'Error Bound': error_bound,
        'Compressed Size (bytes)': compressed_size,
        'Compression Ratio': compression_ratio,
        'PSNR': psnr,
        'Compression Time (s)': compression_time
    }
    
    os.remove(f"{output_file}.{compressor}")
    os.remove(f"{output_file}.{compressor}.out")
    
    return result

def process_file(file_path, compressors, error_bounds):
    print(f"Processing {file_path}")
    
    data, _ = nc_to_np(file_path)
    data = np.squeeze(data).astype(np.float32)
    data_shape = data.shape
    
    raw_file = f"{file_path}.raw"
    data.tofile(raw_file)
    
    pool = multiprocessing.Pool(4)
    
    tasks = [(file_path, compressor, error_bound, data, data_shape, raw_file) 
             for compressor in compressors for error_bound in error_bounds]
    
    try:
        results = pool.starmap(process_compression, tasks)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Terminating all jobs...")
        pool.terminate()
        return []
    finally:
        pool.close()
        pool.join()
        if os.path.exists(raw_file):
            os.remove(raw_file)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test compressors for time-varying dataset or single volume')
    parser.add_argument('input_path', type=str, help='Path to the dataset file or directory')
    args = parser.parse_args()

    compressors = ['sz3', 'zfp']  # tthresh is slow
    error_bounds = np.logspace(-3, -0.39794, num=10)  # 10 points from 0.001 to 0.4

    all_results = []

    if os.path.isdir(args.input_path):
        file_list = glob(os.path.join(args.input_path, '*.h5')) + glob(os.path.join(args.input_path, '*.nc'))
        for file_path in file_list:
            all_results.extend(process_file(file_path, compressors, error_bounds))
    elif os.path.isfile(args.input_path) and (args.input_path.endswith('.h5') or args.input_path.endswith('.nc')):
        all_results = process_file(args.input_path, compressors, error_bounds)
    else:
        print("Invalid input path. Please provide a valid file or directory.")
        return

    if all_results:
        df = pd.DataFrame(all_results)
        df_avg = df.groupby(['Compressor', 'Error Bound']).agg({
            'Compressed Size (bytes)': 'mean',
            'Compression Ratio': 'mean',
            'PSNR': 'mean',
            'Compression Time (s)': 'mean'
        }).reset_index()

        dataset_name = os.path.basename(args.input_path)
        df.to_csv(f"{dataset_name}_compression_results.csv", index=False)
        df_avg.to_csv(f"{dataset_name}_compression_results_avg.csv", index=False)

        print(f"Results saved to {dataset_name}_compression_results.csv and {dataset_name}_compression_results_avg.csv")
    else:
        print("No results to save due to early termination or invalid input.")

if __name__ == "__main__":
    main()
