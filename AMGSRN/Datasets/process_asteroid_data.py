import numpy as np
from AMGSRN.Datasets.sample_vtk import vtk_to_sampled_grid
import os
import tarfile
import glob
from pathlib import Path
from tqdm import tqdm
import tempfile
from AMGSRN.Other.utility_functions import npy_to_cdf

def process_asteroid_data():
    input_dir = Path(r"C:\Users\Sky\Documents\GitHub\AMGSRN\Data\asteroid_compressed").resolve()
    output_dir = Path(r"C:\Users\Sky\Documents\GitHub\AMGSRN\Data\asteroid_npy").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    field_stats = {}

    for tar_file in tqdm(glob.glob(str(input_dir / "*.tar"))):
        # Check if the tar file has already been processed
        tar_stem = Path(tar_file).stem
        existing_files = list(output_dir.glob(f"{tar_stem}_*.npy"))
        if existing_files:
            print(f"Skipping {tar_stem}: already processed")
            continue
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(tar_file, "r") as tar:
                # Extract all files from the tar archive to the temp directory
                tar.extractall(path=temp_dir)
            
            # Find all .vtm files in the extracted directory
            vtk_files = list(Path(temp_dir).glob('**/*.vtm'))
            
            for vtk_file in vtk_files:
                vtk_path = vtk_file.resolve()
                try:
                    sampled_data = vtk_to_sampled_grid(str(vtk_path))
                    
                    for field_name, array in sampled_data.items():
                        # Save numpy array
                        output_file = output_dir / f"{Path(tar_file).stem}_{field_name}.npy"
                        np.save(output_file, array)
                        
                
                except Exception as e:
                    print(f"Error processing {vtk_path}: {str(e)}")
            
        # The temporary directory and its contents are automatically cleaned up when exiting the context manager

    # Print field statistics
    # Find global min/max values for each field
    print("Finding global min/max values for each field...")
    global_field_stats = {}
    for npy_file in tqdm(list(output_dir.glob("*.npy"))):
        field_name = npy_file.stem.split('_')[-1]
        data = np.load(npy_file)
        
        if field_name not in global_field_stats:
            global_field_stats[field_name] = {'min': np.inf, 'max': -np.inf}
        
        global_field_stats[field_name]['min'] = min(global_field_stats[field_name]['min'], np.min(data))
        global_field_stats[field_name]['max'] = max(global_field_stats[field_name]['max'], np.max(data))

    # Print global field statistics
    print("\nGlobal field statistics:")
    for field_name, stats in global_field_stats.items():
        print(f"{field_name}: min = {stats['min']}, max = {stats['max']}")

    # Update field_stats with global values
    field_stats = global_field_stats
    
    # Rescale the data to 0.0-1.0 range
    print("Rescaling data to 0.0-1.0 range...")
    for npy_file in tqdm(output_dir.glob("*.npy")):
        field_name = npy_file.stem.split('_')[-1]
        if field_name in field_stats:
            data = np.load(npy_file)
            min_val = field_stats[field_name]['min']
            max_val = field_stats[field_name]['max']
            
            # Avoid division by zero
            if min_val != max_val:
                scaled_data = (data - min_val) / (max_val - min_val)
            else:
                scaled_data = np.zeros_like(data)
            
            np.save(npy_file, scaled_data)
            print(f"Rescaled and saved: {npy_file}")
        else:
            print(f"Warning: No statistics found for {field_name}. Skipping {npy_file}")

    print("Rescaling complete.")
    
    for npy_file in tqdm(output_dir.glob("*.npy")):
        data = np.load(npy_file)
        npy_to_cdf(data[None,None,...], npy_file.stem + ".nc")

def main():
    process_asteroid_data()

if __name__ == "__main__":
    main()
    