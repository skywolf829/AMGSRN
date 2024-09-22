import numpy as np
from AMGSRN.Datasets.sample_vtk import vtk_to_sampled_grid
import os
from pathlib import Path
from tqdm import tqdm

def process_wildfire_data():
    input_dir = Path(r"C:\Users\Sky\Documents\GitHub\AMGSRN\Data\fire").resolve()
    output_dir = Path(r"C:\Users\Sky\Documents\GitHub\AMGSRN\Data\fire_npy").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    field_stats = {}

    for vts_file in tqdm(input_dir.glob("*.vts")):
        # Check if the vts file has already been processed
        vts_stem = vts_file.stem
        existing_files = list(output_dir.glob(f"{vts_stem}_*.npy"))
        if existing_files:
            print(f"Skipping {vts_stem}: already processed")
            continue

        try:
            sampled_data = vtk_to_sampled_grid(str(vts_file))
            
            for field_name, array in sampled_data.items():
                # Save numpy array
                output_file = output_dir / f"{vts_stem}_{field_name}.npy"
                np.save(output_file, array)
                
                # Update field statistics
                if field_name not in field_stats:
                    field_stats[field_name] = {'min': np.inf, 'max': -np.inf}
                field_stats[field_name]['min'] = min(field_stats[field_name]['min'], np.min(array))
                field_stats[field_name]['max'] = max(field_stats[field_name]['max'], np.max(array))
        
        except Exception as e:
            print(f"Error processing {vts_file}: {str(e)}")

    # Print field statistics
    print("\nField statistics:")
    for field_name, stats in field_stats.items():
        print(f"{field_name}: min = {stats['min']}, max = {stats['max']}")

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

def main():
    process_wildfire_data()

if __name__ == "__main__":
    main()