import os
import matplotlib.pyplot as plt
import numpy as np
import zipfile

def get_model_size(model_path):
    # Check for compressed model first
    zip_path = os.path.join(model_path, 'compressed_model.zip')
    if os.path.exists(zip_path):
        return os.path.getsize(zip_path) / (1024 * 1024)  # Convert to MB
    
    # Fall back to uncompressed model
    model_path = os.path.join(model_path, 'model.ckpt')
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    
    return None

def get_psnr(model_path):
    log_path = os.path.join(model_path, 'test_log.txt')
    if not os.path.exists(log_path):
        return None
        
    with open(log_path, 'r') as f:
        content = f.read()
        
    # Find last occurrence of Mean PSNR
    lines = content.split('\n')
    for line in reversed(lines):
        if 'Mean PSNR:' in line:
            return float(line.split()[-1])
    return None

# Setup data structures
datasets = ['asteroid', 'mantle', 'mixing', 'supernova', 'isotropic', 'chameleon']
models = ['AMGSRN', 'fVSRN', 'NGP']
sizes = ['small', 'medium', 'large']
colors = ['blue', 'red', 'green']
markers = ['o', 's', '^']

# Set larger font sizes
plt.rcParams.update({'font.size': 20})

# Create figure with 3x2 subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
axes = axes.flatten()

# Process each dataset
for dataset_idx, dataset in enumerate(datasets):
    ax = axes[dataset_idx]
    
    # Process each model type
    for model_idx, model in enumerate(models):
        model_sizes = []
        psnrs = []
        
        # Process each size
        for size in sizes:
            model_name = f"{dataset}_{model}_{size}"
            model_path = os.path.join('SavedModels', model_name)
            
            if not os.path.exists(model_path):
                continue
                
            size_mb = get_model_size(model_path)
            psnr = get_psnr(model_path)
            
            if size_mb is not None and psnr is not None:
                model_sizes.append(size_mb)
                psnrs.append(psnr)
        
        if model_sizes:  # Only plot if we have data
            ax.semilogx(model_sizes, psnrs, color=colors[model_idx], marker=markers[model_idx], 
                       label=model, linestyle='-', markersize=8)
    
    ax.set_title(dataset.capitalize(), fontsize=22)
    ax.set_xlabel('Model Size (MB)', fontsize=20)
    ax.set_ylabel('PSNR (dB)', fontsize=20)
    ax.grid(True)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.savefig('baseline_results.png', dpi=300, bbox_inches='tight')
plt.close()
