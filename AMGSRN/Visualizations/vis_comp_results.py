import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# Setup data structures
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
markers = ['o', 's', '^', 'D', 'v', '<']

# Set larger font sizes
plt.rcParams.update({'font.size': 20})

# Create figure with 3x2 subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
axes = axes.flatten()

# Get all _avg.csv files in the CompressionResults directory
csv_files = glob.glob(os.path.join('Output', 'CompressionResults', '*_avg.csv'))

# Process each CSV file
for dataset_idx, file_path in enumerate(csv_files):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Extract dataset name from file name, remove extension and trailing numbers
    dataset_name = os.path.basename(file_path).split('_')[0].split('.')[0].rstrip('0123456789')
    
    # PSNR vs Compression Ratio chart (left column)
    ax_psnr = axes[dataset_idx * 2]
    for comp_idx, compressor in enumerate(df['Compressor'].unique()):
        data = df[df['Compressor'] == compressor]
        ax_psnr.semilogx(data['Compression Ratio'], data['PSNR'], 
                        color=colors[comp_idx], marker=markers[comp_idx],
                        label=compressor, linestyle='-', markersize=8)
    
    ax_psnr.set_title(f"{dataset_name.capitalize()} - PSNR", fontsize=22)
    ax_psnr.set_xlabel('Compression Ratio', fontsize=20)
    ax_psnr.set_ylabel('PSNR (dB)', fontsize=20)
    ax_psnr.grid(True)
    ax_psnr.legend(fontsize=18)
    ax_psnr.tick_params(axis='both', which='major', labelsize=18)
    ax_psnr.invert_xaxis()

    # Compression Time vs Compression Ratio chart (right column)
    ax_time = axes[dataset_idx * 2 + 1]
    for comp_idx, compressor in enumerate(df['Compressor'].unique()):
        data = df[df['Compressor'] == compressor]
        ax_time.loglog(data['Compression Ratio'], data['Compression Time (s)'],
                      color=colors[comp_idx], marker=markers[comp_idx],
                      label=compressor, linestyle='-', markersize=8)
    
    ax_time.set_title(f"{dataset_name.capitalize()} - Time", fontsize=22)
    ax_time.set_xlabel('Compression Ratio', fontsize=20)
    ax_time.set_ylabel('Compression Time (s)', fontsize=20)
    ax_time.grid(True)
    ax_time.legend(fontsize=18)
    ax_time.tick_params(axis='both', which='major', labelsize=18)
    ax_time.invert_xaxis()

plt.tight_layout()

# Create output directory and save figure
output_dir = os.path.join('Output', 'CompressionCharts')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'compression_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"High-resolution comparison chart has been generated and saved in the Output/CompressionCharts directory.")
