import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# Set the style for a clean and modern look
plt.style.use('bmh')  # Use a valid style name

# Function to create charts for each dataset
def create_charts(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Extract dataset name from file name
    dataset_name = os.path.basename(file_path).split('_')[0]
    
    # Define a color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['Compressor'].unique())))
    
    # Create PSNR vs Compression Ratio chart
    plt.figure(figsize=(20, 14))
    
    for i, compressor in enumerate(df['Compressor'].unique()):
        data = df[df['Compressor'] == compressor]
        plt.scatter(data['Compression Ratio'], data['PSNR'], label=compressor, s=100, color=colors[i])
        plt.plot(data['Compression Ratio'], data['PSNR'], linewidth=3, color=colors[i])
    
    plt.xscale('log')
    plt.xlabel('Compression Ratio', fontsize=48)
    plt.ylabel('PSNR (dB)', fontsize=48)
    plt.title(f'Compression Quality vs Compression Ratio for {dataset_name}', fontsize=40, fontweight='bold')
    plt.legend(fontsize=36)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', which='major', labelsize=36)
    plt.gca().invert_xaxis()
    plt.gca().set_facecolor('#f8f8f8')
    plt.tight_layout()
    
    output_dir = os.path.join('Output', 'CompressionCharts')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_compression_quality_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create Compression Time vs Compression Ratio chart
    plt.figure(figsize=(20, 14))
    
    for i, compressor in enumerate(df['Compressor'].unique()):
        data = df[df['Compressor'] == compressor]
        plt.scatter(data['Compression Ratio'], data['Compression Time (s)'], label=compressor, s=100, color=colors[i])
        plt.plot(data['Compression Ratio'], data['Compression Time (s)'], linewidth=3, color=colors[i])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Compression Ratio', fontsize=36)
    plt.ylabel('Compression Time (s)', fontsize=36)
    plt.title(f'Compression Time vs Compression Ratio for {dataset_name}', fontsize=40, fontweight='bold')
    plt.legend(fontsize=36)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis='both', which='major', labelsize=36)
    plt.gca().invert_xaxis()
    plt.gca().set_facecolor('#f8f8f8')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_compression_time_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Get all _avg.csv files in the CompressionResults directory
csv_files = glob.glob(os.path.join('Output', 'CompressionResults', '*_avg.csv'))

# Create charts for each CSV file
for file_path in csv_files:
    create_charts(file_path)

print(f"High-resolution charts have been generated and saved in the Output/CompressionCharts directory.")
