import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# Set the style for a clean and modern look
plt.style.use('bmh')  # Use a valid style name

def plot_time_varying_compression(results_file):
    # Read the CSV file
    df = pd.read_csv(results_file)
    
    # Get unique error bounds
    error_bounds = df['Error Bound'].unique()
    
    for error_bound in error_bounds:
        # Filter the dataframe for the current error bound
        df_filtered = df[df['Error Bound'] == error_bound]
        
        # Sort by Timestep to ensure correct order
        df_filtered = df_filtered.sort_values('Timestep')
        
        # Create a new figure
        plt.figure(figsize=(12, 6))
        
        # Plot compression ratio for each compressor
        for compressor in df_filtered['Compressor'].unique():
            compressor_data = df_filtered[df_filtered['Compressor'] == compressor]
            plt.plot(compressor_data['Timestep'], compressor_data['Compression Ratio'], label=compressor, marker='o')
        
        plt.title(f'Compression Ratio over Time (Error Bound: {error_bound})')
        plt.xlabel('Timestep')
        plt.ylabel('Compression Ratio')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        output_file = f'time_varying_compression_eb_{error_bound}.png'
        plt.savefig(output_file)
        plt.close()
        
        print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot time-varying compression results')
    parser.add_argument('results_file', type=str, help='Path to the CSV file containing compression results')
    
    args = parser.parse_args()
    
    plot_time_varying_compression(args.results_file)
