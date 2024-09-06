import vtk
import numpy as np
import argparse
from vtkmodules.util.numpy_support import vtk_to_numpy
import netCDF4 as nc

def load_vtm_file(file_path):
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def resample_to_grid(multiblock):
    # Create a bounding box for the entire dataset
    bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
    for i in range(multiblock.GetNumberOfBlocks()):
        block = multiblock.GetBlock(i)
        if isinstance(block, vtk.vtkMultiPieceDataSet):
            for j in range(block.GetNumberOfPieces()):
                piece = block.GetPiece(j)
                piece_bounds = piece.GetBounds()
                bounds[0] = min(bounds[0], piece_bounds[0])
                bounds[1] = max(bounds[1], piece_bounds[1])
                bounds[2] = min(bounds[2], piece_bounds[2])
                bounds[3] = max(bounds[3], piece_bounds[3])
                bounds[4] = min(bounds[4], piece_bounds[4])
                bounds[5] = max(bounds[5], piece_bounds[5])
        else:
            block_bounds = block.GetBounds()
            bounds[0] = min(bounds[0], block_bounds[0])
            bounds[1] = max(bounds[1], block_bounds[1])
            bounds[2] = min(bounds[2], block_bounds[2])
            bounds[3] = max(bounds[3], block_bounds[3])
            bounds[4] = min(bounds[4], block_bounds[4])
            bounds[5] = max(bounds[5], block_bounds[5])

    # Calculate dimensions based on aspect ratio
    x_size = bounds[1] - bounds[0]
    y_size = bounds[3] - bounds[2]
    z_size = bounds[5] - bounds[4]
    max_size = max(x_size, y_size, z_size)
    dimensions = [
        max(int(1024 * x_size / max_size), 1),
        max(int(1024 * y_size / max_size), 1),
        max(int(1024 * z_size / max_size), 1)
    ]

    # Create a vtkImageData as the output
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dimensions)
    image_data.SetOrigin(bounds[0], bounds[2], bounds[4])
    image_data.SetSpacing(
        x_size / (dimensions[0] - 1),
        y_size / (dimensions[1] - 1),
        z_size / (dimensions[2] - 1)
    )
    # Create a probe filter to resample the data
    probe_filter = vtk.vtkCompositeDataProbeFilter()
    probe_filter.SetSourceData(multiblock)
    probe_filter.SetInputData(image_data)
    probe_filter.Update()
    return probe_filter.GetOutput()

def get_volume_array(image_data):
    point_data = image_data.GetPointData()
    num_arrays = point_data.GetNumberOfArrays()
    
    volume_arrays = {}
    for i in range(num_arrays):
        array = point_data.GetArray(i)
        array_name = point_data.GetArrayName(i)
        numpy_array = vtk_to_numpy(array)
        dimensions = image_data.GetDimensions()
        reshaped_array = numpy_array.reshape(dimensions, order='F')
        volume_arrays[array_name] = reshaped_array
    
    return volume_arrays

def save_to_netcdf(volume_arrays, output_file):
    with nc.Dataset(output_file, 'w') as dataset:
        # Create dimensions
        shape = next(iter(volume_arrays.values())).shape
        dataset.createDimension('x', shape[0])
        dataset.createDimension('y', shape[1])
        dataset.createDimension('z', shape[2])

        # Create variables and store data
        for name, array in volume_arrays.items():
            var = dataset.createVariable(name, array.dtype, ('x', 'y', 'z'))
            var[:] = array

def main():
    parser = argparse.ArgumentParser(description='Process a VTM file.')
    parser.add_argument('file_path', type=str, help='Path to the VTM file')
    parser.add_argument('--output', type=str, default='output.nc', help='Output NetCDF file name')
    args = parser.parse_args()

    try:
        multiblock = load_vtm_file(args.file_path)
        
        # Resample to grid
        resampled_data = resample_to_grid(multiblock)
        
        # Get volume data as numpy arrays
        volume_arrays = get_volume_array(resampled_data)
        
        # Save to NetCDF file
        save_to_netcdf(volume_arrays, args.output)
        
        print(f"Data saved to {args.output}")
        
        for name, array in volume_arrays.items():
            print(f"Array '{name}' shape: {array.shape}")
            print(f"Min value: {array.min()}, Max value: {array.max()}")
            print(f"Value at (0, 0, 0): {array[0, 0, 0]}")
            print()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()