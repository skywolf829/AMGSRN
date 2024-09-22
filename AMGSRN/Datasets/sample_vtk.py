import vtk
import numpy as np
import argparse
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import time
from AMGSRN.Other.utility_functions import tensor_to_cdf, make_coord_grid
import torch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def nc_to_sampled_grid(filename, min_grid_size=512):
    # Load the NetCDF file
    ds = xr.open_dataset(filename)
    
    # Check if 'temperature' variable exists
    if 'temperature' not in ds.data_vars:
        raise ValueError("'temperature' variable not found in the NetCDF file")
    
    # Get the temperature data
    temp_data = ds['temperature']
    return {'temperature': temp_data.values}

def load_vtm_file(file_path):
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def load_vts_file(file_path):
    pass

def create_probe_filter(multiblock):
    probe_filter = vtk.vtkCompositeDataProbeFilter()
    probe_filter.SetSourceData(multiblock)
    return probe_filter

def sample_points(multiblock, points):
    probe_filter = create_probe_filter(multiblock)

    # Convert numpy array to vtkPoints
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))

    # Create a polydata with these points
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Use the probe filter
    probe_filter.SetInputData(polydata)
    probe_filter.Update()

    # Get the probed data
    probed_data = probe_filter.GetOutput()
    point_data = probed_data.GetPointData()

    # Extract all arrays from the point data
    num_arrays = point_data.GetNumberOfArrays()
    result = {}
    for i in range(num_arrays):
        array = point_data.GetArray(i)
        array_name = point_data.GetArrayName(i)
        numpy_array = vtk_to_numpy(array)
        result[array_name] = numpy_array

    return result

def sample_points_v2(multiblock, points, sample_shape):
    # Assuming points is a 3D numpy array of shape (nx, ny, nz, 3)
    nx, ny, nz = sample_shape

    # Create a vtkStructuredGrid

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, nz)

    # Flatten the points array and convert to vtkPoints
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    grid.SetPoints(vtk_points)

    # Create and set up the probe filter
    probe_filter = vtk.vtkCompositeDataProbeFilter()
    probe_filter.SetSourceData(multiblock)
    probe_filter.SetInputData(grid)
    probe_filter.Update()


    # Get the probed data
    probed_data = probe_filter.GetOutput()
    point_data = probed_data.GetPointData()

    # Extract all arrays from the point data
    result = {}
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        array_name = point_data.GetArrayName(i)
        numpy_array = vtk_to_numpy(array)
        result[array_name] = numpy_array.reshape((nx, ny, nz, -1))

    return result

def resample_vtm(multiblock, sample_shape, bounds):
    # Create the resample filter
    # Convert to image data
    to_image = vtk.vtkResampleToImage()
    to_image.SetInputDataObject(multiblock)
    #to_image.SetInputConnection(geometry_filter.GetOutputPort())
    to_image.SetSamplingDimensions(sample_shape[0], sample_shape[1], sample_shape[2])
    to_image.SetUseInputBounds(False)  # Use the input bounds for sampling
    to_image.SetSamplingBounds(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
    to_image.Update()
    
    # Get the output
    output = to_image.GetOutput()
    point_data = output.GetPointData()
    
    # Extract all arrays from the point data
    result = {}
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        array_name = point_data.GetArrayName(i)
        if array_name in ["snd"]:
            numpy_array = vtk_to_numpy(array)
            result[array_name] = numpy_array
    
    return result

def vtk_to_sampled_grid(filename, min_grid_size=512):
    # Time the data loading
    bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]

    if filename.lower().endswith('.vtm'):
        data = load_vtm_file(filename)
        # Get the bounds of the multiblock dataset
        for i in range(data.GetNumberOfBlocks()):
            block = data.GetBlock(i)
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
    elif filename.lower().endswith('.vts'):
        data = load_vts_file(filename)
        bounds = data.GetBounds()
    elif filename.lower().endswith(".nc"):
        return nc_to_sampled_grid(filename, min_grid_size)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    
    # Calculate the aspect ratio based on the bounds
    x_range = bounds[1] - bounds[0]
    y_range = bounds[3] - bounds[2]
    z_range = bounds[5] - bounds[4]
    
    # Set the minimum dimension (adjustable)
    min_dim = min_grid_size
    
    # Calculate the other dimensions based on the aspect ratio
    x_dim = int(min_dim * (x_range / min(x_range, y_range, z_range)))
    y_dim = int(min_dim * (y_range / min(x_range, y_range, z_range)))
    z_dim = int(min_dim * (z_range / min(x_range, y_range, z_range)))
    
    sample_shape = (x_dim, y_dim, z_dim)

    sampled_data = resample_vtm(data, sample_shape, bounds)

    # reshape the sampled data to the sample shape
    for array_name, values in sampled_data.items():
        sampled_data[array_name] = values.reshape([sample_shape[2], sample_shape[1], sample_shape[0]])
    
    return sampled_data
