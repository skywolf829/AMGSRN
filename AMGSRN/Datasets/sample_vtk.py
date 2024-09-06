import vtk
import numpy as np
import argparse
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import time

def load_vtm_file(file_path):
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def create_probe_filter(multiblock):
    probe_filter = vtk.vtkCompositeDataProbeFilter()
    probe_filter.SetSourceData(multiblock)
    return probe_filter

def sample_points(probe_filter, points):
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

def main():
    parser = argparse.ArgumentParser(description='Sample points from a VTM file.')
    parser.add_argument('file_path', type=str, help='Path to the VTM file')
    args = parser.parse_args()

    try:
        # Time the data loading
        start_time = time.time()
        multiblock = load_vtm_file(args.file_path)
        load_time = time.time() - start_time
        print(f"Data loading time: {load_time:.2f} seconds")

        # Create the probe filter
        probe_filter = create_probe_filter(multiblock)

        # Generate random points
        num_points = 100000
        points = np.random.rand(num_points, 3)

        # Get the bounds of the multiblock dataset
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
        
        # Rescale points to the domain size
        for i in range(3):
            points[:, i] = points[:, i] * (bounds[2*i+1] - bounds[2*i]) + bounds[2*i]

        # Time the sampling
        start_time = time.time()
        sampled_data = sample_points(probe_filter, points)
        sampling_time = time.time() - start_time
        print(f"Sampling time: {sampling_time:.2f} seconds")
        
        print("Sampled data:")
        for array_name, values in sampled_data.items():
            print(f"{array_name}: shape {values.shape}")
            print(f"  Min: {values.min()}, Max: {values.max()}")
            print(f"  First 5 values: {values[:5]}")
            print()

        # Optionally, save the sampled data
        np.savez('sampled_data.npz', **sampled_data)
        print("Sampled data saved to 'sampled_data.npz'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()