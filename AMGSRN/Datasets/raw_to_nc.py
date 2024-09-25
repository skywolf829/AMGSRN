from pathlib import Path
from AMGSRN.Other.utility_functions import npy_to_cdf
import numpy as np

def raw_to_nc(raw_path: Path, dtype, shape, nc_path: Path):
    raw_data = np.fromfile(raw_path, dtype=dtype).reshape(shape)
    print(raw_data.min(), raw_data.max())
    npy_to_cdf(raw_data[None,None], nc_path)
    quit()
    raw_data = raw_data.astype(np.float32)
    print(raw_data.min(), raw_data.max())
    raw_data -= raw_data.min()
    raw_data /= raw_data.max()
    npy_to_cdf(raw_data[None,None], nc_path)

raw_to_nc(Path("./Data/pig_heart_2048x2048x2612_int16.raw").resolve(), np.int16, (2048, 2048, 2612), 
          Path("./Data/pig_heart.nc").resolve())
