from AMGSRN.Other.utility_functions import nc_to_np

def process_woodbranch():
    # Load the data
    data, full_shape = nc_to_np("./Data/woodbranch.nc")
    print(full_shape)
    print(data.shape)
    print(data.dtype)
    print(data.min())
    print(data.max())

if __name__ == "__main__":
    process_woodbranch()
