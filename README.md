# AMGSRN
Improvements over APMGSRN.


## Installation
Windows: supported.
WSL2: supported.
Linux: supported.
MacOS: not supported.

Install CUDA 12.4: https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local.
`
source tools/setup_env.sh
`

Optional: install TinyCudaNN for faster training. In your conda environment, run the following:
`
pip install "tiny-cuda-nn @ git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
`
On Windows, the above needs to be executed in x64 Native Tools Command Prompt for VS run in administrator mode.

## Training
`
conda activate .venv
python AMGSRN/start_jobs.py --settings train.json
`

## Testing
`
conda activate .venv
python AMGSRN/start_jobs.py --settings test.json
`

## Web renderer
TODO
