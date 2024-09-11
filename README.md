# AMGSRN
Improvements over APMGSRN.


## Installation
Windows: supported via WSL2.
Linux: supported.
MacOS: not supported.

Install CUDA 12.4: https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local.
`
source tools/setup_env.sh
`

Training is slower on WSL2. Can install in Windows for faster training.

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
