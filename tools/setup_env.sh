#!/bin/bash
set -e

echo "Creating conda environment in .venv/ with Python 3.11..."
conda create -y -p .venv/ python=3.11
echo "Conda environment created."


echo "Activating environment"
conda activate .venv/


echo "Installing dependencies"

# Extract extra index URLs from pyproject.toml
pip install toml
EXTRA_INDEX_URLS=$(python -c "
import toml
config = toml.load('pyproject.toml')
indexes = config.get('tool', {}).get('pip', {}).get('extra-index-urls', [])
for url in indexes:
    print(url)
")

# Set pip extra index URLs environment variable
if [[ ! -z "$EXTRA_INDEX_URLS" ]]; then
    export PIP_EXTRA_INDEX_URL="$EXTRA_INDEX_URLS"
fi
pip install -e .[renderer,tcnn]

echo "Installing nerfacc"
pip install "nerfacc @ git+https://github.com/nerfstudio-project/nerfacc.git"

echo "Installing CUDA accelerated AMGSRN"
pip install AMGSRN/Models/AMGSRN/

echo "Setup complete."
