# Exit on error
$ErrorActionPreference = "Stop"

Write-Host "Creating conda environment in .venv/ with Python 3.11..."
conda create -y -p .venv/ python=3.11
Write-Host "Conda environment created."

Write-Host "Activating environment"
conda activate .venv/

Write-Host "Installing dependencies"

# Install toml package
pip install toml

# Extract extra index URLs from pyproject.toml
$EXTRA_INDEX_URLS = python -c @"
import toml
config = toml.load('pyproject.toml')
indexes = config.get('tool', {}).get('pip', {}).get('extra-index-urls', [])
for url in indexes:
    print(url)
"@

# Set PIP_EXTRA_INDEX_URL environment variable if extra index URLs are found
if ($EXTRA_INDEX_URLS) {
    $env:PIP_EXTRA_INDEX_URL = $EXTRA_INDEX_URLS -join "`n"
}

# Install the package in editable mode
pip install -e .

Write-Host "Installing CUDA accelerated AMGSRN"
pip install AMGSRN/Models/AMGSRN/

Write-Host "Setup complete."