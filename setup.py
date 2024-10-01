from setuptools import setup, find_packages
import subprocess
import os
import sys

def install_subpackage():
    cuda_setup_path = os.path.join(os.path.dirname(__file__), 'AMGSRN', 'Models', 'AMGSRN')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', cuda_setup_path])

print("Installing base packages")
setup(
    name="AMGSRN",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'amgsrn=AMGSRN.CLI.bindings:main',
        ],
    },
)
