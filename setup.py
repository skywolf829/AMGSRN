from setuptools import setup, find_packages

setup(
    name="AMGSRN",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'amgsrn=AMGSRN.CLI.bindings:compress_decompress',
            'amgsrn-render=AMGSRN.CLI.bindings:renderer',
        ],
    }
)
