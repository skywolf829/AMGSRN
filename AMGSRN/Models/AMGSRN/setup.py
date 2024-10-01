from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools.command.build_ext import build_ext
import os
os.path.dirname(os.path.abspath(__file__))


setup(
    name='AMG_Encoder',
    version='0.1',
    packages=['AMG_Encoder'],
    ext_modules=[
        CUDAExtension(
            name='AMG_Encoder._C', 
            sources=['src/AMG_encoder.cpp', 'src/AMG_encoder_kernels.cu'],
            extra_compile_args={'nvcc': ['-lineinfo']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
