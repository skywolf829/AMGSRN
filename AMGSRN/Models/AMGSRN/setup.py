from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
os.path.dirname(os.path.abspath(__file__))


setup(
    name='AMG_Encoder',
    version='0.1',
    packages=['AMG_Encoder'],
    ext_modules=[
        CUDAExtension(
            name='AMG_Encoder._C', 
            sources=['src_testinggrounds/AMG_encoder.cpp', 'src_testinggrounds/AMG_encoder_kernels.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
