# setup.py

from setuptools import setup, find_packages

setup(
    name='spec',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
                      'einops>=0.7.0',
                      'fairscale==0.4.13',
                      'numpy==1.24.4',
                      'open_clip_torch==2.24.0',
                      'Pillow==10.3.0',
                      'PyYAML==6.0.1',
                      'setuptools==69.2.0',
                      'timm==0.9.16',
                      'torch==2.2.1',
                      'torchvision==0.17.1',
                      'tqdm==4.66.2',
                      'transformers==4.38.2',
                      'huggingface-hub==0.21.4',
                      'zipp==3.17.0'
    ]
)