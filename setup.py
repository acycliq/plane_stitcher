import os
from setuptools import setup, find_packages

install_deps = ['scipy', 'numba', 'numpy', 'scikit-image', 'tqdm',
                'fastremap', 'fill_voids', 'requests', 'pytest']

version = None
with open(os.path.join('plane_stitcher', '_version.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


setup(
    name="plane_stitcher",
    version=version,
    license="BSD",
    author="Dimitris Nicoloutsopoulos",
    author_email="dimitris.nicoloutsopoulos@gmail.com",
    description="plane_stitcher",
    long_description_content_type="text/markdown",
    url="https://github.com/acycliq/plane_stitcher",
    packages=find_packages(),
    install_requires=install_deps,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)