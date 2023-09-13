from setuptools import setup, find_packages

setup(
    name='niflow_diffusion_package',
    version='0.1',
    author ='Ziad',
    description= 'Diffusion processing pipelines',
    packages=find_packages(),
    install_requires=[r.strip() for r in open('requirements.txt').readlines()],
    setup_requires=['setuptools>=49', 'wheel']
    
)
