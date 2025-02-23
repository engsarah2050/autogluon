#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import os
from setuptools import setup
import importlib.util
filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(filepath, '..', 'core', 'src', 'autogluon', 'core', '_setup_utils.py')
spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)
###########################

version = ag.load_version_file()
version = ag.update_version(version)

submodule = 'DeepInsight_auto'

requirements = [
    'scikit-learn>=0.22',
    'pandas',
    
    'psutil>=5.7.3,<5.9',  # TODO: Consider capping to <6.0 instead, capping to 5.9 to avoid possible issues.
    'networkx>=2.3,<3.0', 
    f'autogluon.core=={version}',
    f'autogluon.features=={version}',
    #f'autogluon.tabular_to_image=={version}', 
]

test_requirements = [
    'pytest',
    'openml',
]

"""
extras_require = {
    'torch': [
        'torch>=1.0,<2.0',
        
    ],
    
    }
 """

extras_require={
        'ImageTransformer_fit_plot': ['matplotlib']
    }

all_requires = []
# TODO: Consider adding 'skex' to 'all'
for extra_package in ['ImageTransformer_fit_plot']:
    all_requires += extras_require[extra_package]
all_requires = list(set(all_requires))
extras_require['all'] = all_requires 
 

 
install_requires = requirements + test_requirements
install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == '__main__':
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        #extras_require=extras_require,
        **setup_args,
    )


''' setup(
    name='DeepInsight',
    version='0.1.0',
    packages=['pyDeepInsight'],
    url='https://github.com/alok-ai-lab/deepinsight',
    license='GPLv3',
    author='Keith A. Boroevich',
    author_email='kaboroevich@gmail.com',
    description='A methodology to transform a non-image data to an image for'
                ' convolution neural network architecture',
    install_requires=install_requires,
    extras_require={
        'ImageTransformer_fit_plot': ['matplotlib']
    }
) '''

