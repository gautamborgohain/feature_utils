from setuptools import setup, find_packages

setup(name='feature_utils',
      version='0.1c',
      description='Utilities for generating features for machine learning pipelines',
      author='Gautam Borgohain',
      author_email='gautamborgohain90@gmail.com',
      license='MIT',
      packages=['feature_utils','feature_utils.imports', 'feature_utils.utils'],
      install_requires=['sklearn','numpy','pandas', 'tqdm','lightgbm'],
      zip_safe=False)


