
from setuptools import setup

setup(name='feature_utils',
      version='0.1',
      description='Utilities for generating features for machine learning pipelines',
      author='Gautam Borgohain',
      author_email='gautamborgohain90@gmail.com',
      license='MIT',
      packages=['feature_utils'],
      install_requires=['sklearn','numpy','pandas', 'tqdm','lightgbm'],
      zip_safe=False)


