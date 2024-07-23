from setuptools import setup, find_packages

setup(
    name='Forex_Forecast',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'sysidentpy',
        'torch',
        'scikit-learn',
    ],
)