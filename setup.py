from setuptools import setup, find_packages

setup(
    name='mlbasics',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'scipy',
        'seaborn',
        'numpy',
        'pandas',
        'matplotlib',
        "tqdm"
    ]
)
