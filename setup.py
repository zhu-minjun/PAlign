from setuptools import setup, find_packages

setup(
    name='PAlign',
    version='0.1.0',
    description='Personalized Alignment with PASO for Large Language Models',
    author='xxx',
    author_email='xxx',
    url='xxx',
    packages=find_packages(where='./PAlign'),
    package_dir={'': './PAlign'},
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'transformers',
        'scikit-learn',
        'tqdm',
        'openai',
        'matplotlib',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
