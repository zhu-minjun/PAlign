from setuptools import setup, find_packages

setup(
    name='PAlign',
    version='0.1.0',
    description='Personalized Alignment with PASO for Large Language Models',
    author='xxx',
    author_email='xxx',
    url='xxx',
    packages=find_packages(),
    install_requires=[
        'baukit @ git+https://github.com/davidbau/baukit.git@9d51abd51ebf29769aecc38c4cbef459b731a36e',
        'einops>=0.8.2',
        'huggingface_hub>=0.36.2',
        'jinja2>=3.1.0',
        'matplotlib',
        'numpy',
        'openpyxl',
        'pandas',
        'safetensors>=0.7.0',
        'scikit-learn>=0.23.2',
        'torch>=2.5.1',
        'tqdm>=4.67.3',
        'transformers>=4.50,<5',
        'xlrd>=2.0.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
