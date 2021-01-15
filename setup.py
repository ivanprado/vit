from setuptools import setup, find_packages

setup(
    name='vit',
    version='0.1.0',
    description='Playing with visual transformers',
    author='Ivan de Prado Alonso',
    author_email='ivan.prado@gmail.com',
    url='https://github.com/ivanprado/vit',
    packages=find_packages(exclude=['tests']),
    package_data={"vit": ["py.typed"]},
    zip_safe=False,
    install_requires=[
        'timm',
        'torch',
        'torchvision',
        'matplotlib',
        'numpy',
        'pytorch-lightning',
        'neptune-client',
        'psutil',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
