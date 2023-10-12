from setuptools import setup

setup(
    name="openbmi",
    version="0.1",
    description="open source [ca] bmi imaging tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/catubc/bmi",
    author="https://github.com/catubc/bmi",
    author_email="<mitelutco@gmail.com>",
    packages=["openbmi"],
    install_requires=[
        'numpy',
        "matplotlib",
        'parmap',
        'tqdm',
        'scipy',
        'opencv-python',
        'scikit-learn',
        'pandas'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

