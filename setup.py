import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README_pypi.rst").read_text()

# This call to setup() does all the work
setup(
    name="bjsfm",
    version="0.1.1",
    description="Bolted Joint Stress Field Model",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/BenjaminETaylor/bjsfm",
    author="Benjamin E. Taylor",
    author_email="benjaminearltaylor@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords='joints stress engineering',
    packages=["bjsfm"],
    # packages=find_packages(exclude=("tests",)),
    # include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires='~=3.6',
    # entry_points={
    #     "console_scripts": [
    #         "realpython=reader.__main__:main",
    #     ]
    # },
)
