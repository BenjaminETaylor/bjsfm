import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README_pypi.rst").read_text()

# This call to setup() does all the work
setup(
    name="bjsfm",
    version="0.5.1",
    description="Bolted Joint Stress Field Model",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/BenjaminETaylor/bjsfm",
    author="Benjamin E. Taylor",
    author_email="benjaminearltaylor@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    keywords='bolted joints stress engineering composites',
    packages=["bjsfm"],
    # packages=find_packages(exclude=("tests",)),
    # include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "nptyping",
    ],
    python_requires='~=3.9',
    entry_points={
        "console_scripts": [
            "bjsfm=bjsfm.__main__:main",
        ]
    },
)
