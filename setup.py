from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mad_fw",
    version="0.1.2",
    license="MIT",
    description="Multitask Aggression Detection (MAD)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Diptanu Sarkar",
    author_email="diptanusarkar@hotmail.com",
    url="https://github.com/imdiptanu/MAD",
    download_url="https://github.com/imdiptanu/MAD/archive/refs/tags/0.1.1.tar.gz",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires="==3.7",
    install_requires=[
        "pyyaml==5.4.1",
        "utils/FARM",
        "pathlib==1.0.1",
    ],
)
