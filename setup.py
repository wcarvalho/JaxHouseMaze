
from setuptools import setup, find_packages

setup(
    name="jaxhousemaze",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.16",
        "distrax>=0.1.5",
        "flax>=0.8.2",
        "numpy>=1.26.4"
    ],
    author="Wilka Carvalho",
    author_email="wcarvalho92@gmail.com",
    description="A library to easily create maze environment with strings and an image dictionary.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wcarvalho/https://github.com/wcarvalho/JaxHouseMaze",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
