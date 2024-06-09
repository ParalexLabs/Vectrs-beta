from setuptools import setup, find_packages


setup(
    name="vectrs",
    version="0.1.0",
    author="Mir Sakib",
    author_email="sakib@paralex.tech",
    url="https://github.com/ParalexLabs/Vectrs-beta",
    description="Pre release of Vectrs, a decentralized and distributed vector database network",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "hnswlib==0.8.0",
        "kademlia==2.2.2",
        "numpy==1.24.4",
        "rpcudp==5.0.0",
        "u-msgpack-python==2.8.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'vectrs=main:main',
        ],
    },
)
