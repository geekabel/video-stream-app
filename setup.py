from setuptools import setup, find_packages

setup(
    name='video-streaming-app',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
    ],
)
