from setuptools import setup, find_packages

setup(
    name="sigmacam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",          # PyTorch
        "numpy",         # Array operations
        "matplotlib",    # Plotting
        "imageio-ffmpeg",# Video export
        "imageio",       # Animation I/O
    ],
    author="Your Name",
    description="SigmaCam: decision-boundary visualization for neural nets",
)
