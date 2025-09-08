import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "pytunetag",
    version = "0.0.1",
    author = "Gregory Taylor",
    author_email = "gregory.taylor.au@gmail.com",
    description = "A Python library to automatically detect and modify music genres for mp3 files.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Aveygo/pyTuneTag",
    project_urls = {
        "Bug Tracker": "https://github.com/Aveygo/pyTuneTag/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.12",
    install_requires=[
        "torch",
        "numpy",
        "pydub",
        "mutagen",
        "pandas",
        "scikit-learn",
        "omegaconf",
        "librosa",
        "torchaudio",
        "transformers",
    ],
    entry_points={
        'console_scripts': [
            'pytunetag = pytunetag.src.cli:cli',
        ],
    },
)