from setuptools import setup, find_packages

setup(
    name="PandaZam",
    version="0.0.1",
    author="CogWorks 2021",
    author_email="",
    description="Song Recognition Package",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/A-Group-of-Pandas/AudioProject",
    project_urls={
        "Bug Tracker": "https://github.com/A-Group-of-Pandas/AudioProjects/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "song_recognition"},
    packages=find_packages(where="song_recognition"),
    python_requires=">=3.6",
)