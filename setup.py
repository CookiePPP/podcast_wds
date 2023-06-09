import setuptools

setuptools.setup(
    name="podcast_wds",
    version="0.1.0",
    author="CookiePPP",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch',
        'webdataset',
        'torchaudio',
        'ffmpeg-python',
        'librosa',
    ]
)