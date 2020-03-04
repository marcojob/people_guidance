import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="people_guidance",
    version="0.0.1",
    author="Theo Roizard, Adrian Schneebeli, Marco Job, Lorenz Hetzel",
    author_email="hetzell@student.ethz.ch",
    description="Monocular Object detection for the visually impaired.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)