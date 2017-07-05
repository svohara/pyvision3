from setuptools import setup, find_packages

setup(
    name="pyvision3",
    version="0.22",
    packages=find_packages(),
    url="https://github.com/svohara/pyvision3",
    license="MIT",
    author="Stephen O'Hara",
    author_email="svohara@gmail.com",
    description="Facilitates computer vision research and prototyping using python and openCV",
    package_data={
        # include all files found in the 'data' subdirectory
        # of the 'pyvision' package
        'pyvision': ['data/*']
    },
    test_suite="tests"
)
