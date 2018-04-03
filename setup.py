from setuptools import setup, find_packages
import os
import json

with open(os.path.join('pyvision', 'data', 'project_info.json'), 'r') as infile:
    _info = json.load(infile)

_version = ".".join([str(x) for x in _info["version_tuple"]])

setup(
    name="pyvision",
    version=_version,
    packages=find_packages(),
    url=_info["url"],
    license=_info["license"],
    author=_info["author"],
    author_email=_info["email"],
    description=_info["description"],
    package_data={
        # include all files found in the 'data' subdirectory
        # of the 'pyvision' package
        'pyvision': ['data/*']
    },
    test_suite="tests",
    download_url="{}/archive/{}.tar.gz".format(_info["url"], _version),
    keywords=["images", "video", "computer vision"],
    classifiers=[],
)
