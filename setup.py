from setuptools import setup
import re
import os
import codecs
from setuptools.command.test import test as TestCommand
import sys


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import shlex
        import pytest

        self.pytest_args += (
            " --cov=meshparty --cov-report html --junitxml=test-reports/test.xml"
        )

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


with open("test_requirements.txt", "r") as f:
    test_required = f.read().splitlines()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()
    print(required)

dependency_links = []
del_ls = []
for i_l in range(len(required)):
    l = required[i_l]
    if l.startswith("-e"):
        dependency_links.append(l.split("-e ")[-1])
        del_ls.append(i_l)

        required.append(l.split("=")[-1])

for i_l in del_ls[::-1]:
    del required[i_l]

setup(
    use_scm_version=True,
    name="meshparty",
    description="a service to work with meshes",
    author="Sven Dorkenwald, Forrest Collman, Casey Schneider-Mizell",
    author_email="svenmd@princeton.edu, forrestc@alleninstitute.org, caseys@alleninstitute.org",
    url="https://github.com/sdorkenw/MeshParty.git",
    packages=["meshparty"],
    include_package_data=True,
    install_requires=required,
    extras_require={"SDF": ["pyembree"], "REPAIR": ["caveclient>=4.0.0"]},
    setup_requires=["pytest-runner", "setuptools_scm"],
    dependency_links=dependency_links,
    tests_require=test_required,
    cmdclass={"test": PyTest},
)
