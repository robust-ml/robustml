from setuptools import setup
from codecs import open # For a consistent encoding
from os import path
import re


here = path.dirname(__file__)


with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def read(*names, **kwargs):
    with open(
        path.join(here, *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='robustml',

    version=find_version('robustml', '__init__.py'),

    description='Robust ML API',
    long_description=long_description,

    url='https://github.com/robust-ml/robust-ml',

    author='Anish Athalye',
    author_email='me@anishathalye.com',

    license='MIT',

    packages=['robustml'],

    install_requires=[
        'numpy>=1,<2',
        'Pillow>=5,<6',
    ],
)
