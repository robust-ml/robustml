robust-ml
=========

Interfaces for defining Robust ML models and precisely specifying the threat
models under which they claim to be secure.

Also includes interfaces for specifying attacks and evaluating attacks against
models.

Packaging
---------

1. Update version information.

2. Build the package using ``python setup.py sdist bdist_wheel``.

3. Sign and upload the package using ``twine upload -s dist/*``.

4. Create a signed tag in the git repo with the version number that was
   uploaded to PyPI.
