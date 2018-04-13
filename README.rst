robust-ml
=========

Interfaces for defining Robust ML models and precisely specifying the threat
models under which they claim to be secure. Also includes interfaces for
specifying attacks and evaluating attacks against models.

The motivation behind this project is to make it easy to make specific,
testable claims about the robustness about machine learning models. Read more
in the `FAQ <http://www.robust-ml.org/faq/>`__.

Installation
------------

You can install from PyPI: ``pip install robustml``.

Usage
-----

See `this repository <https://github.com/robust-ml/example>`__ for a complete
example of implenenting a model, implementing an attack, and evaluating the
attack against the model.

If you're implementing a **defense**, you should implement
``robustml.model.Model``. See `here
<https://github.com/robust-ml/example/blob/master/inception_v3.py>`__ for an
example.

If you're implementing an **attack** against a specific defense, you should
implement ``robustml.attack.Attack``. See `here
<https://github.com/robust-ml/example/blob/master/attack.py>`__ for an example.

To **evaluate** a specific attack against a specific defense, use
``robustml.evaluate.evaluate()``. See `here
<https://github.com/robust-ml/example/blob/master/run.py>`__ for an example.

Contributing
------------

Do you have ideas on how to improve the robustml package? Have a feature
request (such as a specification of a new threat model) or bug report? Great!
Please open an `issue <https://github.com/robust-ml/robustml/issues>`__ or
submit a `pull request <https://github.com/robust-ml/robustml/pulls>`__.

Before contributing a major change, it's recommended that you open a pull
request first and get feedback on the idea before investing time in the
implementation.

Packaging
---------

1. Update version information.

2. Build the package using ``python setup.py sdist bdist_wheel``.

3. Sign and upload the package using ``twine upload -s dist/*``.

4. Create a signed tag in the git repo with the version number that was
   uploaded to PyPI.
