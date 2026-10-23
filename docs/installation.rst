Installation
============

``fronttrack`` needs Python 3.10 or newer, NumPy and Matplotlib.  From a clone
of the repository:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"

The ``dev`` extra adds pytest and Hypothesis, for the test suite:

.. code-block:: bash

   pytest

Documentation
-------------

The ``docs`` extra installs Sphinx and the theme used here:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -W docs docs/_build/html
