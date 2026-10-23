# Contributing

Issues and pull requests are welcome.

## Setting up

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Before opening a pull request

```bash
pytest                                   # unit, property and doctests
sphinx-build -W docs docs/_build/html    # docs must build without warnings
```

A change to the solver should come with a test, and where it touches the
theory -- a new flux, domain or entropy check -- preferably a property test in
`tests/test_properties.py` or an exact reference in `fronttrack.exact`.

The code works in exact rational arithmetic (`fractions.Fraction`); please
keep floating point out of the solver itself. It is fine in the plotting,
the finite-volume comparison schemes and the numerical exact solutions.
