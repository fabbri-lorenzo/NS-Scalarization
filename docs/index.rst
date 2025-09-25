NS‑Scalarization documentation

Welcome to the documentation for the **NS‑Scalarization** project.  This
project explores scalar fields around neutron stars and implements a
shooting method to solve for self‑consistent scalar profiles.  The code
is organized as a small library with a `Utils/` package and a
stand‑alone `main.py` script that orchestrates the numerical workflow.

This documentation provides:

* An overview of the workflow executed by the `main.py` script, with a
  Mermaid diagram illustrating the key steps in the shooting algorithm.
* Auto‑generated API reference pages for the Python modules in this
  repository (excluding those under development).

If you are reading these pages locally, you can build the HTML
documentation by running ``sphinx-build`` from the root of this
repository::

   sphinx-build -b html docs docs/_build/html

Alternatively, you can use the provided ``Makefile`` targets if you
generated them via ``sphinx-quickstart``.  The configuration file
``docs/conf.py`` contains explanations of the enabled extensions and
settings.

```{toctree}
:maxdepth: 2
:caption: Contents:

workflow
api_reference
```
