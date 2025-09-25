# API reference

This section contains the API reference for the Python modules in
**NS‑Scalarization**.  The reference is generated automatically by
Sphinx from the source code docstrings.  To keep the documentation
focused, modules that are incomplete or undergoing heavy development
(such as `Utils/scan.py`) are excluded.

If you have installed Sphinx and its extensions locally, you can
generate these pages yourself by running:

```bash
sphinx-apidoc -o docs/api ../Utils
```

and then building the documentation with ``sphinx-build``.  The
``autosummary`` directive below tells Sphinx to create summary pages
for each module and to import their members to document the functions
and classes.  To update the API reference after modifying the code,
delete the generated files under ``docs/api/`` and rerun the
``sphinx-apidoc`` command.

::

   .. autosummary::
      :toctree: api
      :recursive:

      Utils.params
      Utils.EOS
      Utils.TOV_EMG
      Utils.shooting
      Utils.solver
      Utils.graphics
      Utils.diagnostic
