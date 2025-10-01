pystellib - Making synthetic spectra from libraries
===================================================

:author: M Fouesneau

full documentation: http://mfouesneau.github.io/pystellibs/

This is a set of tools to compute synthetic spectra from libraries in a simple
way, ideal to integrate in larger projects.

In this package, multiple spectral libraries and atmosphere libraries are
provided with a common usage. This allows any user to transparently switch from
one set of models to another without pain (or combine sets together)


This package handles **units** provided through a frozen version of `pint`
included in this package.

## Todolist

- [ ] Add a function to download library files from the link, when missing.
- [x] Move to astropy units and remove pint dependency. (Astropy already needed)
- [ ] See if removing simpletable dependency is possible (for pandas DataFrame with aliasing layer)
- [ ] Documentation fix and improvement
- [ ] Remove `pystellibs.future` and use `matplotlib.path.Path` directly (recent versions of matplotlib)


## Known issues
- [ ] Inconsistency between astropy and ezunits/pint on Lsun
