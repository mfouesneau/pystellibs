pystellib - Making synthetic spectra from libraries
===================================================

:author: M Fouesneau

full documentation: http://mfouesneau.github.io/pystellibs/

This is a set of tools to compute synthetic spectra from libraries in a simple
way, ideal to integrate in larger projects.

In this package, multiple spectral libraries and atmosphere libraries are
provided with a common usage. This allows any user to transparently switch from
one set of models to another without pain (or combine sets together)

This package handles **units** provided through Astropy with added values to
handle specific spectroscopic units.

## Todolist

- [x] Move to astropy units and remove pint dependency. (Astropy already needed)
- [x] Remove simpletable dependency
- [x] Documentation fix 
- [ ] Add a function to download library files from the link, when missing.
- [ ] Documentation improvement


## Known issues
- [ ] Inconsistency between astropy and ezunits/pint on Lsun
