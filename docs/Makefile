.PHONY: help

BUILDDIR      = _build
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build

APIBUILD      = sphinx-apidoc
APIOPTS       = -o ./
APIDIR        = ../pystellibs

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "apidoc       to generate the API files"

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*

apidoc:
	$(APIBUILD) $(APIOPTS) $(APIDIR)

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
