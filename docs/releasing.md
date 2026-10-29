# Packaging and release

The distribution, Python package and console command are all `howhard`. The version's
single source is `src/howhard/__init__.py`; setuptools reads it dynamically, `--version`
prints it, and Sphinx uses the same value. Use stable `MAJOR.MINOR.PATCH` versions.
Increment the patch for compatible fixes, the minor for compatible additions, and the
major for incompatible public interfaces. While below 1.0, document any incompatible
change explicitly. Cache and prior format versions are separate compatibility contracts.

Before release:

1. Set the intended stable package version and add a dated matching changelog section.
   A source version is a release candidate until published; a changelog entry alone
   does not assert that a release exists on PyPI.
2. Land the exact release commit through `agc land`, with `scripts/check` as its gate.
   The gate builds the wheel and source distribution, checks the version and packaged
   defaults, runs the complete behavioral suite and builds the canonical docs.
3. Install the built wheel into a clean environment outside the checkout. Check
   `howhard --version`, `howhard pick bookkeeping --seed 42 --json`, and package data
   through `importlib.resources`. Check the source distribution can also build a wheel.
4. Tag the verified commit `vX.Y.Z` and create a GitHub release only when requested.
   Attach the verified distributions and their SHA-256 checksums.
5. Upload to PyPI only if the user explicitly authorizes that upload. Implementation,
   tests, commits, pushes, merges, tags and GitHub releases do not authorize PyPI.

The workflows contain no upload credentials and do not publish packages. The Read the
Docs configuration builds Sphinx from this repository with strict warnings and the same
canonical index. External hosting must be connected separately before a hosted docs
URL can be claimed.
