# Packaging and release

The distribution, Python package and console command are all `howhard`. The version's
single source is `src/howhard/__init__.py`; setuptools reads it dynamically, `--version`
prints it, and Sphinx uses the same value. Use stable `MAJOR.MINOR.PATCH` versions.
Increment the patch for compatible fixes, the minor for compatible additions, and the
major for incompatible public interfaces. While below 1.0, document any incompatible
change explicitly. Receipt and request format versions are separate compatibility contracts.
Keep the version fields in the portable, Claude Code and Antigravity plugin manifests
in sync with that Python version; distribution validation checks them in the source archive.
The [1.0 upgrade guide](upgrading-1.0.md) records the current compatibility boundary;
the published 0.6.1 maintenance release has its own retained changelog section.

Before release:

1. Set the intended stable package version and add a dated matching changelog section.
   A source version is a release candidate until published; a changelog entry alone
   does not assert that a release exists on PyPI.
2. Land the exact release commit through `agc land`, with `scripts/check` as its gate.
   The gate builds the wheel and source distribution, checks the version and packaged
   defaults, runs the complete behavioral suite and builds the canonical docs.
3. Build a wheel from the source distribution and install that exact wheel into a clean
   environment outside the checkout. Check `howhard --version`, package data through
   `importlib.resources`, and the `context --candidates FILE` / `decide --seed 42`
   protocol using explicitly labeled synthetic forecasts. These mechanical probes
   require no provider calls or real execution observations. Standalone `pick` needs
   a configured inference provider and is not a provider-free release check.
4. Tag the verified commit `vX.Y.Z` and create a GitHub release only when requested.
   Attach the verified distributions and their SHA-256 checksums.
5. Upload to PyPI only if the user explicitly authorizes that upload. Implementation,
   tests, commits, pushes, merges, tags and GitHub releases do not authorize PyPI.

The workflows contain no upload credentials and do not publish packages. The connected
[Read the Docs site](https://howhard.readthedocs.io/en/latest/) builds Sphinx from this
repository with strict warnings and the same canonical index.
