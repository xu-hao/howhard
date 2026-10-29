from importlib.metadata import version as distribution_version

project = "howhard"
release = distribution_version("howhard")
version = release
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
master_doc = "index"
html_theme = "furo"
exclude_patterns = ["_build"]
