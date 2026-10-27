"""Skip the converter tests when deepcontrail-opencontrails is not installed.

It is opt-in (``uv sync --group convert``) because it brings TensorFlow; the
``opencontrails`` CI job installs it and runs these tests.
"""

from importlib.util import find_spec

if find_spec("deepcontrail.opencontrails") is None:
    collect_ignore_glob = ["test_*.py"]
