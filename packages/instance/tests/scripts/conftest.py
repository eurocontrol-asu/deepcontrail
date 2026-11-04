"""Make this package's ``scripts/`` importable as ``instance_scripts`` in the script tests.

The scripts are command-line tools, not part of the installed package; a distinct name keeps
them apart from other packages' ``scripts/`` folders.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_scripts = types.ModuleType("instance_scripts")
_scripts.__path__ = [str(Path(__file__).resolve().parents[2] / "scripts")]
sys.modules.setdefault("instance_scripts", _scripts)
