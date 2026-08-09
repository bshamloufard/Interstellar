"""`python3 -m interstellar ...` entry point. See interstellar/cli.py."""
from __future__ import annotations

import sys

from interstellar.cli import main

if __name__ == "__main__":
    sys.exit(main())
