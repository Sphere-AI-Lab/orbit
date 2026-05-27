from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from runtime_step0_parity_utils import main as _parity_main  # noqa: E402


def main() -> None:
    _parity_main(default_precision_profile="int4")


if __name__ == "__main__":
    main()
