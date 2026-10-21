"""``python -m preferencefactory`` 入口，与 ``python main.py`` 等价。"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
