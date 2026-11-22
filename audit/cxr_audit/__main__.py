"""Allow ``python -m cxr_audit`` to behave like ``cxr-audit``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
