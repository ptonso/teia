"""``python -m teia`` entrypoint.

Shipped by the ``teia`` distribution. This is a module, NOT an ``__init__.py``, so it
does not turn ``teia/`` into a regular package — the PEP 420 namespace is preserved.
"""

from teia.core.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
