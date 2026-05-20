"""FlashRT serving glue.

Submodules in this package adapt FlashRT models to third-party serving
protocols. They are intentionally optional: importing
``flash_rt.serving.openpi_adapter`` requires ``openpi-client`` to be
installed, but the bare ``flash_rt`` import does not.

Install with::

    pip install -e ".[serving]"
"""

from __future__ import annotations
