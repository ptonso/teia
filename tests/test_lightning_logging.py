from __future__ import annotations

import logging
import unittest

from teia.core.runtime.engine import _LightningLitLoggerTipFilter, _suppress_lightning_litlogger_tip


class LightningLoggingTests(unittest.TestCase):
    def test_litlogger_tip_filter_only_blocks_litlogger_tip(self) -> None:
        filter_ = _LightningLitLoggerTipFilter()

        litlogger_record = logging.LogRecord(
            name="lightning.pytorch.utilities.rank_zero",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="💡 Tip: For seamless cloud logging and experiment tracking, try installing litlogger.",
            args=(),
            exc_info=None,
        )
        gpu_record = logging.LogRecord(
            name="lightning.pytorch.utilities.rank_zero",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="GPU available: True (cuda), used: True",
            args=(),
            exc_info=None,
        )

        self.assertFalse(filter_.filter(litlogger_record))
        self.assertTrue(filter_.filter(gpu_record))

    def test_litlogger_tip_filter_registration_is_idempotent(self) -> None:
        logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")
        before = sum(isinstance(filter_, _LightningLitLoggerTipFilter) for filter_ in logger.filters)

        _suppress_lightning_litlogger_tip()
        _suppress_lightning_litlogger_tip()

        after = sum(isinstance(filter_, _LightningLitLoggerTipFilter) for filter_ in logger.filters)
        self.assertEqual(after, max(before, 1))


if __name__ == "__main__":
    unittest.main()
