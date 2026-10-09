from __future__ import annotations

import sys
import unittest
from pathlib import Path


from teia.core.runtime.engine import _prepare_module_config


class CheckpointRestoreConfigTests(unittest.TestCase):
    def test_prepare_module_config_disables_nested_pretrained_flags(self) -> None:
        prepared = _prepare_module_config(
            config={
                "netmodule": {
                    "_target_": "tests.test_checkpoint_restore.FakeModule",
                    "pretrained": True,
                    "encoder": {"pretrained": True, "model_name": "vit_base_patch16_siglip_512"},
                    "nested": [{"pretrained": True}, {"pretrained": "keep-me"}],
                    "__task__": "multi-cls",
                }
            },
            checkpoint="runs/example/checkpoints/best.ckpt",
        )

        self.assertEqual(
            prepared,
            {
                "_target_": "tests.test_checkpoint_restore.FakeModule",
                "_recursive_": False,
                "_convert_": "all",
                "pretrained": False,
                "encoder": {"pretrained": False, "model_name": "vit_base_patch16_siglip_512"},
                "nested": [{"pretrained": False}, {"pretrained": "keep-me"}],
            },
        )


if __name__ == "__main__":
    unittest.main()
