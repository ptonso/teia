"""TaskContract checks: net, eval and data boundaries (teia:base/task.md)."""

from __future__ import annotations

import unittest

from teia.base.task import Blob, Dim, Names, Ragged, TaskContract, TaskContractError, Tensor


class Cls(TaskContract):
    meta = {"num_classes": Dim(), "class_names": Names("num_classes")}
    batch = {"image": Tensor("B 3 H W", "float32"), "cls": Tensor("B", "int64", target=True)}
    capture = {"cls.scores": Tensor("B num_classes", "float32")}


class Det(TaskContract):
    meta = {"num_classes": Dim(), "class_names": Names("num_classes"), "kpt_shape": Dim()}
    batch = {
        "image": Tensor("B 3 H W"),
        "bboxes": Ragged("N 4", index="batch_idx", target=True),
        "batch_idx": Ragged("N", index="batch_idx", target=True),
        "keypoints": Ragged("N kpt_shape", index="batch_idx", target=True),
        "mask": Blob("png", target=True),
    }
    capture = {
        "det.boxes": Ragged("N 4", index="det.sample_idx"),
        "det.score": Ragged("N", index="det.sample_idx"),
        "det.sample_idx": Ragged("N", index="det.sample_idx"),
    }


CLS_META = {"num_classes": 3, "class_names": ["a", "b", "c"]}


class CheckNetTests(unittest.TestCase):
    def test_valid(self):
        Cls().check_net({"batch.image", "batch.cls", "meta.num_classes", "feat.x"}, {"cls.scores": "act.probs"})

    def test_undeclared_batch_field(self):
        with self.assertRaisesRegex(TaskContractError, r"test_task_contract\.Cls: netmodule reads undeclared batch field 'batch.text'"):
            Cls().check_net({"batch.text"}, {"cls.scores": "act.probs"})

    def test_undeclared_meta(self):
        with self.assertRaisesRegex(TaskContractError, "meta.num_targets"):
            Cls().check_net({"meta.num_targets"}, {"cls.scores": "act.probs"})

    def test_missing_capture_atom(self):
        with self.assertRaisesRegex(TaskContractError, r"lacks \['cls.scores'\]"):
            Cls().check_net({"batch.image"}, {})

    def test_ragged_index_is_derived(self):
        self.assertEqual(Det().derived(), {"det.sample_idx"})
        Det().check_net({"batch.image"}, {"det.boxes": "post.act.boxes", "det.score": "post.act.score"})


class CheckEvalTests(unittest.TestCase):
    def test_valid(self):
        Cls().check_eval({"capture.cls.scores", "batch.cls", "meta.class_names", "log.val/loss", "eval.cls.confusion"})

    def test_non_target_batch_field(self):
        with self.assertRaisesRegex(TaskContractError, "batch.image"):
            Cls().check_eval({"batch.image"})

    def test_unknown_capture_and_namespace(self):
        for key in ("capture.det.boxes", "feat.pooled"):
            with self.assertRaisesRegex(TaskContractError, key):
                Cls().check_eval({key})

    def test_targets(self):
        self.assertEqual(Det().targets(), ["bboxes", "batch_idx", "keypoints", "mask"])


class CheckDataTests(unittest.TestCase):
    def test_valid(self):
        Cls().check_data({"image": (3, 32, 32), "cls": ()}, CLS_META, ["image", "cls"])

    def test_meta_missing_and_names_length(self):
        with self.assertRaisesRegex(TaskContractError, "does not publish meta 'class_names'"):
            Cls().check_data({}, {"num_classes": 3}, [])
        with self.assertRaisesRegex(TaskContractError, "2 names but 'num_classes' is 3"):
            Cls().check_data({}, {"num_classes": 3, "class_names": ["a", "b"]}, [])

    def test_missing_field_and_bad_shape(self):
        with self.assertRaisesRegex(TaskContractError, "does not emit batch field 'cls'"):
            Cls().check_data({"image": (3, 8, 8)}, CLS_META, ["image"])
        with self.assertRaisesRegex(TaskContractError, r"'image' has shape \(1, 8, 8\)"):
            Cls().check_data({"image": (1, 8, 8), "cls": ()}, CLS_META, ["image", "cls"])

    def test_ragged_tuple_meta_and_blob_skip(self):
        meta = {**CLS_META, "kpt_shape": (17, 3)}
        fields = ["image", "bboxes", "batch_idx", "keypoints", "mask"]
        Det().check_data({"image": (3, 8, 8), "bboxes": (4,), "batch_idx": (), "keypoints": (17, 3)}, meta, fields)
        with self.assertRaisesRegex(TaskContractError, "keypoints"):
            Det().check_data({"image": (3, 8, 8), "bboxes": (4,), "batch_idx": (), "keypoints": (5, 3)}, meta, fields)

    def test_free_dims_bind_across_fields(self):
        class Seg(TaskContract):
            batch = {"image": Tensor("B 3 H W"), "mask": Tensor("B H W", target=True)}

        Seg().check_data({"image": (3, 8, 6), "mask": (8, 6)}, {}, ["image", "mask"])
        with self.assertRaisesRegex(TaskContractError, "'mask'"):
            Seg().check_data({"image": (3, 8, 6), "mask": (8, 7)}, {}, ["image", "mask"])

    def test_trailing_star_matches_any_rank(self):
        class Rl(TaskContract):
            batch = {"action": Tensor("B *")}

        Rl().check_data({"action": ()}, {}, ["action"])
        Rl().check_data({"action": (6,)}, {}, ["action"])

    def test_list_field_is_presence_checked_only(self):
        Cls().check_data({}, CLS_META, ["image", "cls"])


if __name__ == "__main__":
    unittest.main()
