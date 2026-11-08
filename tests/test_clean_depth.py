"""CPU cleaner and read-only loader admission tests; no training runs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from open_vision_banana.data import clean_depth as clean


class CleanDepthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def row(self, name="a", invalid=0, task="depth", split="train"):
        depth = np.ones((10, 10), dtype=np.float32)
        depth.ravel()[:invalid] = 0
        np.save(self.root / f"{name}.npy", depth)
        for suffix in ("input", "target"):
            Image.new("RGB", (10, 10)).save(self.root / f"{name}-{suffix}.png")
        return dict(
            input=f"{name}-input.png",
            target=f"{name}-target.png",
            metric_depth=f"{name}.npy",
            task=task,
            split=split,
            sample_id=name,
            source_dataset="hypersim",
        )

    def manifest(self, rows, name="pairs.jsonl"):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_depth_aliases_require_native_evidence(self):
        for task in ("depth", "metric_depth", "depth_estimation", "depth_edge"):
            row = self.row(task=task)
            row.pop("metric_depth")
            self.assertTrue(clean.task_is_depth(row))
            with self.assertRaisesRegex(ValueError, "requires metric_depth"):
                clean.build_clean_manifest([self.manifest([row])], self.root / task)

    def test_admitted_split_normalized_with_original_audit(self):
        missing = self.row("missing")
        missing.pop("split")
        alias = self.row("alias", split="training")
        official = self.row("official", split="official_train")
        official["source_split"] = "official_train"
        path = self.manifest([missing, alias, official])
        output = self.root / "clean"
        report = clean.build_clean_manifest([path], output)
        kept = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
        audits = [
            json.loads(line) for line in (output / "admitted_audit.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["split"] for row in kept], ["train"] * 3)
        self.assertEqual(report["counts"]["assumed_train_rows"], 1)
        self.assertEqual(audits[0]["original_split_labels"], {})
        self.assertEqual(audits[1]["original_split_labels"], {"split": "training"})
        self.assertEqual(kept[2]["source_split"], "official_train")
        self.assertNotIn("split", missing)

    def test_dirty_depth_edge_rejected(self):
        path = self.manifest([self.row("edge", invalid=3, task="depth_edge")])
        report = clean.build_clean_manifest([path], self.root / "clean")
        self.assertEqual(report["counts"]["depth_rejected"], 1)
        self.assertEqual(report["counts"]["depth_rows_scanned"], 1)
        self.assertEqual((self.root / "clean/pairs.jsonl").read_text(), "")

    def test_truncated_native_png_rejected_even_with_rgb_tolerance(self):
        row = self.row()
        row.pop("metric_depth")
        row["valid_mask"] = "native_mask.png"
        path = self.root / row["valid_mask"]
        Image.new("L", (10, 10), 255).save(path)
        path.write_bytes(path.read_bytes()[:-16])
        with patch("PIL.ImageFile.LOAD_TRUNCATED_IMAGES", True), self.assertRaises(ValueError):
            clean.validate_training_depth_row(row, self.root)

    def test_exact_threshold(self):
        for invalid, passes in ((0, True), (2, True), (3, False)):
            _, stats = clean.assess_depth_row(self.row(invalid=invalid), self.root)
            self.assertEqual(stats["passes"], passes)
        self.assertTrue(clean.within_invalid_fraction(2, 100))
        self.assertFalse(clean.within_invalid_fraction(20000001, 1000000000))

    def test_nonfinite_and_nonpositive(self):
        row = self.row()
        depth = np.ones((10, 10), dtype=np.float32)
        depth.ravel()[:5] = [np.nan, np.inf, -np.inf, 0, -1]
        np.save(self.root / "a.npy", depth)
        _, stats = clean.assess_depth_row(row, self.root)
        self.assertEqual(stats["invalid_pixels"], 5)
        self.assertFalse(stats["passes"])

    def test_mask_depth_intersection(self):
        row = self.row(invalid=1)
        mask = np.ones((10, 10), dtype=np.uint8) * 255
        mask.ravel()[1:3] = 0
        Image.fromarray(mask).save(self.root / "mask.png")
        row["valid_mask"] = "mask.png"
        valid, stats = clean.assess_depth_row(row, self.root)
        self.assertEqual(stats["invalid_pixels"], 3)
        self.assertEqual(valid.dtype, np.bool_)

    def test_mask_only_scalar_png_and_npy(self):
        for suffix in ("png", "npy"):
            mask = np.ones((10, 10), dtype=np.uint8)
            mask.ravel()[:2] = 0
            path = self.root / f"mask.{suffix}"
            if suffix == "png":
                Image.fromarray(mask).save(path)
            else:
                np.save(path, mask)
            _, stats = clean.assess_depth_row({"valid_mask": path.name}, self.root)
            self.assertTrue(stats["passes"])
            _, stats = clean.assess_depth_row({"metric_depth": path.name}, self.root)
            self.assertTrue(stats["passes"])

    def test_fail_closed_missing_corrupt_rgb(self):
        row = self.row()
        for changes in (
            {"metric_depth": "missing.npy"},
            {"metric_depth": "a-input.png"},
            {"metric_depth": ""},
            {"valid_mask": "missing.npy"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                clean.assess_depth_row(dict(row, **changes), self.root)
        (self.root / "a.npy").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            clean.assess_depth_row(row, self.root)

    def test_shape_mismatch(self):
        row = self.row()
        np.save(self.root / "mask.npy", np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "shapes"):
            clean.assess_depth_row(dict(row, valid_mask="mask.npy"), self.root)
        Image.new("RGB", (2, 2)).save(self.root / "a-input.png")
        with self.assertRaisesRegex(ValueError, "geometry"):
            clean.assess_depth_row(row, self.root)

    def test_relative_paths_counts_hashes_and_no_overwrite(self):
        rows = [
            self.row("good", 2),
            self.row("bad", 3),
            self.row("normal", 100, "normal"),
            self.row("validation", 0, split="val"),
            self.row("test", 0, split="test"),
        ]
        path = self.manifest(rows)
        before = path.read_bytes()
        output = self.root / "clean"
        report = clean.build_clean_manifest([path], output, expected_depth_rejections=1)
        self.assertEqual(report["counts"]["kept_rows"], 2)
        self.assertEqual(report["counts"]["non_train_split"], 2)
        kept = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
        for row in kept:
            for key in ("input", "target", "metric_depth"):
                self.assertTrue(Path(row[key]).is_absolute())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(report["sources"][0]["sha256"], clean.file_sha256(path))
        complete = json.loads((output / "COMPLETE.json").read_text())
        for name, sha in complete.items():
            self.assertEqual(clean.file_sha256(output / name), sha)
        with self.assertRaises(FileExistsError):
            clean.build_clean_manifest([path], output)

    def test_expected_count_assertion_no_output(self):
        path = self.manifest([self.row()])
        with self.assertRaisesRegex(ValueError, "measured 0"):
            clean.build_clean_manifest([path], self.root / "clean", expected_depth_rejections=1)
        self.assertFalse((self.root / "clean").exists())

    def test_duplicates_and_conflicts(self):
        row = self.row()
        path = self.manifest([row, row])
        report = clean.build_clean_manifest([path], self.root / "clean")
        self.assertEqual(report["counts"]["duplicate"], 1)
        path = self.manifest([row, dict(row, prompt="conflicting")])
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            clean.build_clean_manifest([path], self.root / "conflict")

    def test_heldout_duplicate_order_independent(self):
        row = self.row()
        for index, rows in enumerate(
            ([row, dict(row, split="val")], [dict(row, split="test"), row])
        ):
            with self.assertRaisesRegex(ValueError, "Train/heldout duplicate"):
                clean.build_clean_manifest([self.manifest(rows)], self.root / str(index))

    def test_shared_input_distinct_task_preserved(self):
        row = self.row()
        report = clean.build_clean_manifest(
            [self.manifest([row, dict(row, task="normal")])], self.root / "clean"
        )
        self.assertEqual(report["counts"]["kept_rows"], 2)
        self.assertEqual(report["counts"]["depth_rows_scanned"], 1)

    def test_source_identity_conflicts_across_assets(self):
        first, second = self.row("first"), self.row("second")
        second["sample_id"] = first["sample_id"]
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            clean.build_clean_manifest([self.manifest([first, second])], self.root / "clean")

    def test_composition_roots_and_source_split(self):
        first, second = self.row("first"), self.row("second")
        second["source_split"] = "official_test"
        a = self.manifest([first], "a.jsonl")
        b = self.manifest([second], "b.jsonl")
        report = clean.build_clean_manifest(
            [a, b], self.root / "clean", roots=[self.root, self.root]
        )
        self.assertEqual(report["counts"]["kept_rows"], 1)
        self.assertEqual(len(report["sources"]), 2)

    def test_legacy_discovery_bound_to_native_bytes(self):
        (self.root / "targets").mkdir()
        (self.root / "metric_depth").mkdir()
        Image.new("RGB", (10, 10), "black").save(self.root / "targets/x.png")
        np.save(self.root / "metric_depth/x.npy", np.ones((10, 10)))
        row = dict(target="targets/x.png", task="depth", sample_id="x", source_dataset="hypersim")
        _, stats = clean.assess_depth_row(row, self.root)
        self.assertTrue(stats["passes"])
        self.assertIn("metric_depth", stats["validity_sources"])


class TrainingAdmissionTests(unittest.TestCase):
    """Actual loaders are optional only when their installed dependencies are absent."""

    setUp = CleanDepthTests.setUp
    tearDown = CleanDepthTests.tearDown
    row = CleanDepthTests.row
    manifest = CleanDepthTests.manifest

    @classmethod
    def setUpClass(cls):
        try:
            import importlib

            cls.loaders = [
                importlib.import_module(name).ManifestVisionDataset
                for name in (
                    "open_vision_banana.training.trainer",
                    "scripts.train_sana_512",
                    "scripts.train_flux",
                )
            ]
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"Training loader dependency unavailable: {exc}") from exc

    def test_dirty_after_cap_rejects_entire_manifest(self):
        path = self.manifest([self.row("clean"), self.row("dirty", invalid=3)])
        before = path.read_bytes()
        for loader in self.loaders:
            with (
                self.subTest(loader=loader.__module__),
                self.assertRaisesRegex(ValueError, "exceeds 2%"),
            ):
                loader(path, 7, "train", {"depth"}, max_samples=1)
        self.assertEqual(path.read_bytes(), before)

    def test_exact_two_percent_accepts_without_geometry_changes(self):
        import torch

        path = self.manifest([self.row(invalid=2)])
        pixels = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
        Image.fromarray(pixels).save(self.root / "a-target.png")
        for loader in self.loaders:
            kwargs = (
                {"aspect_preserving_crop": True, "seed": 19} if loader is self.loaders[0] else {}
            )
            with self.subTest(loader=loader.__module__):
                accepted = loader(path, 7, "train", {"depth"}, **kwargs)[0]
                # Reproduce the pre-admission loader: only its read-only check is
                # suppressed, leaving the original processing method untouched.
                with patch("open_vision_banana.data.clean_depth.validate_training_depth_row"):
                    baseline = loader(path, 7, "train", {"depth"}, **kwargs)[0]
                self.assertEqual(accepted.keys(), baseline.keys())
                for key, value in accepted.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(value, baseline[key]), key)
                    else:
                        self.assertEqual(value, baseline[key])

    def test_validation_and_unrelated_tasks_unchanged(self):
        dirty_val = self.row("val", invalid=100, split="val")
        normal = self.row("normal", invalid=100, task="normal")
        dirty_val.pop("metric_depth")
        normal.pop("metric_depth")
        path = self.manifest([dirty_val, normal])
        for loader in self.loaders:
            self.assertEqual(len(loader(path, 7, "val", {"depth"})), 1)
            self.assertEqual(len(loader(path, 7, "train", {"normal"})), 1)

    def test_unknown_evidence_and_unavailable_pair_never_silently_skipped(self):
        for broken in ("missing_evidence", "missing_input", "bad_evidence"):
            row = self.row()
            if broken == "missing_evidence":
                row.pop("metric_depth")
            elif broken == "missing_input":
                row["input"] = "missing.png"
            else:
                (self.root / row["metric_depth"]).write_bytes(b"bad")
            path = self.manifest([row])
            for loader in self.loaders:
                with (
                    self.subTest(loader=loader.__module__, broken=broken),
                    self.assertRaises(
                        (ValueError, FileNotFoundError) if broken == "missing_input" else ValueError
                    ),
                ):
                    loader(path, 7, "train", {"depth"})

    def test_actual_raw_v3_and_tar_adapters_are_rejected_at_tuner_boundary(self):
        from open_vision_banana.data.datasets import HypersimDepthDataset, V3FlatDataset
        from open_vision_banana.data.tar_dataset import TarArchiveDataset, TarTargetDataset
        from open_vision_banana.training.tuner import VisionTaskDataset

        for cls in (
            HypersimDepthDataset,
            V3FlatDataset,
            TarArchiveDataset,
            TarTargetDataset,
            VisionTaskDataset,
        ):
            # Read-only guard examines metadata; no data loading, model, GPU or
            # training is needed to exercise these real adapter types.
            dataset = object.__new__(cls)
            dataset._reader = None
            dataset.task = "depth"
            dataset.samples = []
            for attribute in ("datasets", "vision_datasets", "task_datasets"):
                wrapped = SimpleNamespace(**{attribute: {"depth": dataset}})
                with self.assertRaisesRegex(ValueError, "clean --pair-manifest"):
                    clean.validate_training_dataset(wrapped)
        clean.validate_training_dataset(SimpleNamespace(task="normal", samples=[]))
        with self.assertRaisesRegex(ValueError, "legacy tar/V3"):
            clean.require_depth_manifest(["depth"], "legacy tar/V3")


if __name__ == "__main__":
    unittest.main()
