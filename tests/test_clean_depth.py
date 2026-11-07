"""CPU synthetic tests; run with unittest or pytest, no training imports."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "clean_depth",
    Path(__file__).resolve().parents[1] / "src/open_vision_banana/data/clean_depth.py",
)
clean = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clean)


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


if __name__ == "__main__":
    unittest.main()
