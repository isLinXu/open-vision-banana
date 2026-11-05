"""Offline tests for merged pipeline publication preflight (no auth or upload)."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_to_hub.py"
SPEC = importlib.util.spec_from_file_location("publish_merged_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_tensor(path, key="weight"):
    header = json.dumps({key: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")


def make_pipeline(root):
    checks = ("download_sha256", "adapter_loaded_exactly", "all_lora_merge_formulas",
              "all_exported_tensor_hashes", "copied_component_hashes", "offline_component_loading")
    metadata = {"base_repo": "Qwen/Qwen-Image-2.1",
                "validation": {check: True for check in checks}}
    (root / "merge_metadata.json").write_text(json.dumps(metadata))
    (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
    (root / "README.md").write_text("Built with Qwen. Test fixture only.")
    (root / "LICENSE").write_text("Qwen RESEARCH LICENSE AGREEMENT")
    (root / "NOTICE.txt").write_text("Test fixture notice")
    (root / "MERGE_VALIDATED").write_text("Test fixture completed")
    for component, index in (("transformer", "diffusion_pytorch_model.safetensors.index.json"),
                             ("text_encoder", "model.safetensors.index.json")):
        directory = root / component
        write_tensor(directory / "shard.safetensors")
        (directory / index).write_text(json.dumps({"weight_map": {"weight": "shard.safetensors"}}))
        (directory / "config.json").write_text("{}")
    (root / "transformer" / "shard.safetensors.NOTICE.txt").write_text("Modified fixture shard")
    write_tensor(root / "vae" / "diffusion_pytorch_model.safetensors")
    (root / "vae" / "config.json").write_text("{}")
    for component, config in (("processor", "preprocessor_config.json"),
                              ("scheduler", "scheduler_config.json")):
        (root / component).mkdir()
        (root / component / config).write_text("{}")


class PublishMergedPreflightTests(unittest.TestCase):
    def test_valid_minimal_fixture_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pipeline(root)
            manifest = MODULE._merged_upload_manifest(root)
            for name, (size, digest) in manifest.items():
                self.assertEqual(size, (root / name).stat().st_size)
                self.assertEqual(digest, MODULE._sha256(root / name))
            self.assertIn("vae/diffusion_pytorch_model.safetensors", manifest)

    def test_cache_and_unrelated_files_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pipeline(root)
            (root / ".env").write_text("never upload")
            (root / "unrelated.py").write_text("never upload")
            (root / "processor" / ".cache").mkdir()
            (root / "processor" / ".cache" / "upload.json").write_text("never upload")
            manifest = MODULE._merged_upload_manifest(root)
            self.assertNotIn(".env", manifest)
            self.assertNotIn("unrelated.py", manifest)
            self.assertNotIn("processor/.cache/upload.json", manifest)

    def test_missing_weight_stops_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pipeline(root)
            (root / "transformer" / "shard.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "Missing weight shard"):
                MODULE._merged_upload_manifest(root)

    def test_missing_modified_shard_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pipeline(root)
            (root / "transformer" / "shard.safetensors.NOTICE.txt").unlink()
            with self.assertRaisesRegex(ValueError, "Missing modified-shard notice"):
                MODULE._merged_upload_manifest(root)

    def test_incomplete_merge_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pipeline(root)
            (root / "MERGE_INCOMPLETE").write_text("fixture interrupted")
            with self.assertRaises(ValueError):
                MODULE._merged_upload_manifest(root)

    def test_header_index_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pipeline(root)
            write_tensor(root / "transformer" / "shard.safetensors", key="different")
            with self.assertRaisesRegex(ValueError, "Weight keys disagree"):
                MODULE._merged_upload_manifest(root)

    def test_truncated_safetensors_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.safetensors"
            write_tensor(path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                MODULE._safetensors_keys(path)

    def test_unmerged_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.safetensors"
            write_tensor(path, key="layer.lora_A.weight")
            with self.assertRaisesRegex(ValueError, "unmerged adapter"):
                MODULE._safetensors_keys(path)

    def test_dry_run_does_not_authenticate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pipeline(root)
            result = MODULE.publish_merged_pipeline(root, dry_run=True)
            self.assertEqual(result, 0)


class ModelScopePublishTests(unittest.TestCase):
    """Offline checks for the ModelScope path; no SDK install, auth or network."""

    class FakeApi:
        def __init__(self, entries):
            self.entries = entries

        def get_model_files(self, repo_id, recursive=True):
            self.asserted_recursive = recursive
            return self.entries

    def test_dry_run_needs_no_sdk_and_no_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "merged"
            root.mkdir()
            make_pipeline(root)
            result = MODULE.publish_merged_pipeline_modelscope(
                root, repo_id="someone/merged", dry_run=True)
            self.assertEqual(result, 0)
            self.assertFalse((Path(tmp) / ".modelscope_upload_merged").exists())

    def test_repo_id_must_be_namespaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "merged"
            root.mkdir()
            make_pipeline(root)
            for bad in ("merged", "someone/", "/merged", "a/b/c", "bad name/x"):
                with self.assertRaisesRegex(ValueError, "explicit --repo-id"):
                    MODULE.publish_merged_pipeline_modelscope(
                        root, repo_id=bad, dry_run=True)

    def test_remote_files_skip_directories(self):
        api = self.FakeApi([
            {"Path": "transformer", "Type": "tree"},
            {"Path": "vae/diffusion_pytorch_model.safetensors", "Size": 12},
        ])
        remote = MODULE._ms_remote_files(api, "someone/merged")
        self.assertEqual(set(remote), {"vae/diffusion_pytorch_model.safetensors"})
        self.assertTrue(api.asserted_recursive)

    def test_compare_missing_size_and_sha(self):
        manifest = {"a.json": (5, "aa"), "b.bin": (9, "bb"), "c.txt": (1, "cc")}
        remote = {"a.json": {"Path": "a.json", "Size": 5, "Sha256": "AA"},
                  "b.bin": {"Path": "b.bin", "Size": 8}}
        matched, sha_checked, problems = MODULE._ms_compare(remote, manifest)
        self.assertEqual(matched, 1)
        self.assertEqual(sha_checked, 1)
        self.assertTrue(any(item.startswith("missing: c.txt") for item in problems))
        self.assertTrue(any("size differs: b.bin" in item for item in problems))

    def test_compare_reports_sha_mismatch(self):
        manifest = {"a.json": (5, "aa")}
        remote = {"a.json": {"Path": "a.json", "Size": 5, "Sha256": "ff"}}
        matched, _, problems = MODULE._ms_compare(remote, manifest)
        self.assertEqual(matched, 0)
        self.assertEqual(problems, ["sha256 differs: a.json"])

    def test_staging_uses_hardlinks_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "merged"
            root.mkdir()
            make_pipeline(root)
            manifest = MODULE._merged_upload_manifest(root)
            staging = MODULE._ms_staging(root, list(manifest))
            source = root / "vae" / "diffusion_pytorch_model.safetensors"
            linked = staging / "vae" / "diffusion_pytorch_model.safetensors"
            self.assertEqual(linked.stat().st_ino, source.stat().st_ino)
            self.assertEqual(sorted(p.name for p in staging.iterdir()),
                             sorted({name.split("/")[0] for name in manifest}))
            again = MODULE._ms_staging(root, list(manifest))
            self.assertEqual(again, staging)
            self.assertEqual(linked.stat().st_ino, source.stat().st_ino)

    def test_staging_conflict_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "merged"
            root.mkdir()
            make_pipeline(root)
            staging = Path(tmp) / ".modelscope_upload_merged" / "vae"
            staging.mkdir(parents=True)
            (staging / "diffusion_pytorch_model.safetensors").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "Staging file conflicts"):
                MODULE._ms_staging(root, ["vae/diffusion_pytorch_model.safetensors"])

    def test_credential_patterns_are_redacted(self):
        text = MODULE._redact("token ms-12345678-1234-1234-1234-123456789abc leaked")
        self.assertNotIn("ms-1234", text)
        self.assertIn("[REDACTED]", text)


if __name__ == "__main__":
    unittest.main()
