import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

spec = importlib.util.spec_from_file_location("restore", Path(__file__).parents[1] / "scripts/restore_external_eval_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(snapshot, member="val/a.txt"):
    (snapshot / "manifests").mkdir(parents=True)
    (snapshot / "assets").mkdir()
    blob = b"test data"
    archive = snapshot / "assets/a.part000"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    record = dict(path=member, size=len(blob), sha256=hashlib.sha256(blob).hexdigest())
    (snapshot / "manifests/diode-files.json").write_text(json.dumps(dict(files=[record])))
    (snapshot / "manifests/diode-archive.json").write_text(json.dumps(dict(
        archive_sha256=module.sha256(archive), archive_size=archive.stat().st_size,
        chunks=[dict(path="assets/a.part000", size=archive.stat().st_size, sha256=module.sha256(archive))])))


def test_restore_checks_content_and_refuses_overwrite(tmp_path):
    snapshot, root = tmp_path / "snapshot", tmp_path / "root"
    fixture(snapshot)
    module.restore(snapshot, "diode", root)
    assert (root / "val/a.txt").read_bytes() == b"test data"
    module.restore(snapshot, "diode", root)
    (root / "val/a.txt").write_text("changed")
    with pytest.raises(ValueError, match="mismatched"):
        module.restore(snapshot, "diode", root)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/bad", "a\\b"])
def test_unsafe_member_rejected(tmp_path, path):
    snapshot = tmp_path / "snapshot"
    fixture(snapshot, path)
    with pytest.raises(ValueError, match="Unsafe"):
        module.restore(snapshot, "diode", tmp_path / "root")
