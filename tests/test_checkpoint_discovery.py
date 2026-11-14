"""All suffixes, future steps, path updates and content-only discovery dedup."""
from types import SimpleNamespace

from scripts.discover_external_eval_checkpoints import inventory, scan


def files(folder, sha="a", metadata="m"):
    return {
        f"ckpt/{folder}/lora_adapter/adapter_config.json": dict(blob_id="cfg", size=10),
        f"ckpt/{folder}/lora_adapter/adapter_model.safetensors": dict(sha256=sha, size=100),
        f"ckpt/{folder}/metadata.json": dict(blob_id=metadata, size=10),
    }


def test_inventory_has_no_step_or_best_filter_and_reports_incomplete_layouts():
    source = {**files("checkpoint-2500"), **files("checkpoint-50000_best", "b"), **files("checkpoint-final", "c")}
    source["ckpt/checkpoint-60000/metadata.json"] = dict(blob_id="only-meta", size=10)
    rows, missing = inventory("owner/model", "fixed", source)
    assert len(rows) == 3 and {r["step"] for r in rows} == {2500, 50000, None}
    assert missing[0]["checkpoint_path"] == "ckpt/checkpoint-60000"
    assert all("/tree/fixed/" in r["checkpoint_url"] for r in rows)


class Api:
    revision = "first"
    weight = "a"
    metadata = "m"

    def model_info(self, repo):
        return SimpleNamespace(sha=self.revision)

    def list_repo_tree(self, repo, **kwargs):
        assert kwargs["revision"] == self.revision
        for path, entry in files("checkpoint-2500", self.weight, self.metadata).items():
            yield SimpleNamespace(path=path, size=entry["size"], blob_id=entry.get("blob_id"),
                                  lfs=SimpleNamespace(sha256=entry["sha256"]) if "sha256" in entry else None)


def test_document_only_revision_is_seen_but_never_assumed_evaluated():
    api = Api()
    initial = scan(api, ["owner/model"])
    api.revision = "readme-update"
    seen = scan(api, ["owner/model"], initial)
    assert seen["new_or_updated_content_ids"] == []
    assert seen["checkpoints"][0]["evaluation_status"] == "requires_acceptance_check"
    assert seen["checkpoints"][0]["revision"] == "readme-update"
    api.weight = "updated-weight"
    updated = scan(api, ["owner/model"], seen)
    assert len(updated["new_or_updated_content_ids"]) == 1
    assert updated["checkpoints"][0]["discovery_change"] == "updated_path"
    assert len(updated["checkpoint_history"]) == 2
    assert updated["checkpoint_history"][0]["revision"] == "first"
    api.metadata = "new-runtime-prompts"
    assert scan(api, ["owner/model"], updated)["new_or_updated_content_ids"]
