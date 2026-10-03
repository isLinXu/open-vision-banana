"""Portable report aliases without machine roots; actual IO paths stay private.

Aliases describe the supplied dataset/checkpoint roots, not files copied into
the output directory. Re-run using the original profile and CLI bindings.
"""
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


def _absolute(value):
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


class PortableReport:
    def __init__(self, config, project_root):
        self.anchors = []
        dataset = config.get("dataset", {})
        dataset_root = dataset.get("root") or (Path(dataset["manifest"]).parent if dataset.get("manifest") else None)
        if dataset_root:
            self.anchors.append((Path(dataset_root).resolve(), "dataset"))
        for model in config.get("models", []):
            prefix = "checkpoints/" + model["id"]
            checkpoint = model.get("checkpoint")
            if checkpoint:
                path = Path(checkpoint).resolve()
                self.anchors.append((path if path.is_dir() else path.parent, prefix))
            provenance = model.get("options", {}).get("weights_provenance_path")
            if provenance:
                self.anchors.append((Path(provenance).resolve().parent, prefix + "/cache"))
            base = model.get("base_model_id")
            if base and _absolute(base):
                self.anchors.append((Path(base).resolve(), "base-models/" + model["id"]))
        self.anchors.append((Path(project_root).resolve(), "."))
        self.anchors.sort(key=lambda item: len(item[0].parts), reverse=True)

    def path(self, value):
        path = Path(value)
        for root, alias in self.anchors:
            try:
                relative = path.resolve().relative_to(root)
            except ValueError:
                continue
            return (Path(alias) / relative).as_posix()
        # Never include another volume, home directory or parent hierarchy.
        name = PureWindowsPath(value).name if PureWindowsPath(value).is_absolute() else path.name
        return "external/" + name

    def clean(self, value):
        if isinstance(value, dict):
            return {key: self.clean(item) for key, item in value.items() if key != "author"}
        if isinstance(value, (list, tuple)):
            return [self.clean(item) for item in value]
        if not isinstance(value, str):
            return value
        if _absolute(value):
            return self.path(value)
        # Exceptions may embed paths in prose; remove their machine roots too.
        for root, alias in self.anchors:
            for spelling in (str(root), root.as_posix()):
                value = value.replace(spelling, alias)
        # Third-party errors can mention paths outside the supplied roots.
        def quoted_path(match):
            quote, content = match.group(1), match.group(2)
            return quote + (self.path(content) if _absolute(content) else content) + quote

        value = re.sub(r"(['\"])([^'\"\r\n]+)\1", quoted_path, value)
        value = re.sub(r"(?<!\w)(?:[A-Za-z]:[\\/]|\\\\)[^'\"\r\n]+",
                       lambda match: self.path(match.group(0)), value)
        return value
