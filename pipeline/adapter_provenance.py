from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROVENANCE_FILENAME = "adapter_provenance.json"
PROVENANCE_SCHEMA_VERSION = "adapter_provenance_v1"
STAGE_ORDER = ("cpt", "fact_sft", "dpo", "grpo")
_STAGE_INDEX = {stage: index for index, stage in enumerate(STAGE_ORDER)}
_MANIFEST_KEYS = {"schema_version", "provenance_complete", "stages"}
_STAGE_KEYS = {"stage", "status", "created_at_utc"}
_REQUIRED_STAGE_KEYS = {"stage", "status"}
_LEGACY_METADATA_FILENAMES = (
    "grpo_training_metadata.json",
    "dpo_training_metadata.json",
    "fact_sft_training_metadata.json",
    "training_metadata.json",
    "base_dpo_training_metadata.json",
    "base_fact_sft_training_metadata.json",
    "base_cpt_training_metadata.json",
    "base_training_metadata.json",
)


@dataclass(frozen=True)
class ProvenanceStage:
    stage: str
    status: str = "completed"
    created_at_utc: str | None = None

    def as_dict(self) -> dict[str, str]:
        data = {"stage": self.stage, "status": self.status}
        if self.created_at_utc is not None:
            data["created_at_utc"] = self.created_at_utc
        return data


@dataclass(frozen=True)
class ProvenanceResolution:
    stage_records: tuple[ProvenanceStage, ...]
    source: str
    complete: bool
    warnings: tuple[str, ...] = ()

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(record.stage for record in self.stage_records)

    def report_data(self) -> dict[str, Any]:
        return {
            "applied_stages": list(self.stages),
            "provenance_source": self.source,
            "provenance_complete": self.complete,
            "provenance_warnings": list(self.warnings),
        }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _normalise_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return candidate


def validate_adapter_provenance_manifest(payload: Any) -> dict[str, Any]:
    """Validate and return a whitelist-only provenance manifest."""
    if not isinstance(payload, dict):
        raise ValueError("Adapter provenance manifest must be a JSON object.")
    if set(payload) != _MANIFEST_KEYS:
        raise ValueError("Adapter provenance manifest fields do not match the v1 schema.")
    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported adapter provenance schema version.")
    if not isinstance(payload.get("provenance_complete"), bool):
        raise ValueError("Adapter provenance completeness must be boolean.")

    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("Adapter provenance stages must be a non-empty list.")

    records: list[ProvenanceStage] = []
    previous_index = -1
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict):
            raise ValueError("Each adapter provenance stage must be a JSON object.")
        keys = set(raw_stage)
        if not _REQUIRED_STAGE_KEYS.issubset(keys) or not keys.issubset(_STAGE_KEYS):
            raise ValueError("Adapter provenance stage fields do not match the v1 schema.")
        stage = raw_stage.get("stage")
        if stage not in _STAGE_INDEX:
            raise ValueError("Adapter provenance contains an unknown stage.")
        stage_index = _STAGE_INDEX[stage]
        if stage_index <= previous_index:
            raise ValueError("Adapter provenance stages must be unique and ordered.")
        if raw_stage.get("status") != "completed":
            raise ValueError("Adapter provenance only accepts completed stages.")
        timestamp = None
        if "created_at_utc" in raw_stage:
            timestamp = _normalise_timestamp(raw_stage.get("created_at_utc"))
            if timestamp is None:
                raise ValueError("Adapter provenance stage timestamp is invalid.")
        records.append(ProvenanceStage(stage=stage, created_at_utc=timestamp))
        previous_index = stage_index

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "provenance_complete": payload["provenance_complete"],
        "stages": [record.as_dict() for record in records],
    }


def inspect_peft_adapter(path: Path) -> str | None:
    """Return a loadability issue, or None for a structurally valid PEFT adapter."""
    if not path.exists():
        return "directory does not exist"
    if not path.is_dir():
        return "path is not a directory"
    missing: list[str] = []
    if not (path / "adapter_config.json").is_file():
        missing.append("adapter_config.json")
    if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        missing.append("adapter_model.safetensors or adapter_model.bin")
    if missing:
        return f"missing {', '.join(missing)} at adapter directory root"
    return None


def _classify_legacy_metadata(payload: dict[str, Any]) -> str | None:
    if payload.get("status") != "completed":
        return None
    if isinstance(payload.get("grpo"), dict):
        return "grpo"
    if isinstance(payload.get("dpo"), dict):
        return "dpo"
    if payload.get("assistant_only_loss") is True and payload.get("full_token_loss") is False:
        return "fact_sft"
    if payload.get("full_token_loss") is True and payload.get("assistant_only_loss") is False:
        return "cpt"
    return None


def _legacy_base_path(raw_path: str, metadata_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    candidates = ((Path.cwd() / candidate).resolve(), (metadata_dir / candidate).resolve())
    return next((path for path in candidates if path.exists()), candidates[0])


def _legacy_resolution(adapter_dir: Path, *, max_depth: int) -> ProvenanceResolution:
    records: dict[str, ProvenanceStage] = {}
    warnings: list[str] = []
    visited: set[Path] = set()
    complete = True

    def warn(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    def visit(directory: Path, depth: int, ancestry: frozenset[Path]) -> None:
        nonlocal complete
        try:
            resolved = directory.resolve()
        except (OSError, RuntimeError):
            complete = False
            warn("Legacy provenance contains an invalid base adapter path.")
            return
        if resolved in ancestry:
            complete = False
            warn("Legacy provenance contains a base adapter cycle.")
            return
        if resolved in visited:
            return
        if depth >= max_depth:
            complete = False
            warn("Legacy provenance exceeded the maximum base adapter depth.")
            return
        if not resolved.is_dir():
            complete = False
            warn("Legacy provenance references a missing base adapter directory.")
            return

        visited.add(resolved)
        next_ancestry = ancestry | {resolved}
        base_paths: set[Path] = set()
        for filename in _LEGACY_METADATA_FILENAMES:
            metadata_path = resolved / filename
            if not metadata_path.is_file():
                continue
            try:
                payload = _read_json(metadata_path)
            except (OSError, UnicodeError, json.JSONDecodeError):
                complete = False
                warn("Legacy provenance metadata could not be parsed.")
                continue
            if not isinstance(payload, dict):
                complete = False
                warn("Legacy provenance metadata is not a JSON object.")
                continue

            stage = _classify_legacy_metadata(payload)
            if stage is None:
                complete = False
                warn("Legacy provenance metadata does not identify a completed stage.")
            elif stage not in records:
                records[stage] = ProvenanceStage(
                    stage=stage,
                    created_at_utc=_normalise_timestamp(payload.get("created_at_utc")),
                )

            raw_base = payload.get("base_adapter_dir")
            if raw_base in (None, ""):
                continue
            if not isinstance(raw_base, str):
                complete = False
                warn("Legacy provenance contains an invalid base adapter path.")
                continue
            try:
                base_paths.add(_legacy_base_path(raw_base, resolved))
            except (OSError, RuntimeError, ValueError):
                complete = False
                warn("Legacy provenance contains an invalid base adapter path.")

        for base_path in sorted(base_paths, key=str):
            visit(base_path, depth + 1, next_ancestry)

    visit(adapter_dir, 0, frozenset())
    ordered = tuple(records[stage] for stage in STAGE_ORDER if stage in records)
    if not ordered:
        complete = False
        warn("No recognizable adapter provenance metadata was found.")
    return ProvenanceResolution(
        stage_records=ordered,
        source="legacy_recursive" if ordered else "none",
        complete=complete,
        warnings=tuple(warnings),
    )


def resolve_adapter_provenance(adapter_dir: Path, *, max_depth: int = 16) -> ProvenanceResolution:
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer.")
    adapter_dir = Path(adapter_dir)
    manifest_path = adapter_dir / PROVENANCE_FILENAME
    manifest_warning: str | None = None
    if manifest_path.is_file():
        try:
            manifest = validate_adapter_provenance_manifest(_read_json(manifest_path))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            manifest_warning = "Adapter provenance manifest is invalid; legacy metadata fallback was used."
        else:
            records = tuple(
                ProvenanceStage(
                    stage=item["stage"],
                    created_at_utc=item.get("created_at_utc"),
                )
                for item in manifest["stages"]
            )
            warnings: tuple[str, ...] = ()
            if not manifest["provenance_complete"]:
                warnings = ("Adapter provenance manifest declares incomplete ancestry.",)
            return ProvenanceResolution(
                stage_records=records,
                source="manifest",
                complete=manifest["provenance_complete"],
                warnings=warnings,
            )

    legacy = _legacy_resolution(adapter_dir, max_depth=max_depth)
    if manifest_warning is None:
        return legacy
    return ProvenanceResolution(
        stage_records=legacy.stage_records,
        source=legacy.source,
        complete=legacy.complete,
        warnings=(manifest_warning, *legacy.warnings),
    )


def write_adapter_provenance(
    output_dir: Path,
    *,
    stage: str,
    created_at_utc: str,
    base_adapter_dir: Path | None = None,
) -> dict[str, Any]:
    if stage not in _STAGE_INDEX:
        raise ValueError("Unknown adapter provenance stage.")
    timestamp = _normalise_timestamp(created_at_utc)
    if timestamp is None:
        raise ValueError("Adapter provenance stage timestamp is invalid.")

    previous_records: tuple[ProvenanceStage, ...] = ()
    complete = True
    if base_adapter_dir is not None:
        previous = resolve_adapter_provenance(Path(base_adapter_dir))
        previous_records = previous.stage_records
        complete = previous.complete
    if previous_records:
        previous_stage_index = _STAGE_INDEX[previous_records[-1].stage]
        current_stage_index = _STAGE_INDEX[stage]
        if previous_stage_index > current_stage_index:
            raise ValueError("Adapter provenance stage cannot precede its base adapter stages.")
        if previous_stage_index == current_stage_index:
            previous_records = previous_records[:-1]

    manifest = validate_adapter_provenance_manifest(
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "provenance_complete": complete,
            "stages": [
                *(record.as_dict() for record in previous_records),
                ProvenanceStage(stage=stage, created_at_utc=timestamp).as_dict(),
            ],
        }
    )
    _write_json(Path(output_dir) / PROVENANCE_FILENAME, manifest)
    return manifest
