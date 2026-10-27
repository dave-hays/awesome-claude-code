"""Stage handoff manifests.

A stage writes ``stage_N_manifest.json`` with its gate results and the SHA-256 of each output.
The next stage calls ``require`` before reading anything: it fails if the manifest is missing,
if any gate failed, or if an output changed after the manifest was written. The digests make
the handoff tamper-evident; they are not a cryptographic signature with a key.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from . import config


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(stage: int, workspace: Path = config.WORKSPACE) -> Path:
    return workspace / f"stage_{stage}_manifest.json"


def write_manifest(
    stage: int, outputs: list[Path], gates: dict[str, bool], details: dict, workspace: Path = config.WORKSPACE
) -> Path:
    """Record gate results and output digests; raise GateError after writing if any gate failed."""
    passed = all(gates.values())
    record = {
        "stage": stage,
        "status": "passed" if passed else "failed",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "gates": gates,
        "details": details,
        "config_sha256": sha256(Path(config.__file__)),
        "outputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in outputs],
    }
    path = manifest_path(stage, workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    if not passed:
        failed = ", ".join(k for k, v in gates.items() if not v)
        raise GateError(f"stage {stage} gate failed: {failed}")
    return path


def require(stage: int, workspace: Path = config.WORKSPACE) -> dict:
    """Load a passed manifest whose outputs are unchanged."""
    path = manifest_path(stage, workspace)
    if not path.exists():
        raise GateError(f"stage {stage} has no manifest at {path}")
    record = json.loads(path.read_text())
    if record["status"] != "passed":
        raise GateError(f"stage {stage} manifest status is {record['status']}")
    for out in record["outputs"]:
        p = Path(out["path"])
        if not p.exists() or sha256(p) != out["sha256"]:
            raise GateError(f"stage {stage} output changed or missing: {p}")
    return record
