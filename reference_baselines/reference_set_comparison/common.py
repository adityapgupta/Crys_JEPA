"""Small, dependency-light provenance and storage helpers for staged experiments."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from reference_baselines.core.io import sha256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def receipt(path, identity):
    """Return True only for a completed stage with unchanged inputs and outputs."""
    if not path.exists():
        return False
    saved = read_json(path)
    if saved["identity"] != identity:
        raise ValueError(f"Stale stage/cache: {path}; use a new run directory")
    for item in saved["outputs"]:
        p = path.parent / item["name"]
        if not p.is_file() or sha256(p) != item["sha256"]:
            raise ValueError(f"Modified or incomplete output: {p}")
    return True


def finish(path, identity, outputs, **extra):
    write_json(path, {"identity": identity, "outputs": [
        {"name": str(p.relative_to(path.parent)), "sha256": sha256(p)} for p in outputs
    ], **extra})


@contextmanager
def database(path):
    connection = sqlite3.connect(path, timeout=120)
    # DELETE journals work on the shared home filesystem; no WAL sidecar on NFS.
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_structures(connection):
    connection.execute("""CREATE TABLE IF NOT EXISTS structures (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, chemical_system TEXT NOT NULL,
        formula TEXT NOT NULL, natoms INTEGER NOT NULL,
        structure_hash TEXT NOT NULL, structure_json TEXT NOT NULL)""")
    connection.execute("CREATE INDEX IF NOT EXISTS structures_system ON structures(chemical_system, kind)")


def structure_record(identifier, kind, structure):
    data = structure.as_dict()
    return (identifier, kind, structure.composition.chemical_system,
            structure.composition.reduced_formula, len(structure), fingerprint(data), canonical(data))


def save_structure(connection, record):
    connection.execute("INSERT OR REPLACE INTO structures VALUES (?,?,?,?,?,?,?)", record)


@contextmanager
def timed(output, stage):
    start = time.monotonic()
    try:
        yield
    finally:
        with database(output / "runtime.sqlite") as db:
            db.execute("CREATE TABLE IF NOT EXISTS runtime(stage TEXT, seconds REAL, timestamp REAL)")
            db.execute("INSERT INTO runtime VALUES (?,?,?)", (stage, time.monotonic()-start, time.time()))
