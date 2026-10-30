"""Map fixed-hull decomposition phases to known reference rows.

The target is the set of reference crystals that appear in a candidate's
equilibrium decomposition. Candidate stability labels are never read here.
"""

import json
import re

import numpy as np
import pandas as pd

from reference_baselines.paths import ROOT, RESULTS_DIR


SUFFIX = re.compile(r"-(?:GGA(?:\+U)?|R2SCAN)$")


def strip_entry_id(entry_id):
    """Return the Materials Project material ID without the energy suffix."""
    value = str(entry_id).strip()
    return SUFFIX.sub("", value)


def parse_decomposition(value):
    """Parse one saved decomposition string into phase records."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    records = json.loads(text)
    if not isinstance(records, list):
        raise ValueError("Decomposition must be a list")
    return records


def material_index():
    """Return reference row for each MP material ID."""
    metadata = pd.read_csv(ROOT / "embeddings/data/metadata.csv", usecols=["material_id"])
    ids = metadata.material_id.astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate reference material IDs")
    return {value: row for row, value in enumerate(ids)}


def decomposition_table():
    """Read only candidate rows and hull-phase strings.

    Training and graph construction must not open the candidate label table.
    That table contains candidate stability labels.
    """
    table = pd.read_csv(
        RESULTS_DIR / "candidate_decompositions.csv",
        usecols=["candidate_row", "decomposition"],
        keep_default_na=False,
    )
    np.testing.assert_array_equal(table.candidate_row, np.arange(len(table)))
    return table


def build_targets(rows, table=None, index=None):
    """Return phase positives, unmapped IDs, and per-row counts.

    ``positives`` maps a candidate row to the sorted reference rows of its
    mapped decomposition phases. Duplicate phases are counted once.
    Unmapped phases stay in the target set of no reference row. They are
    reported, not silently treated as negatives for a bank reference.
    """
    table = decomposition_table() if table is None else table
    index = material_index() if index is None else index
    positives, unmapped, phase_counts = {}, {}, {}
    for row in np.asarray(rows, dtype=np.int64):
        records = parse_decomposition(table.decomposition.iat[int(row)])
        mapped, missing = set(), []
        for record in records:
            entry_id = strip_entry_id(record["entry_id"])
            reference_row = index.get(entry_id)
            if reference_row is None:
                missing.append(entry_id)
            else:
                mapped.add(reference_row)
        positives[int(row)] = np.array(sorted(mapped), dtype=np.int64)
        unmapped[int(row)] = missing
        phase_counts[int(row)] = len(records)
    return positives, unmapped, phase_counts


def target_stats(rows, positives, unmapped, phase_counts):
    """Summarise target coverage without reading candidate stability."""
    rows = np.asarray(rows, dtype=np.int64)
    mapped_phases = np.array([len(positives[int(r)]) for r in rows])
    total_phases = np.array([phase_counts[int(r)] for r in rows])
    missing = np.array([len(unmapped[int(r)]) for r in rows])
    distinct = len({int(v) for r in rows for v in positives[int(r)]})
    return {
        "candidates": int(len(rows)),
        "candidates_with_any_mapped_phase": int((mapped_phases > 0).sum()),
        "candidates_with_all_phases_mapped": int(((missing == 0) & (total_phases > 0)).sum()),
        "mean_mapped_phases": float(mapped_phases.mean()) if len(rows) else 0.0,
        "mean_total_phases": float(total_phases.mean()) if len(rows) else 0.0,
        "unmapped_phase_count": int(missing.sum()),
        "distinct_mapped_reference_rows": int(distinct),
        "rule": "Phase entry_id without its -GGA, -GGA+U, or -R2SCAN suffix maps to one known reference row.",
        "label_limit": "A phase without a bank row has an unknown reference. Absent bank rows stay negative because they are not part of the saved decomposition.",
        "candidate_stability_labels_read": False,
    }
