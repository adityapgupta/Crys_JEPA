"""Input validation and provenance for the descriptive embedding experiments."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reference_baselines.paths import RESULTS_DIR

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = RESULTS_DIR / "embedding_structure"


def arguments(description, data=False):
    parser = argparse.ArgumentParser(description=description)
    if data:
        parser.add_argument('--data', type=Path, required=True,
                            help='Directory containing embeddings.npy and metadata.csv')
    parser.add_argument('--out', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def read_metadata(path):
    # NaN is a valid chemical formula; do not interpret it as missing data.
    return pd.read_csv(path, keep_default_na=False)


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def provenance(paths, **settings):
    versions = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'networkx', 'pymatgen', 'pymatgen-core'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'input_sha256': {Path(p).name: digest(p) for p in paths},
            'versions': versions, 'settings': settings,
            'code_sha256': {p.name: digest(p) for p in sorted(HERE.glob('*.py'))}}


def save_json(path, result):
    Path(path).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def aligned_communities(meta, communities):
    """Join by material ID, not by an assumed meaning of num_id or CSV order."""
    for frame in (meta, communities):
        if 'material_id' not in frame or frame.material_id.duplicated().any():
            raise ValueError('Unique material_id values are required for alignment')
    if set(meta.material_id) != set(communities.material_id):
        raise ValueError('Community and metadata material IDs differ')
    aligned = communities.set_index('material_id').loc[meta.material_id].reset_index()
    for source, saved in [('spacegroup.number', 'spacegroup_number'),
                          ('pretty_formula', 'pretty_formula'),
                          ('chemical_system', 'chemical_system')]:
        if source in meta and saved in aligned:
            if not np.array_equal(meta[source].astype(str).to_numpy(), aligned[saved].astype(str).to_numpy()):
                raise ValueError(f'Metadata mismatch for {source}')
    return aligned
