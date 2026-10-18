"""Create element-blind, periodic SOAP fingerprints for known reference crystals."""

from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
import argparse
import json
import warnings

from dscribe.descriptors import SOAP
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.paths import ROOT, DATA_DIR, RESULTS_DIR


SETTINGS = {
    'physical': {'r_cut': 5., 'sigma': .3},
    'density_normalized': {'r_cut': 3., 'sigma': .15},
}
DESCRIPTORS = {}


def setup():
    global DESCRIPTORS
    DESCRIPTORS = {name: SOAP(species=['H'], periodic=True, n_max=6, l_max=6,
                              average='outer', **settings) for name, settings in SETTINGS.items()}


def geometry_atoms(structure, normalize_density):
    """Ignore species. Optionally set volume per atom to one length-unit cubed."""
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_atomic_numbers(np.ones(len(atoms), dtype=int))
    if normalize_density:
        scale = (atoms.get_volume() / len(atoms)) ** (1 / 3)
        atoms.set_cell(atoms.cell / scale, scale_atoms=True)
    return atoms


def normalized_fingerprint(atoms, name):
    result = np.asarray(DESCRIPTORS[name].create(atoms), dtype=np.float64)
    norm = np.linalg.norm(result)
    if not np.isfinite(result).all() or norm <= 0:
        raise ValueError('Non-finite or empty SOAP fingerprint.')
    return result / norm


def featurize(record):
    row, material_id, cif, formula, system = record
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            structure = Structure.from_str(cif, fmt='cif')
            if not structure.is_ordered or structure.volume <= 0:
                raise ValueError('Structure must be ordered with positive volume.')
            if structure.composition.chemical_system != system:
                raise ValueError('CIF elements disagree with embedding metadata.')
            if structure.composition.fractional_composition != Composition(formula).fractional_composition:
                raise ValueError('CIF composition disagrees with embedding metadata.')
            fingerprints = {name: normalized_fingerprint(geometry_atoms(structure, name == 'density_normalized'), name)
                            for name in SETTINGS}
        return row, fingerprints, {'reference_row': row, 'material_id': material_id, 'success': True,
                                   'num_atoms': len(structure), 'volume_per_atom': structure.volume / len(structure),
                                   'warning_count': len(caught), 'warning': str(caught[0].message) if caught else '',
                                   'error': ''}
    except Exception as exc:
        return row, None, {'reference_row': row, 'material_id': material_id, 'success': False,
                           'num_atoms': None, 'volume_per_atom': None, 'warning_count': None,
                           'warning': '', 'error': f'{type(exc).__name__}: {exc}'}


def verify_invariances():
    """Check the descriptor on bcc/fcc structures and equivalent representations."""
    setup()
    from ase.build import bulk
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.set_atomic_numbers(np.ones(len(atoms), dtype=int))
    maximum_error = 0.
    for name in SETTINGS:
        a = atoms.copy()
        if name == 'density_normalized':
            a.set_cell(a.cell / (a.get_volume() / len(a)) ** (1 / 3), scale_atoms=True)
        expected = normalized_fingerprint(a, name)
        variants = [a[::-1], a.repeat((2, 1, 1))]
        translated = a.copy(); translated.translate([.23, -.17, .4]); translated.wrap(); variants.append(translated)
        rotated = a.copy(); rotated.rotate(37, 'z', rotate_cell=True); variants.append(rotated)
        for other in variants:
            error = float(np.abs(normalized_fingerprint(other, name) - expected).max())
            maximum_error = max(maximum_error, error)
            np.testing.assert_allclose(normalized_fingerprint(other, name), expected, rtol=1e-6, atol=1e-7)
        other = bulk('Fe', 'bcc', a=2.87, cubic=True)
        other.set_atomic_numbers(np.ones(len(other), dtype=int))
        if name == 'density_normalized':
            other.set_cell(other.cell / (other.get_volume() / len(other)) ** (1 / 3), scale_atoms=True)
        assert np.linalg.norm(normalized_fingerprint(other, name) - expected) > .01
    # Geometry-only conversion must ignore chemical labels and normalized size.
    original = Structure.from_sites(AseAtomsAdaptor.get_structure(atoms).sites)
    changed = original.copy(); changed.replace_species({'H': 'Na'})
    np.testing.assert_allclose(normalized_fingerprint(geometry_atoms(original, True), 'density_normalized'),
                               normalized_fingerprint(geometry_atoms(changed, True), 'density_normalized'))
    expanded = original.copy(); expanded.scale_lattice(original.volume * 8)
    np.testing.assert_allclose(normalized_fingerprint(geometry_atoms(original, True), 'density_normalized'),
                               normalized_fingerprint(geometry_atoms(expanded, True), 'density_normalized'), atol=1e-7)
    return {'passed': True, 'translation_rotation_permutation_supercell_max_error': maximum_error,
            'element_blindness_checked': True, 'normalized_scale_invariance_checked': True,
            'bcc_fcc_distinguished': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DATA_DIR / 'mp_20_train.csv.gz')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'embeddings/data/metadata.csv')
    parser.add_argument('--out-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    verification = verify_invariances()
    source, metadata = pd.read_csv(args.source), pd.read_csv(args.metadata)
    if not source.material_id.is_unique or set(source.material_id) != set(metadata.material_id):
        raise ValueError('Source and reference ID sets differ.')
    aligned = source.set_index('material_id').loc[metadata.material_id].reset_index()
    pd.testing.assert_frame_equal(aligned[metadata.columns], metadata, check_dtype=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_features = DESCRIPTORS['physical'].get_number_of_features()
    arrays = {name: np.full((len(metadata), n_features), np.nan) for name in SETTINGS}
    records = [(i, r.material_id, r.cif, r.pretty_formula, r.chemical_system) for i, r in aligned.iterrows()]
    statuses = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=setup) as pool:
        for completed, (row, features, status) in enumerate(pool.map(featurize, records, chunksize=32), 1):
            statuses.append(status)
            if features is not None:
                for name in SETTINGS:
                    arrays[name][row] = features[name]
            if completed % 2000 == 0 or completed == len(records):
                print(f'Structure fingerprints: {completed:,}/{len(records):,}', flush=True)
    for name, values in arrays.items():
        np.save(args.out_dir / f'soap_{name}.npy', values)
    pd.DataFrame(statuses).to_csv(args.out_dir / 'structure_fingerprint_status.csv', index=False)
    report = {
        'reference_count': len(metadata), 'success_count': sum(s['success'] for s in statuses),
        'failure_count': sum(not s['success'] for s in statuses), 'feature_count': n_features,
        'metadata_alignment': 'all shared metadata fields checked after material-ID alignment',
        'descriptor': 'DScribe SOAP, single species, periodic, outer site averaging, unit vector normalization',
        'n_max': 6, 'l_max': 6, 'settings': SETTINGS,
        'density_normalization': 'divide all lengths by (volume/number_of_atoms)^(1/3); volume per atom becomes one',
        'geometry_only': 'all atom identities replaced with one identical label; this does not modify the source CIF',
        'verification': verification,
        'packages': {p: version(p) for p in ['dscribe', 'pymatgen', 'ase', 'numpy']},
        'sha256': {str(p.resolve()): sha256(p) for p in [args.source, args.metadata, Path(__file__).resolve()]},
        'limits': ['Averaged local environments are not a unique representation of an entire crystal.',
                   'Element identities are deliberately ignored; this does not assess bonding or stability.',
                   'Normalized fingerprints also ignore uniform changes in size; physical fingerprints retain size.'],
    }
    (args.out_dir / 'structure_fingerprint_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
