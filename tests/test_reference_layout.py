"""Regression checks for package imports, moved entry points, and shared paths."""

import importlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import tomllib
import unittest

from reference_baselines.paths import CONFIG_DIR, DATA_DIR, DOCS_DIR, PACKAGE_DIR, RESULTS_DIR, ROOT, SOURCE_PACKAGES


class ReferenceLayoutTests(unittest.TestCase):
    def test_all_modules_import_without_running_jobs(self):
        for folder in SOURCE_PACKAGES:
            for path in sorted((PACKAGE_DIR / folder).glob('*.py')):
                module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
                with self.subTest(module=module):
                    importlib.import_module(module)

    def test_paths_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                paths = importlib.reload(importlib.import_module('reference_baselines.paths'))
                self.assertEqual(paths.ROOT, ROOT)
                self.assertEqual(paths.DATA_DIR, PACKAGE_DIR / 'data')
                self.assertEqual(paths.RESULTS_DIR, PACKAGE_DIR / 'results')
            finally:
                os.chdir(previous)
        self.assertTrue(CONFIG_DIR.is_dir())
        self.assertTrue(DOCS_DIR.is_dir())

    def test_worker_functions_keep_importable_module_names(self):
        from reference_baselines.datasets.labels import candidate_entries, hull_group, init_hull
        from reference_baselines.evaluation.selected_duplicates import compare_group, load_folder
        from reference_baselines.structure.descriptors import featurize
        for function in [candidate_entries, hull_group, init_hull, compare_group, load_folder, featurize]:
            with self.subTest(function=function.__name__):
                self.assertIs(pickle.loads(pickle.dumps(function)), function)

    def test_training_settings_are_available(self):
        from reference_baselines.models.reference import MODEL_NAMES
        plan = tomllib.loads((CONFIG_DIR / 'training.toml').read_text())
        self.assertTrue(set(plan['models']).issubset(MODEL_NAMES))
        self.assertFalse(plan['enabled'])
        self.assertNotIn('candidate_only', plan['models'])
        self.assertNotIn('graph_transformer', plan['models'])

    def test_spawned_worker_resolves_package_imports(self):
        from reference_baselines.datasets.labels import hull_group, init_hull
        entries = [{'chemical_system': 'Li', 'formula': 'Li', 'entry_id': 'fixture-Li', 'energy_per_atom': 0.}]
        query = ('Li', [{'candidate_row': 0, 'formula': 'Li', 'corrected_energy_per_atom': .1}])
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_hull, initargs=(entries,)) as pool:
            result = pool.submit(hull_group, query).result(timeout=30)
        self.assertEqual(result[0]['hull_status'], 'ok')

    def test_package_command_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            command = [sys.executable, '-m', 'reference_baselines.training.train', '--help']
            result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--model', result.stdout)

    def test_legacy_screening_entry_point_is_safe_to_import(self):
        from reference_baselines.baselines.screening import main
        wrapper = importlib.import_module('V_screen_precomputed')
        self.assertIs(wrapper.main, main)


if __name__ == '__main__':
    unittest.main()
