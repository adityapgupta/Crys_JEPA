"""Composition-balanced fixed hull minimization, independent of energy convention."""

import numpy as np
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from scipy.optimize import linprog


class FixedHull:
    """No candidate entries are ever inserted into this fixed phase bank."""
    def __init__(self, elements, references, use_phase_diagram=True):
        self.elements = list(elements)
        if not references:
            raise ValueError("Empty fixed phase bank")
        self.references = references
        self.compositions = [Composition(row["formula"]) for row in references]
        if any(not set(c.as_dict()).issubset(self.elements) for c in self.compositions):
            raise ValueError("Reference contains elements outside target chemical system")
        self.energies = np.array([row["energy_per_atom"] for row in references], dtype=float)
        if not np.isfinite(self.energies).all():
            raise ValueError("Nonfinite reference energy")
        self.matrix = np.array([[c.get_atomic_fraction(e) for c in self.compositions] for e in self.elements])
        self.diagram = None
        if use_phase_diagram and len(self.elements) <= 4:
            entries = [PDEntry(c, e*c.num_atoms, name=r["entry_id"])
                       for c, e, r in zip(self.compositions, self.energies, references)]
            try:
                self.diagram = PhaseDiagram(entries)
            except (ValueError, RuntimeError):
                pass
        self.cache = {}

    def minimum(self, formula):
        if formula in self.cache:
            return self.cache[formula]
        composition = Composition(formula)
        if not set(composition.as_dict()).issubset(self.elements):
            raise ValueError("Target composition contains unknown elements")
        target = np.array([composition.get_atomic_fraction(e) for e in self.elements])
        if self.diagram is not None:
            decomposition = self.diagram.get_decomposition(composition)
            energy = sum(entry.energy_per_atom * weight for entry, weight in decomposition.items())
            products = [{"entry_id": entry.name, "fraction": float(weight), "formula": entry.composition.reduced_formula}
                        for entry, weight in decomposition.items()]
            method = "phase_diagram"
        else:
            result = linprog(self.energies, A_eq=self.matrix, b_eq=target, bounds=(0, None), method="highs")
            if not result.success:
                raise ValueError(result.message)
            energy = float(result.fun)
            products = [{"entry_id": row["entry_id"], "fraction": float(weight), "formula": row["formula"]}
                        for row, weight in zip(self.references, result.x) if weight > 1e-8]
            method = "linear_program"
        weights = np.array([row["fraction"] for row in products])
        balance = np.array([[Composition(row["formula"]).get_atomic_fraction(e) for row in products] for e in self.elements]) @ weights
        np.testing.assert_allclose(balance, target, atol=1e-7)
        np.testing.assert_allclose(weights.sum(), 1., atol=1e-7)
        if (weights < -1e-10).any():
            raise ValueError("Negative decomposition fraction")
        value = float(energy), products, method
        self.cache[formula] = value
        return value
