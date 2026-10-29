# Crys-JEPA screening experiments: team summary

This page summarizes the major completed experiment families in this repository.
The common task is to rank generated crystal candidates so that a limited
screening budget contains more candidates that appear near-stable under the
project's evaluation pipeline.

Unless stated otherwise, a "pass" means a corrected **MatterSim proxy** energy no
more than 0.1 eV/atom above the fixed MP2023 hull. It is not a DFT, synthesis,
phonon, or finite-temperature stability result.

## Experiments at a glance

| Experiment | What we tested | Main result | What it supports | Main limitation |
|---|---|---|---|---|
| Data, label, and reference audit | Whether embeddings, metadata, splits, hull labels, duplicates, and known references were internally consistent | 27,136 reference and 324,622 candidate embeddings aligned; all candidates received usable proxy labels; chemical systems do not cross the supervised splits | The later comparisons start from a checked and reproducible dataset | Labels are MatterSim against a fixed hull; encoder pretraining chemistry is not held out |
| Geometry signal in embedding neighbours | Whether nearby Crys-JEPA references are geometrically closer than energy-matched controls | JEPA neighbours were closer under an independent SOAP geometry descriptor | Embedding proximity contains a geometry association beyond the particular energy-matched control | This does not establish screening value or full independence from chemistry and energy |
| Non-parametric reference and graph search | Euclidean/cosine neighbours, intersections, angle gates, strict reference banks, diffusion, conductance, and random controls | On validation at 5%: original 727 passes (38.9%); best intersection 798 (42.7%); 1-degree angle gate 857 (45.8%); diffusion methods about 39.9–40.3% | A simple close-angle gate was the strongest repeatable rule in the exploratory grid | Many settings were compared on an open validation set; graph diffusion did not establish a gain |
| Fixed held-out angle-gate comparison | Whether the frozen 1-degree gate carried its validation gain to the then-unopened test set | Original: 644/1,849 (34.8%); gate: 782/1,849 (42.3%); +138 passes and +7.46 percentage points; paired system-bootstrap interval +3.30 to +11.25 points | The fixed gate improved MatterSim screening yield on that candidate set | The test set is now open; no independent DFT was performed |
| Fresh MatterSim re-relaxation and validity checks | Whether the gate's added candidates survive a fresh, equal calculation and basic validity pipeline | 3,243 union calculations attempted, 3,239 converged; all 1,394 added candidates passed geometry/composition checks; final unique passes were 808 for the gate versus 662 for original | The comparative proxy-yield gain survives a fresh re-relaxation and duplicate/validity checks | MatterSim and rule-based validity still do not prove synthesizability or DFT stability |
| Budget, averaging, and reference-pool ablations | Whether the gain came from the 5% budget, distance weighting, or simply adding neighbours | The gate remained better at 2%, 5%, and 7%; weighting original references gave 692 versus 644 passes at test 5%, while the 1-degree reference pool gave 785 plain/793 weighted; adding five cosine neighbours was inconsistent | Weighting alone does not explain the angular gain, and adding neighbours is not automatically useful | These are exploratory comparisons on already inspected proxy labels |
| Chemistry-matched and same-formula controls | Whether the global gain means better ranking of structures within the same chemistry | With identical chemical-system quotas, gate and original differences were near zero; on 4,622 opposite-label same-formula pairs, original and gate accuracy were 50.8% and 50.7% | Much of the global gain is associated with selecting a different chemistry mix | This is a controlled diagnostic, not a causal decomposition; a structural signal could still exist |
| Embedding communities and property probes | Whether frozen embeddings organize by energy, composition, and symmetry | Community space-group AMI was about 0.148 versus near-zero shuffles; chemical-system AMI about 0.067; linear probes reached energy R² 0.968, oxygen accuracy 95.1%, and space-group top-1 accuracy 34.0% versus a 14.3% majority baseline | Energy/composition information is strong and symmetry information is weaker but non-random | Random row probes and unconditional community tests do not isolate symmetry from energy/composition |
| Space-group concentration | Whether individual space groups concentrate in a small number of embedding communities | 61 of 73 eligible groups had BH-adjusted q ≤ 0.05 under the unconditional label-permutation null; concentration varied strongly by group | Some symmetry labels are non-randomly concentrated in this graph partition | Space group is not a structural prototype; the reported null is not conditioned on energy or composition |
| Historical candidate-stability GNN pilot | Whether a cosine reference graph improved a learned candidate-stability model over a no-graph control | Mean 5% validation yields were about 60.2% no-graph, 60.0% GCN, and 59.7% MPNN; no graph advantage appeared | The pipeline ran, but neighbour message passing did not show value over learned readout | This pilot trained/selected with candidate labels and its reported validation overlaps checkpoint selection; it is **not valid evidence** for the newer reference-utility question |
| Reference-utility GNN with explicit edge attributes | Whether embedding similarity, same-space-group, and electronegativity distance improve the fixed physical reference graph | Mean reserved 5% yield: scalar 44.4%, real attributes 44.4%, shuffled attributes 44.2%, no-message readout 44.2%; all nine real-versus-scalar intervals include zero | These attributes add no demonstrated value on this topology; graph propagation also adds no measurable value | Internal engineering holdout, MatterSim labels, fixed topology, and coarse feature definitions |

## The clearest conclusions so far

1. **The embeddings carry a strong energy/composition signal and a weaker,
   non-random symmetry signal.** The present experiments do not yet show that the
   symmetry signal is independent of chemistry and energy in every setting.
2. **The 1-degree angle gate is the strongest demonstrated screening rule in the
   current pipeline.** Its fixed test and fresh MatterSim checks improved global
   yield, but chemistry-matched and same-formula controls show that this is not
   evidence of better polymorph ranking within a fixed composition.
3. **Graphs have not yet shown an added screening advantage.** Diffusion and
   conductance baselines did not beat the simple gate; the historical GNN pilot did
   not beat its no-graph control; and the completed edge-attribute GNN experiment
   found no reliable value from its real edge attributes or message passing.
4. **All stability conclusions remain proxy conclusions.** The work supports
   comparative screening under MatterSim and a fixed reference hull, not a claim
   that selected materials are synthesizable or DFT-stable.

## Reading guide

- [Latest edge-attribute experiment in simple English](edge_attribute_experiment_plain_english.md)
- [Full edge-attribute protocol](edge_attribute_reference_gnn.md)
- [Fixed test comparison](heldout_experiment.md)
- [Fresh MatterSim validation](mattersim_validation.md)
- [Mean and reference-pool ablations](mean_ablation.md)
- [Chemistry-matched and same-formula controls](pre_gnn_report.md)
- [Embedding community experiment](embedding_structure_experiment.md)
- [Space-group concentration experiment](structure_capture_experiment.md)
- [Non-parametric experiments](nonparametric_experiments.md)

