"""Random-split linear decodability probes on frozen known-crystal embeddings."""
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, recall_score, r2_score
from sklearn.exceptions import ConvergenceWarning
from .build_graph import load_embeddings
from .common import arguments, provenance, save_json


def fit_classifier(X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = LogisticRegression(max_iter=2000).fit(X, y)
    return model, [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]


def main():
    args = arguments(__doc__, data=True).parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    emb, meta = load_embeddings(args.data)
    sg = meta['spacegroup.number'].to_numpy(dtype=int)
    energy = meta.formation_energy_per_atom.to_numpy(dtype=float)
    if not np.isfinite(energy).all() or (meta.chemical_system == '').any():
        raise ValueError('Finite energy and nonempty chemical-system labels required')
    has_o = meta.chemical_system.apply(lambda s: 'O' in s.split('-')).to_numpy()
    tr, te = train_test_split(np.arange(len(emb)), test_size=.25, random_state=args.seed)
    scaler = StandardScaler().fit(emb[tr])
    train, test = scaler.transform(emb[tr]), scaler.transform(emb[te])
    model, sg_warnings = fit_classifier(train, sg[tr])
    probabilities = model.predict_proba(test)
    prediction = model.classes_[probabilities.argmax(axis=1)]
    top5 = model.classes_[np.argsort(probabilities, axis=1)[:, -5:]]
    labels, counts = np.unique(sg[tr], return_counts=True)
    common = labels[np.argsort(counts)[::-1]]
    reg = Ridge(alpha=1, solver='svd').fit(train.astype(np.float64), energy[tr])
    oxygen, oxygen_warnings = fit_classifier(train, has_o[tr])
    oxygen_majority = bool(has_o[tr].mean() >= .5)
    result = {
        'provenance': provenance([args.data/'embeddings.npy', args.data/'metadata.csv'], seed=args.seed,
                                split='random rows, 75/25; not chemistry held out', max_iter=2000),
        'train_rows': len(tr), 'test_rows': len(te),
        'space_group': {'top1_accuracy': float(accuracy_score(sg[te], prediction)),
                        'balanced_accuracy': float(recall_score(sg[te], prediction, labels=np.unique(sg[te]), average='macro', zero_division=0)),
                        'top5_accuracy': float(np.mean(np.any(top5 == sg[te, None], axis=1))),
                        'majority_accuracy': float(np.mean(sg[te] == common[0])),
                        'top5_frequency_baseline': float(np.mean(np.isin(sg[te], common[:5]))),
                        'test_labels_unseen_in_probe_training': int(np.sum(~np.isin(sg[te], labels))),
                        'convergence_warnings': sg_warnings},
        'energy': {'solver': 'float64 SVD, alpha=1', 'r2': float(r2_score(energy[te], reg.predict(test.astype(np.float64)))),
                   'train_mean_baseline_r2': float(r2_score(energy[te], np.full(len(te), energy[tr].mean())))},
        'oxygen': {'accuracy': float(accuracy_score(has_o[te], oxygen.predict(test))),
                   'majority_accuracy': float(np.mean(has_o[te] == oxygen_majority)),
                   'convergence_warnings': oxygen_warnings},
        'limits': 'Frozen-encoder random-split decodability; not an energy-controlled or chemistry-held-out comparison.',
    }
    save_json(args.out/'probe_results.json', result)
    np.savez(args.out/'probe_split.npz', train=tr, test=te)
    print(result)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        main()
