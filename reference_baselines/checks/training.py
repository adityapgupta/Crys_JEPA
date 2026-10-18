"""Check real batches, graph isolation, gradients and inference invariance without training."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from reference_baselines.core.io import sha256
from reference_baselines.datasets.training import OUT, ReferenceDataset, collate
from reference_baselines.evaluation.metrics import unique_order
from reference_baselines.models.reference import MODEL_NAMES, ReferenceModel, segment_sum
from reference_baselines.paths import PACKAGE_DIR
from reference_baselines.training.train import validate


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    data = ReferenceDataset('train')
    # Include small and larger reference pools rather than only the first rows.
    positions = [0, len(data)//3, 2*len(data)//3, len(data)-1]
    items = [data[i] for i in positions]
    batch = collate(items)
    # Structure matches need not be transitive: use the repo's greedy rule, not components.
    assert unique_order([1, 2, 3], {1:{2}, 2:{1,3}, 3:{2}}).tolist() == [1, 3]
    assert np.isfinite(batch['x'].numpy()).all()
    assert (batch['batch'][batch['edge'][0]] == batch['batch'][batch['edge'][1]]).all()
    results = []
    for name in MODEL_NAMES:
        start = time.perf_counter()
        model = ReferenceModel(name).eval()
        logits, weights = model(batch)
        assert logits.shape == (len(items),) and torch.isfinite(logits).all()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['target'])
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
        with torch.no_grad():
            one, _ = model(collate([items[0]]))
            torch.testing.assert_close(one, logits[:1], atol=2e-6, rtol=2e-5)
            changed = dict(batch, target=1-batch['target'])
            unchanged, _ = model(changed)
            torch.testing.assert_close(unchanged, logits)
            # Reindex every reference node and its edges; the prediction must not depend on order.
            perm = torch.randperm(len(batch['x']))
            inverse = torch.argsort(perm)
            reordered = dict(batch, x=batch['x'][perm], pair=batch['pair'][perm],
                             batch=batch['batch'][perm], edge=inverse[batch['edge']])
            same, _ = model(reordered)
            torch.testing.assert_close(same, logits, atol=2e-6, rtol=2e-5)
            if name != 'candidate_only':
                sums = segment_sum(weights, batch['batch'], len(items))
                torch.testing.assert_close(sums, torch.ones_like(sums))
        results.append({'model': name, 'parameters': sum(p.numel() for p in model.parameters()),
                        'forward_backward_seconds': time.perf_counter()-start, 'loss': float(loss.detach()),
                        'checks': 'finite gradients, batch isolation, target isolation, reference-order invariance, normalized readout'})
    # Dataset rows and all source arrays are shared by training and inference.
    empty = dict(items[0], x=items[0]['x'][:0], pair=items[0]['pair'][:0],
                 edge=torch.empty((2, 0), dtype=torch.long), weight=items[0]['weight'][:0])
    for name in MODEL_NAMES:
        output, _ = ReferenceModel(name)(collate([empty]))
        assert torch.isfinite(output).all()
    metrics, exported_predictions = validate(ReferenceModel('candidate_only'), [batch], torch.device('cpu'), {})
    assert exported_predictions.shape == (len(items), 3)
    assert 0 <= metrics['unique_precision_top_0.05'] <= 1
    # Exercise checkpoint loading and reference export without performing an optimizer step.
    with tempfile.TemporaryDirectory(prefix='inference_check_', dir=OUT) as directory:
        checkpoint = Path(directory)/'fixture.pt'
        model = ReferenceModel('reference_attention')
        torch.save({'state_dict': model.state_dict(), 'config': {'model':'reference_attention', 'hidden':128,
                   'pool':'union','k':10,'batch_size':2}, 'epoch':0},checkpoint)
        subprocess.run([sys.executable, '-m', 'reference_baselines.training.predict',
                        str(checkpoint),'--limit','2'],check=True,capture_output=True,text=True)
        import pandas as pd
        exported = pd.read_csv(Path(directory)/'validation_learned_first2_references.csv')
        np.testing.assert_allclose(exported.groupby('candidate_row').attention_weight.sum(),1.,atol=1e-6)
        assert not exported.duplicated(['candidate_row','reference_row']).any()
    report = {'status': 'passed', 'optimizer_steps': 0, 'test_predictions': 0,
              'actual_training_rows_checked': [i['row'] for i in items],
              'actual_reference_counts': [len(i['x']) for i in items], 'empty_pool_check': 'passed',
              'checkpoint_reference_export_check':'passed (untrained temporary fixture; removed)',
              'training_validation_metric_hook':'passed on a training-only fixture',
              'models': results, 'sha256': {str(p.relative_to(PACKAGE_DIR)): sha256(p) for p in [Path(__file__).resolve(),
                  PACKAGE_DIR / 'models/reference.py', PACKAGE_DIR / 'datasets/training.py',
                  PACKAGE_DIR / 'training/train.py', PACKAGE_DIR / 'training/predict.py',
                  PACKAGE_DIR / 'evaluation/metrics.py', PACKAGE_DIR / 'paths.py',
                  PACKAGE_DIR / 'core/io.py']}}
    (OUT / 'training_smoke_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
