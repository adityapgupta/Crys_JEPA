"""Export candidate predictions and ranked reference weights from a trained checkpoint."""

from pathlib import Path
import argparse
import json

import pandas as pd
import torch

from reference_baselines.datasets.training import OUT, ReferenceDataset, collate
from reference_baselines.models.reference import ReferenceModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--limit', type=int, default=128, help='Use 0 for the complete chosen split')
    parser.add_argument('--uniform-readout', action='store_true', help='Validation ablation: replace learned reference weights by uniform weights')
    args = parser.parse_args()
    torch.set_num_threads(6)
    saved = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    cfg = saved['config']
    model = ReferenceModel(cfg['model'], cfg['hidden'])
    model.load_state_dict(saved['state_dict']); model.eval()
    dataset = ReferenceDataset(args.split, cfg['pool'], cfg['k'])
    metadata = pd.read_csv(OUT / 'reference_rows.csv')
    records, predictions = [], []
    count = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    with torch.no_grad():
        for start in range(0, count, cfg['batch_size']):
            items = [dataset[i] for i in range(start, min(start+cfg['batch_size'], count))]
            batch = collate(items)
            logits, attention = model(batch, uniform_readout=args.uniform_readout)
            for j, item in enumerate(items):
                predictions.append({'candidate_row': item['row'], 'predicted_near_stable_probability': float(logits[j].sigmoid())})
                if cfg['model'] == 'candidate_only':
                    continue
                weights = attention[batch['batch'] == j].numpy()
                order = sorted(range(len(weights)), key=lambda x: (-weights[x], int(item['reference_ids'][x])))
                for rank, index in enumerate(order, 1):
                    ref = int(item['reference_ids'][index])
                    records.append({'candidate_row': item['row'], 'reference_row': ref,
                                    'material_id': metadata.loc[ref, 'material_id'], 'rank': rank,
                                    'attention_weight': float(weights[index]),
                                    'chemically_eligible': bool(item['pair'][index, 1])})
    suffix = 'uniform' if args.uniform_readout else 'learned'
    destination = args.checkpoint.parent / f'{args.split}_{suffix}_first{count}'
    pd.DataFrame(predictions).to_csv(str(destination)+'_predictions.csv', index=False)
    pd.DataFrame(records).to_csv(str(destination)+'_references.csv', index=False)
    Path(str(destination)+'_note.json').write_text(json.dumps({
        'candidate_count': count, 'reference_weights': 'contextual attention, not physical decomposition probabilities',
        'uniform_readout': args.uniform_readout, 'checkpoint_epoch': saved['epoch']}, indent=2)+'\n')
    print(destination)


if __name__ == '__main__':
    main()
