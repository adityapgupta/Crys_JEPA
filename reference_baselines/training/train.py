"""Train one reference model. This command never evaluates the locked test split."""

import argparse
import json
import random
import time

from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import torch

from reference_baselines.datasets.training import OUT, ReferenceDataset, collate
from reference_baselines.evaluation.metrics import unique_order, load_adjacency
from reference_baselines.models.reference import MODEL_NAMES, ReferenceModel


def validate(model, loader, device, adjacency):
    model.eval()
    probabilities, targets, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits, _ = model(batch)
            probabilities.extend(logits.sigmoid().cpu().tolist())
            targets.extend(batch['target'].cpu().tolist())
            rows.extend(batch['rows'].cpu().tolist())
    probabilities, targets, rows = map(np.asarray, [probabilities, targets, rows])
    order = np.lexsort((rows, -probabilities))
    unique = unique_order(rows[order], adjacency)
    target_by_row = np.full(int(rows.max())+1, np.nan)
    target_by_row[rows.astype(int)] = targets
    metrics = {'count': len(rows)}
    for fraction in [.01, .05, .1, .2]:
        budget = max(1, int(len(rows) * fraction))
        metrics[f'precision_top_{fraction:g}'] = float(targets[order[:budget]].mean())
        metrics[f'unique_precision_top_{fraction:g}'] = float(target_by_row[unique[:budget]].sum()/budget)
    metrics['brier_score'] = float(np.square(probabilities - targets).mean())
    return metrics, np.column_stack([rows, probabilities, targets])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    parser.add_argument('--pool', choices=['union', 'chemical', 'knn', 'intersection_fallback'], default='union')
    parser.add_argument('--k', type=int, choices=[5, 10, 20, 50], default=10)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    device = torch.device(args.device)
    adjacency = load_adjacency()
    train = ReferenceDataset('train', args.pool, args.k)
    validation = ReferenceDataset('validation', args.pool, args.k)
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate)
    model = ReferenceModel(args.model, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # Unweighted BCE preserves the observed class prior and a proper probability target.
    loss_fn = nn.BCEWithLogitsLoss()
    directory = OUT / 'training_runs' / f'{args.model}_{args.pool}_k{args.k}_lr{args.learning_rate:g}_seed{args.seed}_h{args.hidden}'
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'config.json').write_text(json.dumps(vars(args), indent=2)+'\n')
    best, stale, history = -np.inf, 0, []
    for epoch in range(args.epochs):
        start = time.perf_counter()
        model.train()
        total_loss, count = 0., 0
        for batch in training_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch)
            loss = loss_fn(logits, batch['target'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(logits); count += len(logits)
        metrics, predictions = validate(model, validation_loader, device, adjacency)
        record = {'epoch': epoch + 1, 'training_bce': total_loss / count,
                  'seconds': time.perf_counter() - start, **metrics}
        history.append(record)
        (directory / 'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        score = metrics['unique_precision_top_0.05']
        if score > best:
            best, stale = score, 0
            torch.save({'state_dict': model.state_dict(), 'config': vars(args), 'epoch': epoch+1,
                        'validation_metrics': metrics}, directory / 'best.pt')
            np.savetxt(directory / 'best_validation_predictions.csv', predictions, delimiter=',',
                       header='candidate_row,predicted_near_stable_probability,proxy_target', comments='',
                       fmt=['%d', '%.9g', '%d'])
        else:
            stale += 1
            if stale >= args.patience:
                break


if __name__ == '__main__':
    main()
