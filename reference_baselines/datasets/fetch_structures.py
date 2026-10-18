"""Download the paper's pinned MP-20 reference structure file."""

from pathlib import Path
import argparse
import hashlib
import json
import ssl
import urllib.request

import certifi

from reference_baselines.paths import DATA_DIR


REVISION = 'ad9bdac5cfd49188174617a2c11de48baad1d329'
EXPECTED_SHA256 = '237437a664c436e1a092223cce5c00e8795cb3eb9d5ea958346a00adaa6cfa9f'
URL = f'https://huggingface.co/datasets/liun-online/Crys_JEPA/resolve/{REVISION}/data/base/mp_20/train.csv.gz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=DATA_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / 'mp_20_train.csv.gz'
    if path.exists():
        data = path.read_bytes()
    else:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL, context=context, timeout=90) as response:
            data = response.read()
    checksum = hashlib.sha256(data).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise ValueError('Downloaded structure file has an unexpected checksum.')
    path.write_bytes(data)
    (args.out_dir / 'source.json').write_text(json.dumps({
        'url': URL, 'dataset_revision': REVISION, 'bytes': len(data), 'sha256': checksum,
    }, indent=2) + '\n')
    print(f'Verified structure source: {path}')


if __name__ == '__main__':
    main()
