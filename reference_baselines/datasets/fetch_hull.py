"""Download and verify the pinned MP-2023 evaluation reference database."""

import gzip
import hashlib
import json
import shutil
import ssl
import urllib.request

import certifi

from reference_baselines.datasets.fetch_release import BASE, REVISION
from reference_baselines.paths import DATA_DIR


def main():
    out = DATA_DIR / 'hull'
    out.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    listing = f'https://huggingface.co/api/datasets/liun-online/Crys_JEPA/tree/{REVISION}/ref_dataset/mp'
    with urllib.request.urlopen(listing, context=context, timeout=90) as response:
        info = next(x for x in json.load(response) if x['path'].endswith('/mp.gz'))
    expected = info['lfs']['oid']
    target = out / 'mp.gz'
    url = BASE + 'ref_dataset/mp/mp.gz'
    if not target.exists():
        with urllib.request.urlopen(url, context=context, timeout=120) as response, target.with_suffix('.part').open('wb') as f:
            shutil.copyfileobj(response, f, length=8 * 1024 * 1024)
        target.with_suffix('.part').rename(target)
    h = hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError('Reference hull checksum mismatch')
    with gzip.open(target, 'rb') as source, (out / 'mp.lmdb').open('wb') as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    (out / 'source.json').write_text(json.dumps({'url': url, 'revision': REVISION, 'sha256': expected,
                                               'bytes': target.stat().st_size}, indent=2) + '\n')
    print('Verified and decompressed MP-2023 hull database', flush=True)


if __name__ == '__main__':
    main()
