"""Read selected files from the pinned release ZIP using checked HTTP ranges."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import io
import json
import ssl
import struct
import urllib.request
import zipfile
import zlib

import certifi

from reference_baselines.paths import DATA_DIR


REVISION = 'ad9bdac5cfd49188174617a2c11de48baad1d329'
BASE = f'https://huggingface.co/datasets/liun-online/Crys_JEPA/resolve/{REVISION}/'
ZIP_SIZE = 7388771042


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.context = ssl.create_default_context(cafile=certifi.where())
        self.cache_start, self.cache = 0, b''
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def read(self, size=-1):
        start = self.position
        stop = self.size if size < 0 else min(self.size, start + size)
        if stop <= start:
            return b''
        if not (self.cache_start <= start and stop <= self.cache_start + len(self.cache)):
            end = min(self.size, max(stop, start + 1024 * 1024))
            req = urllib.request.Request(self.url + f'?range_start={start}&range_end={end - 1}',
                                         headers={'Range': f'bytes={start}-{end - 1}'})
            with urllib.request.urlopen(req, context=self.context, timeout=120) as response:
                expected = f'bytes {start}-{end - 1}/{self.size}'
                if response.headers.get('Content-Range') != expected:
                    raise ValueError(f'Range response mismatch: {response.headers.get("Content-Range")} != {expected}')
                self.cache = response.read()
            if len(self.cache) != end - start:
                raise ValueError('Incomplete HTTP range')
            self.cache_start = start
            self.transferred += len(self.cache)
        result = self.cache[start - self.cache_start:stop - self.cache_start]
        self.position = stop
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-base-eval', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=DATA_DIR / 'release')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    remote = RangeFile(BASE + 'exp_logs.zip', ZIP_SIZE)
    with zipfile.ZipFile(remote) as archive:
        entries = [{'name': i.filename, 'bytes': i.file_size, 'compressed_bytes': i.compress_size, 'crc32': i.CRC}
                   for i in archive.infolist()]
        (args.out_dir / 'archive_manifest.json').write_text(json.dumps(entries, indent=2) + '\n')
        chosen = [i for i in archive.infolist() if '/base/mp_20/eval/' in '/' + i.filename
                  and not i.is_dir()]
        print('Base MP-20 evaluation files:', len(chosen), 'compressed bytes:', sum(i.compress_size for i in chosen), flush=True)
        print('Examples:', [i.filename for i in chosen[:20]], flush=True)
        if args.extract_base_eval:
            groups = {}
            for info in chosen:
                groups.setdefault(str(Path(info.filename).parent), []).append(info)

            def extract_group(infos):
                source = RangeFile(BASE + 'exp_logs.zip', ZIP_SIZE)
                for info in sorted(infos, key=lambda i: i.header_offset):
                    path = args.out_dir / info.filename
                    if '..' in Path(info.filename).parts or Path(info.filename).is_absolute():
                        raise ValueError('Unsafe ZIP path')
                    if path.exists() and path.stat().st_size == info.file_size and zlib.crc32(path.read_bytes()) == info.CRC:
                        continue
                    source.seek(info.header_offset)
                    header = source.read(30)
                    if header[:4] != b'PK\x03\x04':
                        raise ValueError('Invalid ZIP local header')
                    name_size, extra_size = struct.unpack_from('<HH', header, 26)
                    source.seek(info.header_offset + 30 + name_size + extra_size)
                    compressed = source.read(info.compress_size)
                    if info.compress_type == zipfile.ZIP_DEFLATED:
                        data = zlib.decompress(compressed, -15)
                    elif info.compress_type == zipfile.ZIP_STORED:
                        data = compressed
                    else:
                        raise ValueError('Unsupported ZIP compression')
                    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                        raise ValueError('ZIP size/CRC mismatch')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                return source.transferred

            transferred = 0
            with ThreadPoolExecutor(max_workers=8) as pool:
                jobs = [pool.submit(extract_group, group) for group in groups.values()]
                for number, job in enumerate(as_completed(jobs), 1):
                    transferred += job.result()
                    if number % 20 == 0 or number == len(jobs):
                        print(f'Extracted {number}/{len(jobs)} folders; HTTP bytes {transferred:,}', flush=True)
    print('HTTP bytes transferred:', remote.transferred, flush=True)


if __name__ == '__main__':
    main()
