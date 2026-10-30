"""Fetch the pinned original encoder; inspect, rather than invent, its scaler."""

import json
import ssl
import urllib.request
import zipfile
import zlib

import certifi

from reference_baselines.core.io import sha256
from reference_baselines.datasets.fetch_release import (
    BASE,
    REVISION,
    ZIP_SIZE,
    RangeFile,
)
from reference_baselines.paths import DATA_DIR


def main():
    out = DATA_DIR / "jepa_audit"
    out.mkdir(exist_ok=True)
    remote = RangeFile(BASE + "exp_logs.zip", ZIP_SIZE)
    records = []
    with zipfile.ZipFile(remote) as archive:
        for name in [
            "exp_logs/jepa/backup-config.yaml",
            "exp_logs/jepa/saved_model/model_1972.pt",
        ]:
            info = archive.getinfo(name)
            path = out / name.rsplit("/", 1)[-1]
            if (
                not path.exists()
                or path.stat().st_size != info.file_size
                or zlib.crc32(path.read_bytes()) != info.CRC
            ):
                print("Downloading", name, info.file_size, "bytes", flush=True)
                data = archive.read(info)
                if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                    raise ValueError("ZIP verification failed")
                path.write_bytes(data)
            records.append(
                {
                    "member": name,
                    "bytes": info.file_size,
                    "crc32": info.CRC,
                    "sha256": sha256(path),
                }
            )
    # The scaler is not listed in the public data/jepa directory. Record the
    # actual listing so missing preprocessing cannot be silently substituted.
    url = f"https://huggingface.co/api/datasets/liun-online/Crys_JEPA/tree/{REVISION}/data/jepa"
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        listing = json.load(response)
    (out / "source.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "archive": BASE + "exp_logs.zip",
                "files": records,
                "public_jepa_data_listing": listing,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Verified encoder assets; public JEPA data paths:", [x["path"] for x in listing]
    )


if __name__ == "__main__":
    main()
