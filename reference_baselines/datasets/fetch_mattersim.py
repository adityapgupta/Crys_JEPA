"""Fetch the pinned official 1M checkpoint used by the fresh inference audit."""

import json
import ssl
import urllib.request

import certifi

from reference_baselines.core.io import sha256
from reference_baselines.paths import DATA_DIR

REVISION = "5cc5ea92534d7141939225a84fd098ee56083458"
CHECKSUM = "28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be"
URL = f"https://raw.githubusercontent.com/microsoft/mattersim/{REVISION}/pretrained_models/mattersim-v1.0.0-1M.pth"


def main():
    target_dir = DATA_DIR / "mattersim"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "mattersim-v1.0.0-1M.pth"
    if not target.exists():
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL, context=context, timeout=120) as response:
            temporary = target.with_suffix(".part")
            temporary.write_bytes(response.read())
        if sha256(temporary) != CHECKSUM:
            raise ValueError("Downloaded checkpoint checksum mismatch")
        temporary.rename(target)
    if sha256(target) != CHECKSUM:
        raise ValueError("Existing checkpoint checksum mismatch")
    (target_dir / "source.json").write_text(
        json.dumps(
            {
                "url": URL,
                "revision": REVISION,
                "sha256": CHECKSUM,
                "bytes": target.stat().st_size,
            },
            indent=2,
        )
        + "\n"
    )
    print("Verified official MatterSim 1M checkpoint:", target)


if __name__ == "__main__":
    main()
