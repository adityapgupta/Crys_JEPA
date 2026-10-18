"""Shared file-integrity helpers."""

import hashlib


def sha256(path):
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
