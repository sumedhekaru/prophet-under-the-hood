"""Download the public datasets into data/ (or $PUTH_DATA), verifying pinned SHA-256 checksums.

    uv run python -m common.download              # all datasets
    uv run python -m common.download m5 wt        # some of them

Files that already exist with the right checksum are kept. Archives are extracted next to themselves, then the
archive is kept so the checksum can be checked again. Sources (no login needed):
  m4       github.com/Mcompetitions/M4-methods (Dataset/): Daily, Weekly + M4-info.csv
  tourism  Monash forecasting archive, Tourism Monthly (zenodo.org/records/4656096)
  m5       Walmart M5 sales, as packaged by Nixtla (github.com/Nixtla/m5-forecasts)
  wt       Monash forecasting archive, Wikipedia web traffic 2015-07-01 to 2022-06-30 (zenodo.org/records/7371038)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(os.environ.get('PUTH_DATA', Path(__file__).parents[1] / 'data'))

_M4 = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset'

# name -> list of (url, path under data/, sha256)
FILES = {
    'm4': [
        (f'{_M4}/Train/Daily-train.csv', 'm4/Daily-train.csv',
         '78e94591c60c06309f1e544fd7b2ccba28f3616b01913dd336a1d1d98483a1ec'),
        (f'{_M4}/Test/Daily-test.csv', 'm4/Daily-test.csv',
         '633b0815626ae266d36872c4d53e95eb93ee8d43e79587fceec519eba3e6b05e'),
        (f'{_M4}/Train/Weekly-train.csv', 'm4/Weekly-train.csv',
         'd478d3f6ed673e6ed3c2cb0fbed5425f72eedd51318ebebeb0b5ecfcefc37714'),
        (f'{_M4}/Test/Weekly-test.csv', 'm4/Weekly-test.csv',
         '3b2bc4be8e636e802260dcd843ca3c5da40536a1fbe1ecb3229a97dae4f44688'),
        (f'{_M4}/M4-info.csv', 'm4/M4-info.csv',
         '26a75cc9b9f65e351b39b0bdbb7e3464ac837df0baf541ab55abdd2ba1a14c88'),
    ],
    'tourism': [
        ('https://zenodo.org/records/4656096/files/tourism_monthly_dataset.zip?download=1',
         'tourism/tourism_monthly_dataset.zip',
         '6e434ad8a0ef6acb55ebee0e1d1ffe22707ea4e62c05e091326267f3a5b4a93c'),
    ],
    'm5': [
        ('https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip', 'm5/m5.zip',
         'cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774'),
    ],
    'wt': [
        ('https://zenodo.org/records/7371038/files/web_traffic_extended_dataset_without_missing_values.zip'
         '?download=1', 'wt/web_traffic_extended_dataset_without_missing_values.zip',
         'b43dc692d451903f2d9b1cdb09e00444d6da6f58172be7f9e4c6d8ead923f75f'),
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: Path, expected: str) -> str:
    """Download url to path unless it is already there with the expected checksum. Returns the action taken."""
    if path.exists():
        if sha256(path) == expected:
            return 'ok'
        print(f'  {path.name}: checksum mismatch, downloading again')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        while block := r.read(1 << 20):
            f.write(block)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        raise RuntimeError(f'{url}: checksum {got} != pinned {expected}')
    tmp.replace(path)
    return 'downloaded'


def extract(path: Path) -> int:
    with zipfile.ZipFile(path) as z:
        missing = [n for n in z.namelist() if not (path.parent / n).exists()]
        z.extractall(path.parent, members=missing)
    return len(missing)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='*', help=f'datasets (default: all of {", ".join(FILES)})')
    a = p.parse_args()
    unknown = sorted(set(a.names) - set(FILES))
    if unknown:
        p.error(f'unknown dataset(s): {", ".join(unknown)}')
    for name in a.names or FILES:
        print(name)
        for url, rel, expected in FILES[name]:
            path = DATA / rel
            action = fetch(url, path, expected)
            extra = f', extracted {extract(path)} new file(s)' if path.suffix == '.zip' else ''
            print(f'  {rel}: {action}{extra}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
