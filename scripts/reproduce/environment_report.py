"""Report this interpreter's packages without importing models or downloading data.

Run from the repository root: python scripts/reproduce/environment_report.py
This describes the current machine, not the original paper environment.
"""

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    packages = {}
    names = [
        line.strip() for line in (root / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for name in names + ["uk-core-news-sm", "matplotlib", "jupyterlab"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    assets = [
        "datasets_pre_defined/sum_final.jsonlines",
        "datasets_pre_defined/unique_lemmas_homonyms.txt",
        "models/20180506.uk.mova-institute.udpipe",
        "models/translators/opus-mt-zle-en-ct2",
        "models/translators/opus-mt-en-zle-ct2",
    ]
    print(json.dumps({
        "notice": "Current machine only; null means distribution not installed.",
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "asset_exists": {name: (root / name).exists() for name in assets},
    }, indent=2))


if __name__ == "__main__":
    main()
