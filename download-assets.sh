#!/bin/bash
# Downloads the pretrained checkpoints, ANI datasets, and pre-computed
# result caches (data/, model/, results/) from the GitHub Release "assets-v1"
# of this repository. See README.md, section 3.
set -euo pipefail

ASSETS_URL="https://github.com/nusanyoma/Quantum-MLIP/releases/download/assets-v1/code_code_assets-v1.tar.gz"
EXPECTED_SHA256="ecb46096845635401101599a46734309e0b776ebdf6914b890fc59346e8b031d"

cd "$(dirname "$0")"

if [ -d data ] && [ -d model ] && [ -d results ]; then
    echo "data/, model/, and results/ already exist — nothing to do."
    echo "Delete them first if you want to force a re-download."
    exit 0
fi

echo "Downloading assets from $ASSETS_URL ..."
curl -L --fail -o assets.tar.gz "$ASSETS_URL"

echo "Verifying checksum ..."
echo "${EXPECTED_SHA256}  assets.tar.gz" | sha256sum -c -

echo "Extracting ..."
tar -xzf assets.tar.gz
rm assets.tar.gz

echo "Done. data/, model/, and results/ are now populated."
