#!/bin/bash
# Downloads the pretrained checkpoints, ANI datasets, and pre-computed
# result caches needed to run code/code/ (see README.md, "Prerequisites").
# These are not tracked in git (see .gitignore) because of their size;
# instead they're hosted as a GitHub Release asset and fetched by this
# script, following the same pattern as e.g. TorchANI's
# download-dev-data.sh (which fetches from Hugging Face Hub instead).
#
# TODO (before this URL will work): create a GitHub Release tagged
# "assets-v1" on https://github.com/nusanyoma/Quantum-MLIP and attach
# release_assets/code_code_assets-v1.tar.gz to it as a release asset (see
# README.md, "Maintainer note"). If you use a different tag name, update
# ASSETS_URL below to match.
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
