#!/usr/bin/env bash
# Compila nauty localmente en tools/nauty/ (sin sudo ni instalación en el sistema).
# Uso: bash tools/install_nauty.sh      Después: tools/nauty/geng -c 5
set -euo pipefail
VERSION="2_8_9"
DIR="$(cd "$(dirname "$0")" && pwd)"
URL="https://users.cecs.anu.edu.au/~bdm/nauty/nauty${VERSION}.tar.gz"   # mirror: https://pallini.di.uniroma1.it/
cd "$DIR"
if [ -x nauty/geng ]; then echo "nauty ya está compilado en $DIR/nauty"; exit 0; fi
if command -v curl >/dev/null; then curl -fL -o "nauty${VERSION}.tar.gz" "$URL"; else wget -q -O "nauty${VERSION}.tar.gz" "$URL"; fi
rm -rf nauty "nauty${VERSION}"
tar xzf "nauty${VERSION}.tar.gz"
mv "nauty${VERSION}" nauty
rm "nauty${VERSION}.tar.gz"
cd nauty
./configure > /dev/null
make -j"$(nproc)" geng showg countg > /dev/null
echo "Listo: $DIR/nauty/geng"
