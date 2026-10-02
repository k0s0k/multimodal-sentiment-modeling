#!/usr/bin/env bash
# Ubuntu OpenFace: preserve source/build, compile before retrieving CEN weights.
# Recovery: --build-only [--build-dlib], then --models-only.
# This script never silently substitutes CCNF for CEN.
set -Eeuo pipefail
PREFIX="${OPENFACE_PREFIX:-$PWD/tools}"
REVISION="${OPENFACE_REVISION:-3d4b5cf8d96138be42bed229447f36cbb09a5a29}"
JOBS="${OPENFACE_BUILD_JOBS:-4}"
DOWNLOAD_TIMEOUT="${OPENFACE_DOWNLOAD_TIMEOUT:-300}"
INSTALL_DEPS=0; BUILD_ONLY=0; MODELS_ONLY=0; BUILD_DLIB=0
DLIB_VERSION=19.24.6
DLIB_DIR="${OPENFACE_DLIB_DIR:-}"
while (($#)); do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2;;
    --revision) REVISION="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --download-timeout) DOWNLOAD_TIMEOUT="$2"; shift 2;;
    --dlib-dir) DLIB_DIR="$2"; shift 2;;
    --install-deps) INSTALL_DEPS=1; shift;;
    --build-only) BUILD_ONLY=1; shift;;
    --models-only) MODELS_ONLY=1; shift;;
    --build-dlib) BUILD_DLIB=1; shift;;
    -h|--help)
      echo 'OpenFace recovery: --build-only [--build-dlib], then --models-only.'
      echo 'Reuses an existing official checkout; never silently substitutes CCNF.'
      echo 'Options: --prefix DIR --revision COMMIT --jobs N --install-deps'
      echo '         --build-only --models-only --build-dlib --dlib-dir CMAKE_CONFIG_DIR'
      echo '         --download-timeout SECONDS'; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ && "$DOWNLOAD_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'Jobs/timeout must be positive integers.' >&2; exit 2; }
((BUILD_ONLY + MODELS_ONLY < 2)) || { echo 'Choose only one of --build-only and --models-only.' >&2; exit 2; }
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"
SRC="$PREFIX/OpenFace"; BUILD="$PREFIX/openface-build"
BIN="$BUILD/bin/FeatureExtraction"
ORIGIN='https://github.com/TadasBaltrusaitis/OpenFace.git'
if ((INSTALL_DEPS)); then
  APT=(apt-get)
  if [[ $(id -u) != 0 ]]; then APT=(sudo apt-get); fi
  "${APT[@]}" update
  DEBIAN_FRONTEND=noninteractive "${APT[@]}" install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates pkg-config \
    libopencv-dev libopenblas-dev libdlib-dev libboost-filesystem-dev libboost-system-dev
fi
for tool in /usr/bin/cmake /usr/bin/g++ git curl sha256sum; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; run with --install-deps on Ubuntu." >&2; exit 3; }
done
if [[ ! -d "$SRC/.git" ]]; then
  [[ ! -e "$SRC" ]] || { echo "$SRC exists but is not a git checkout; refusing to overwrite." >&2; exit 4; }
  git init "$SRC"
  git -C "$SRC" remote add origin "$ORIGIN"
  git -C "$SRC" fetch --depth=1 origin "$REVISION"
  git -C "$SRC" checkout --detach FETCH_HEAD
fi
REMOTE="$(git -C "$SRC" remote get-url origin)"
[[ "$REMOTE" == "$ORIGIN" ]] || { echo "Unexpected origin: $REMOTE" >&2; exit 4; }
[[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]] || { echo 'Tracked source edits exist; refusing to change checkout.' >&2; exit 4; }
CURRENT="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
  TARGET="${REVISION,,}"
  if [[ "$CURRENT" != "$TARGET" ]]; then
    if ! git -C "$SRC" cat-file -e "${TARGET}^{commit}" 2>/dev/null; then git -C "$SRC" fetch --depth=1 origin "$TARGET"; fi
    git -C "$SRC" checkout --detach "$TARGET"
  fi
else
  git -C "$SRC" fetch --depth=1 origin "$REVISION"
  git -C "$SRC" checkout --detach FETCH_HEAD
fi
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
[[ -f "$SRC/CMakeLists.txt" && -d "$SRC/lib/local/LandmarkDetector/model" ]] || { echo 'Source checkout is incomplete.' >&2; exit 4; }
mkdir -p "$BUILD"
printf '%s\n' "$COMMIT" > "$BUILD/openface_commit.txt"

# Optional private static dlib installation. No system library is replaced.
DLIB_PREFIX="$PREFIX/dlib-install-$DLIB_VERSION"
DLIB_BUILD="$PREFIX/dlib-build-$DLIB_VERSION"
DLIB_SOURCE="$PREFIX/dlib-$DLIB_VERSION"
if ((BUILD_DLIB)); then
  DLIB_URL="https://codeload.github.com/davisking/dlib/tar.gz/refs/tags/v$DLIB_VERSION"
  DLIB_ARCHIVE="$PREFIX/dlib-$DLIB_VERSION.tar.gz"
  if [[ ! -f "$DLIB_ARCHIVE" ]] || ! tar -tzf "$DLIB_ARCHIVE" >/dev/null 2>&1; then
    curl --fail --location --retry 2 --connect-timeout 20 --max-time "$DOWNLOAD_TIMEOUT" \
      --output "$DLIB_ARCHIVE.part" "$DLIB_URL"
    tar -tzf "$DLIB_ARCHIVE.part" >/dev/null
    mv -- "$DLIB_ARCHIVE.part" "$DLIB_ARCHIVE"
  fi
  if [[ ! -d "$DLIB_SOURCE" ]]; then tar -xzf "$DLIB_ARCHIVE" -C "$PREFIX"; fi
  [[ -f "$DLIB_SOURCE/dlib/CMakeLists.txt" ]] || { echo "Incomplete dlib source: $DLIB_SOURCE" >&2; exit 9; }
  sha256sum "$DLIB_ARCHIVE" > "$BUILD/dlib_source_sha256.txt"
  printf 'source_url=%s\nversion=%s\ninstall_prefix=%s\nlinkage=static\ncuda=OFF\n' \
    "$DLIB_URL" "$DLIB_VERSION" "$DLIB_PREFIX" > "$BUILD/dlib_provenance.txt"
  env -u CMAKE_PREFIX_PATH -u LD_LIBRARY_PATH /usr/bin/cmake -S "$DLIB_SOURCE/dlib" -B "$DLIB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_INSTALL_PREFIX="$DLIB_PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF \
    -DDLIB_USE_CUDA=OFF -DDLIB_NO_GUI_SUPPORT=ON -DDLIB_USE_FFMPEG=OFF
  env -u LD_LIBRARY_PATH /usr/bin/cmake --build "$DLIB_BUILD" --parallel "$JOBS"
  env -u LD_LIBRARY_PATH /usr/bin/cmake --install "$DLIB_BUILD"
  DLIB_DIR="$DLIB_PREFIX/lib/cmake/dlib"
fi
if [[ -z "$DLIB_DIR" && -f "$DLIB_PREFIX/lib/cmake/dlib/dlibConfig.cmake" ]]; then DLIB_DIR="$DLIB_PREFIX/lib/cmake/dlib"; fi
DLIB_OPTIONS=()
if [[ -n "$DLIB_DIR" ]]; then
  [[ -f "$DLIB_DIR/dlibConfig.cmake" ]] || { echo "Missing dlib config: $DLIB_DIR" >&2; exit 9; }
  DLIB_OPTIONS+=("-Ddlib_DIR=$DLIB_DIR")
  printf '%s\n' "$DLIB_DIR" > "$BUILD/selected_dlib_dir.txt"
fi
if ((MODELS_ONLY == 0)); then
  if ! env -u CMAKE_PREFIX_PATH -u LD_LIBRARY_PATH /usr/bin/cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_INSTALL_PREFIX="$PREFIX/openface-install" "${DLIB_OPTIONS[@]}"; then
    echo 'Configure failed. If system dlib is older than 19.13, rerun with --build-dlib --build-only.' >&2
    exit 9
  fi
  env -u LD_LIBRARY_PATH /usr/bin/cmake --build "$BUILD" --target FeatureExtraction --parallel "$JOBS"
fi
[[ -x "$BIN" ]] || { echo 'FeatureExtraction is absent; first run --build-only.' >&2; exit 6; }
env -u LD_LIBRARY_PATH ldd "$BIN" > "$BUILD/runtime_libraries.txt"
if grep -q 'not found' "$BUILD/runtime_libraries.txt"; then cat "$BUILD/runtime_libraries.txt" >&2; exit 7; fi
sha256sum "$BIN" > "$BUILD/binary_sha256.txt"
cp -- "$BUILD/CMakeCache.txt" "$BUILD/build_config_provenance.txt"

MODEL_DIR="$SRC/lib/local/LandmarkDetector/model/patch_experts"
RUNTIME_MODEL_DIR="$BUILD/bin/model/patch_experts"
NAMES=(cen_patches_0.25_of.dat cen_patches_0.35_of.dat cen_patches_0.50_of.dat cen_patches_1.00_of.dat)
# Exact Content-Length and locally computed SHA256 of the four official Dropbox
# responses retrieved on 2026-09-23 (not an upstream-published checksum manifest).
MODEL_BYTES=(60602360 60602360 154289792 154289792)
MODEL_SHA256=(
  99d3df9888115428075de7b8f1bc86176881de640c2fdf47e2ab0cee556661bf
  1070bff51b077ee6a18ce8e1ebbc6f568e1a7a469f6911a8263413878b3df469
  3f08124067e326e83e3261a55c79493e745b01138c9c1b59713dd9b17424168e
  bed961bbfa2cc41a709d44c1d1d8a92b2537bc46f2fd6ab430c137fedd0e21b0
)
# Exact official mirrors from https://github.com/TadasBaltrusaitis/OpenFace/blob/master/download_models.sh
DROPBOX=(
  'https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1'
  'https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1'
  'https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1'
  'https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1'
)
ONEDRIVE=(
  'https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153072&authkey=AKqoZtcN0PSIZH4'
  'https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153079&authkey=ANpDR1n3ckL_0gs'
  'https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153074&authkey=AGi-e30AfRc_zvs'
  'https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153070&authkey=AD6KjtYipphwBPc'
)
valid_model() {
  local i digest
  [[ -f "$1" && $(stat -c%s "$1") -ge 1000000 ]] || return 1
  if head -c 512 "$1" | LC_ALL=C grep -aiEq '<!doctype|<html|git-lfs.github.com'; then return 1; fi
  for i in "${!NAMES[@]}"; do
    if [[ "${1##*/}" == "${NAMES[$i]}"* ]]; then
      [[ $(stat -c%s "$1") == "${MODEL_BYTES[$i]}" ]] || return 1
      digest="$(sha256sum "$1")"
      [[ "${digest%% *}" == "${MODEL_SHA256[$i]}" ]] || return 1
      return 0
    fi
  done
  return 1
}
fetch_model() {
  local i="$1" dest="$MODEL_DIR/${NAMES[$1]}" mirror part url
  if valid_model "$dest"; then echo "Using existing model: $dest"; return 0; fi
  for mirror in dropbox onedrive; do
    if [[ "$mirror" == dropbox ]]; then url="${DROPBOX[$i]}"; else url="${ONEDRIVE[$i]}"; fi
    part="$dest.$mirror.part"
    echo "Downloading ${NAMES[$i]} from official $mirror mirror"
    if curl --fail --location --retry 1 --retry-delay 3 --connect-timeout 20 \
      --max-time "$DOWNLOAD_TIMEOUT" --speed-time 30 --speed-limit 1024 \
      --output "$part" "$url" && valid_model "$part"; then
      mv -- "$part" "$dest"
      printf '%s\t%s\t%s\n' "${NAMES[$i]}" "$mirror" "$url" >> "$BUILD/model_download_sources.tsv"
      return 0
    fi
    echo "Mirror failed or returned an invalid model: $mirror ${NAMES[$i]}" >&2
  done
  return 1
}
mkdir -p "$MODEL_DIR" "$RUNTIME_MODEL_DIR"
if ((BUILD_ONLY == 0)); then
  for i in "${!NAMES[@]}"; do
    if ! fetch_model "$i"; then echo "Download pending: ${NAMES[$i]}" >&2; fi
  done
fi
MISSING=0
# Models downloaded after configure must be copied explicitly into the runtime tree.
for name in "${NAMES[@]}"; do
  if valid_model "$MODEL_DIR/$name"; then
    if ! cmp -s "$MODEL_DIR/$name" "$RUNTIME_MODEL_DIR/$name"; then
      cp -- "$MODEL_DIR/$name" "$RUNTIME_MODEL_DIR/$name.tmp"
      mv -- "$RUNTIME_MODEL_DIR/$name.tmp" "$RUNTIME_MODEL_DIR/$name"
    fi
  else
    echo "Required CEN weight missing/invalid: $MODEL_DIR/$name" >&2
    MISSING=$((MISSING+1))
  fi
done
CFG="$BUILD/bin/model/main_ceclm_general.txt"
[[ -f "$CFG" ]] || { echo "Missing runtime config: $CFG" >&2; exit 6; }
WRAPPER="$PREFIX/openface_feature_extraction"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf '# Explicit landmark model family: CEN\ncd %q\n' "$BUILD/bin"
  for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    printf '[[ -f %q && $(stat -c%%s %q) == %q ]] || { echo %q >&2; exit 8; }\n' \
      "$RUNTIME_MODEL_DIR/$name" "$RUNTIME_MODEL_DIR/$name" "${MODEL_BYTES[$i]}" "Required CEN weight absent/incomplete: $name; rerun setup --models-only"
  done
  printf 'echo %q >&2\n' "OpenFace landmark_model=CEN config=$CFG commit=$COMMIT"
  printf 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"\n'
  printf 'exec env -u LD_LIBRARY_PATH -u LD_PRELOAD %q -ml %q "$@"\n' "$BIN" "$CFG"
} > "$WRAPPER"
chmod +x "$WRAPPER"
printf 'export OPENFACE_BIN=%q\nexport OPENFACE_LANDMARK_MODEL=CEN\n' "$WRAPPER" > "$PREFIX/openface_env.sh"
printf '\nOpenFace commit: %s\nBinary wrapper: %s\nSource environment: source %q\n' "$COMMIT" "$WRAPPER" "$PREFIX/openface_env.sh"
if ((MISSING > 0)); then
  printf 'binary_built_cen_models_missing\n' > "$BUILD/setup_status.txt"
  echo "Build preserved. Rerun --models-only --revision $COMMIT to complete CEN weights." >&2
  if ((BUILD_ONLY)); then exit 0; else exit 5; fi
fi
FILES=()
for name in "${NAMES[@]}"; do FILES+=("$RUNTIME_MODEL_DIR/$name"); done
sha256sum "${FILES[@]}" > "$BUILD/model_sha256.txt"
printf 'binary_and_cen_models_present_smoke_pending\n' > "$BUILD/setup_status.txt"
echo 'Binary and CEN files checked; perform a real video smoke test before the full batch.'
