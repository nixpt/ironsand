#!/usr/bin/env bash
# provision-llvm19.sh — fetch + verify + install the LLVM 19 toolchain that
# `rustc_codegen_nvvm --features llvm19` links against.
#
# WHY THIS EXISTS (s472, 2026-08-22)
#   The LLVM 19 install lived only at /workspace/scratch/llvm19 — a subvolume
#   whose whole contract is "rm -rf must always be safe". It was reclaimed at
#   some point, and because nothing checked for it, the CUDA codegen capability
#   died SILENTLY: `cargo build -p rustc_codegen_nvvm --features llvm19` failed
#   with "no LLVM 19 toolchain was found" and nobody noticed for weeks. The
#   recipe was documented; the PROVISIONING was not, and absence was not
#   detectable. This script fixes both halves.
#
#   We deliberately do NOT fork or build llvm-project. We carry zero patches to
#   LLVM — we consume an official first-party RELEASE ARTIFACT. Forking source
#   would not even yield the binaries we need. What we pin is the artifact:
#   exact version + sha256, so a restore is reproducible and tamper-evident.
#   (AUR is ruled out by policy — supply-chain; and the distro repos ship
#   llvm 20/21/22, never 19.)
#
# USAGE
#   scripts/provision-llvm19.sh            # install if missing, verify, print exports
#   scripts/provision-llvm19.sh --check    # verify only; non-zero + loud if unusable
#   LLVM19_PREFIX=/some/where scripts/provision-llvm19.sh
#
# The default prefix stays under scratch on purpose (8.2 GB of regenerable
# third-party binaries do not belong in a snapshotted/backed-up subvol) — the
# point is that it is now one command to restore, and --check makes its absence
# a loud failure instead of a silent one.
set -uo pipefail

VERSION="19.1.7"
TARBALL="LLVM-${VERSION}-Linux-X64.tar.xz"
URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${VERSION}/${TARBALL}"
SHA256="4a5ec53951a584ed36f80240f6fbf8fdd46b4cf6c7ee87cc2d5018dc37caf679"
PREFIX="${LLVM19_PREFIX:-/workspace/scratch/llvm19}"
CACHE="${LLVM19_CACHE:-/workspace/scratch/llvm19-dl}"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

llvm_config="$PREFIX/bin/llvm-config"

verify() {
    [[ -x "$llvm_config" ]] || { echo "  missing: $llvm_config"; return 1; }
    local v t
    v="$("$llvm_config" --version 2>/dev/null)" || { echo "  llvm-config failed to run"; return 1; }
    [[ "$v" == 19.* ]] || { echo "  wrong major: llvm-config reports '$v', need 19.x"; return 1; }
    t="$("$llvm_config" --targets-built 2>/dev/null)"
    # NVPTX is the whole reason we need this toolchain — a build without it
    # links fine and then cannot emit PTX.
    [[ "$t" == *NVPTX* ]] || { echo "  NVPTX target missing (targets: $t)"; return 1; }
    echo "  llvm-config : $llvm_config"
    echo "  version     : $v"
    echo "  NVPTX       : present"
    return 0
}

if verify >/dev/null 2>&1; then
    echo "LLVM 19 OK"
    verify
    [[ $CHECK_ONLY -eq 1 ]] && exit 0
else
    if [[ $CHECK_ONLY -eq 1 ]]; then
        echo "LLVM 19 NOT USABLE at $PREFIX" >&2
        verify >&2
        echo "" >&2
        echo "  rustc_codegen_nvvm --features llvm19 CANNOT build without this." >&2
        echo "  Restore with: scripts/provision-llvm19.sh" >&2
        exit 1
    fi

    echo "LLVM 19 not usable at $PREFIX — provisioning ${VERSION}"
    mkdir -p "$CACHE" "$(dirname "$PREFIX")"

    if [[ ! -f "$CACHE/$TARBALL" ]] || ! echo "$SHA256  $CACHE/$TARBALL" | sha256sum -c --status 2>/dev/null; then
        echo "  fetching $URL"
        # -C - resumes a partial fetch rather than restarting 1.6 GB
        curl -fL --retry 3 --retry-delay 5 -C - -o "$CACHE/$TARBALL" "$URL" || {
            echo "  download FAILED" >&2; exit 1; }
    else
        echo "  using cached $CACHE/$TARBALL"
    fi

    if [[ "$SHA256" == "__PIN_ME__" ]]; then
        echo "  WARNING: SHA256 is unpinned in this script — recording actual:" >&2
        sha256sum "$CACHE/$TARBALL" >&2
    else
        echo "  verifying sha256"
        echo "$SHA256  $CACHE/$TARBALL" | sha256sum -c --status || {
            echo "  CHECKSUM MISMATCH — refusing to extract. Delete $CACHE/$TARBALL and retry." >&2
            exit 1; }
    fi

    echo "  extracting to $PREFIX (~8.2 GB)"
    rm -rf "$PREFIX.partial"
    mkdir -p "$PREFIX.partial"
    tar -xf "$CACHE/$TARBALL" -C "$PREFIX.partial" --strip-components=1 || {
        echo "  extract FAILED" >&2; rm -rf "$PREFIX.partial"; exit 1; }
    rm -rf "$PREFIX"
    mv "$PREFIX.partial" "$PREFIX"     # only swap in once extraction succeeded

    if ! verify; then
        echo "  provisioned tree still does not verify — see above" >&2
        exit 1
    fi
    echo "LLVM 19 provisioned"
fi

cat <<EOF

Export these before building the backend:

  export CUDA_PATH=/opt/cuda CUDA_ROOT=/opt/cuda CUDA_HOME=/opt/cuda
  export LLVM_CONFIG_19=$llvm_config
  export CARGO_TARGET_DIR=/workspace/scratch/builds/ironsand

  cargo build -p rustc_codegen_nvvm --features llvm19

Note: \`cargo build --workspace\` FAILS — it defaults to the disabled LLVM-7
path. Build specific \`-p\` targets. Full recipe:
.dejavue/references/llvm19-build-recipe.md
EOF
