#!/usr/bin/env bash
set -e

# Detect architecture
ARCH=$(uname -m)

# Normalize architecture names
normalize_arch() {
  case "$1" in
    x86_64|amd64)
      echo "x86_64"
      ;;
    aarch64|arm64)
      echo "aarch64"
      ;;
    i386|i686)
      echo "i686"
      ;;
    riscv64)
      echo "riscv64"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

detect_manylinux_tag() {
  if ! command -v ldd >/dev/null 2>&1; then
    echo "manylinux_2_17"  # Default fallback (works on most systems)
    return
  fi

  # Check if this is a glibc-based system
  if ! ldd --version 2>/dev/null | grep -iq "glibc\|gnu libc"; then
    # Not glibc, might be musl or other
    return 1
  fi

  # Extract glibc version number
  GLIBC_VERSION=$(ldd --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
  GLIBC_MAJOR=$(echo "$GLIBC_VERSION" | cut -d. -f1)
  GLIBC_MINOR=$(echo "$GLIBC_VERSION" | cut -d. -f2)

  # Default fallback
  MANYLINUX="manylinux_2_17"

  if (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 40 )); then
    MANYLINUX="manylinux_2_40"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 39 )); then
    MANYLINUX="manylinux_2_39"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 38 )); then
    MANYLINUX="manylinux_2_38"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 37 )); then
    MANYLINUX="manylinux_2_37"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 36 )); then
    MANYLINUX="manylinux_2_36"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 35 )); then
    MANYLINUX="manylinux_2_35"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 34 )); then
    MANYLINUX="manylinux_2_34"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 33 )); then
    MANYLINUX="manylinux_2_33"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 32 )); then
    MANYLINUX="manylinux_2_32"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 31 )); then
    MANYLINUX="manylinux_2_31"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 28 )); then
    MANYLINUX="manylinux_2_28"
  elif (( GLIBC_MAJOR == 2 && GLIBC_MINOR >= 17 )); then
    MANYLINUX="manylinux_2_17"
  else
    MANYLINUX="manylinux_2_17"
  fi

  echo "$MANYLINUX"
}

detect_linux_variant() {
  local arch=$1
  
  # Check for musl (Alpine, etc.)
  if ldd --version 2>&1 | grep -iq "musl"; then
    echo "${arch}-unknown-linux-musl"
    return
  fi
  
  # Check for Android
  if [[ -n "$ANDROID_ROOT" ]] || [[ -d "/system/bin" && -f "/system/build.prop" ]]; then
    echo "${arch}-linux-android"
    return
  fi
  
  # Try to detect manylinux (glibc-based)
  if MANYLINUX_TAG=$(detect_manylinux_tag); then
    echo "${arch}-${MANYLINUX_TAG}"
    return
  fi
  
  # Fallback to unknown-linux-gnu
  echo "${arch}-unknown-linux-gnu"
}

# Normalize architecture
ARCH=$(normalize_arch "$ARCH")

# Detect Linux platform variant
PY_PLATFORM=$(detect_linux_variant "$ARCH")

echo "Detected platform: $PY_PLATFORM"
uv sync --python-platform "$PY_PLATFORM" "$@"
