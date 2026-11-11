#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# extract_snvc50m.sh — Selective streaming extraction of SN-VC-50M volumes
#
# Concatenates complete SenseNova-Vision-Corpus-50M.tar.gz000~N volumes into
# a decompressor pipe (pigz -dc or gzip -dc) and extracts only the requested
# subdirectories, compatible with both GNU tar (Linux) and BSD tar (macOS).
#
# Usage:
#   bash scripts/extract_snvc50m.sh --dataset-root /path/to/volumes \
#       --extract-root /path/to/extracted --batch seg
#   bash scripts/extract_snvc50m.sh --dataset-root /path/to/volumes \
#       --extract-root /path/to/extracted --dirs Cityscapes Hypersim scannetpp coconut
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/raw/SenseNova-Vision-Corpus-50M}"
EXTRACT_ROOT="${EXTRACT_ROOT:-data/raw/snvc50m_extracted}"
TAR_PREFIX="SenseNova-Vision-Corpus-50M.tar.gz"
INNER_PREFIX="SenseNova-Vision-Corpus-50M"

DEPTH_NORMAL_DIRS=("coco2017" "scannetpp")
SEG_DIRS=("Cityscapes" "coconut" "Hypersim" "hypersim" "IDD" "VIPSeg")
RECON_DIRS=("scannetpp" "DL3DV" "wild_rgbd")

DRY_RUN=false
BATCH=""
LIST_DIRS=false
CUSTOM_DIRS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --extract-root) EXTRACT_ROOT="$2"; shift 2 ;;
        --inner-prefix) INNER_PREFIX="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --list)         LIST_DIRS=true; shift ;;
        --batch)        BATCH="$2"; shift 2 ;;
        --all)          BATCH="all"; shift ;;
        --smoke)        BATCH="smoke"; shift ;;
        --dirs)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CUSTOM_DIRS+=("$1")
                shift
            done
            ;;
        --help|-h)
            echo "Usage: $0 [--dataset-root DIR] [--extract-root DIR] [--batch smoke|depth|seg|recon|all] [--dirs D1 D2 ...]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Discover valid volume files (exclude .aria2, .tmp, .part, .idx, .json)
VOLUMES=()
for f in "$DATASET_ROOT"/${TAR_PREFIX}[0-9]*; do
    [[ -f "$f" ]] || continue
    case "$f" in
        *.aria2|*.tmp|*.part|*.gzidx|*.json) continue ;;
    esac
    if [[ -f "${f}.aria2" ]]; then
        log "Skipping incomplete volume (has .aria2): $f"
        continue
    fi
    VOLUMES+=("$f")
done

if (( ${#VOLUMES[@]} == 0 )); then
    echo "ERROR: No valid ${TAR_PREFIX}[0-9]* volumes found in $DATASET_ROOT" >&2
    exit 1
fi

if command -v pigz >/dev/null 2>&1; then
    DECOMPRESSOR="pigz -dc"
else
    DECOMPRESSOR="gzip -dc"
fi

# Detect whether tar supports --wildcards (GNU tar requires it; BSD/macOS tar rejects it).
# A string test, not `| grep -q`: grep's early exit SIGPIPEs tar, which pipefail reports as failure.
USE_WILDCARDS=false
if [[ "$(tar --help 2>&1)" == *--wildcards* ]]; then
    USE_WILDCARDS=true
fi

if $LIST_DIRS; then
    log "Scanning first volume (${VOLUMES[0]}) for top-level directories..."
    $DECOMPRESSOR "${VOLUMES[0]}" 2>/dev/null | tar -tf - 2>/dev/null | \
        sed "s|^${INNER_PREFIX}/||" | cut -d'/' -f1 | sort -u | head -30 || true
    exit 0
fi

TARGET_DIRS=()
if (( ${#CUSTOM_DIRS[@]} > 0 )); then
    TARGET_DIRS=("${CUSTOM_DIRS[@]}")
else
    case "$BATCH" in
        smoke) TARGET_DIRS=("coco2017" "scannetpp") ;;
        depth) TARGET_DIRS=("${DEPTH_NORMAL_DIRS[@]}") ;;
        seg)   TARGET_DIRS=("${SEG_DIRS[@]}") ;;
        recon) TARGET_DIRS=("${RECON_DIRS[@]}") ;;
        all)   TARGET_DIRS=("${DEPTH_NORMAL_DIRS[@]}" "${SEG_DIRS[@]}" "DL3DV" "wild_rgbd") ;;
        *)
            echo "ERROR: Specify --batch smoke|depth|seg|recon|all or --dirs D1 D2 ..." >&2
            exit 1
            ;;
    esac
fi

PATTERNS=()
for d in "${TARGET_DIRS[@]}"; do
    if [[ -n "$INNER_PREFIX" ]]; then
        PATTERNS+=("${INNER_PREFIX%/}/${d}/*")
    else
        PATTERNS+=("${d}/*")
    fi
done

LAST_IDX=$(( ${#VOLUMES[@]} - 1 ))
log "Volumes (${#VOLUMES[@]}): ${VOLUMES[0]} ... ${VOLUMES[$LAST_IDX]}"
log "Decompressor: $DECOMPRESSOR"
log "Patterns: ${PATTERNS[*]}"
log "Extract root: $EXTRACT_ROOT"

if $DRY_RUN; then
    log "[DRY RUN] Exiting without extracting."
    exit 0
fi

mkdir -p "$EXTRACT_ROOT"
if $USE_WILDCARDS; then
    cat "${VOLUMES[@]}" | $DECOMPRESSOR | tar -xf - -C "$EXTRACT_ROOT" --wildcards "${PATTERNS[@]}"
else
    cat "${VOLUMES[@]}" | $DECOMPRESSOR | tar -xf - -C "$EXTRACT_ROOT" "${PATTERNS[@]}"
fi
log "Selective extraction complete: $EXTRACT_ROOT"
