#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# extract_snvc50m.sh — Selective extraction of SN-VC-50M data assets
#
# The dataset is distributed as 406 split tar.gz volumes.  This script
# selectively extracts only the directories needed for depth / normal /
# segmentation / reconstruction training, saving disk space vs full extract.
#
# Usage:
#   bash scripts/extract_snvc50m.sh [--dry-run] [--batch NAME] [--all]
#
# Batches:
#   smoke   — COCO + ScanNet++ only (smallest, for pipeline smoke test)
#   depth   — All depth/normal source datasets
#   seg     — All segmentation source datasets
#   recon   — Multi-view reconstruction datasets
#   all     — Everything needed for training (depth + seg + recon)
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────
DATASET_ROOT="${DATASET_ROOT:-/media/linxu/thinkHub/datasets/SenseNova-Vision-Corpus-50M}"
EXTRACT_ROOT="${EXTRACT_ROOT:-/media/linxu/thinkHub/datasets/snvc50m_extracted}"
TAR_PREFIX="SenseNova-Vision-Corpus-50M.tar.gz"
INNER_PREFIX="SenseNova-Vision-Corpus-50M/"

# ── Directory groups ──────────────────────────────────────────────────────
# Depth/Normal source datasets (inside the tarball)
DEPTH_NORMAL_DIRS=(
    "coco2017"
    "scannetpp"
    # Larger datasets — uncomment when disk space allows
    # "object365"
    # "sa_1b"
    # "taskonomy"
)

# Segmentation source datasets
SEG_DIRS=(
    "Cityscapes"
    "coconut"
    "Hypersim"
    "hypersim"       # Case may vary in tarball
    "IDD"
    "VIPSeg"
    # Larger datasets — uncomment when disk space allows
    # "LVIS"
    # "MapillaryVistas"
    # "NuScenes"
    # "51WORLD"
    # "entityv2"
    # "Trashcan"
    # "PIDRay"
    # "ZeroWaste"
    # "TAS500"
    # "KITTI"
    # "StreetHazards"
    # "LOVEDA"
    # "UDD5"
    # "UDD6"
    # "TTPLA"
    # "flickr30k"
    # "PSG"
    # "grandf"
    # "refcocog"
)

# Reconstruction source datasets
RECON_DIRS=(
    "scannetpp"      # Already included in depth/normal
    "DL3DV"
    "wild_rgbd"
    # "scannetv2"    # Uncomment if needed
)

# ── Helpers ───────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

build_wildcards() {
    local -n dirs_ref=$1
    local wildcards=()
    for d in "${dirs_ref[@]}"; do
        wildcards+=("${INNER_PREFIX}${d}/*")
    done
    echo "${wildcards[@]}"
}

check_disk_space() {
    local avail_kb
    avail_kb=$(df --output=avail "$EXTRACT_ROOT" 2>/dev/null | tail -1 | tr -d ' ')
    local avail_gb=$(( avail_kb / 1024 / 1024 ))
    log "Available disk space: ${avail_gb} GB"
    if (( avail_gb < 50 )); then
        log "WARNING: Less than 50 GB free. Extraction may fail."
    fi
}

extract_batch() {
    local batch_name=$1
    shift
    local dirs=("$@")

    if (( ${#dirs[@]} == 0 )); then
        log "No directories specified for batch '$batch_name'"
        return
    fi

    log "=== Extracting batch: $batch_name (${#dirs[@]} directories) ==="
    log "Target dirs: ${dirs[*]}"

    mkdir -p "$EXTRACT_ROOT"
    check_disk_space

    # Build --wildcards arguments
    local wildcards=()
    for d in "${dirs[@]}"; do
        wildcards+=("${INNER_PREFIX}${d}/*")
    done

    # Concatenate all split volumes and selectively extract
    # Using cat + tar to handle multi-volume split archives
    log "Reading tarball volumes from $DATASET_ROOT ..."

    cat "$DATASET_ROOT"/${TAR_PREFIX}[0-9]* | \
        tar -xzf - \
            -C "$EXTRACT_ROOT" \
            --wildcards \
            "${wildcards[@]}" \
            2>&1 | tail -5 || true

    log "Batch '$batch_name' extraction complete."
    du -sh "$EXTRACT_ROOT" 2>/dev/null || true
}

list_tarball_dirs() {
    """List top-level directories available in the tarball (slow, first volume only)."""
    log "Scanning first tarball volume for directory listing..."
    tar -tzf "$DATASET_ROOT/${TAR_PREFIX}000" 2>/dev/null | \
        grep "^${INNER_PREFIX}" | \
        sed "s|^${INNER_PREFIX}||" | \
        cut -d'/' -f1 | \
        sort -u | \
        head -30
    log "(Showing first 30 entries from volume 0 only)"
}

# ── Main ──────────────────────────────────────────────────────────────────
DRY_RUN=false
BATCH=""
LIST_DIRS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        --list)     LIST_DIRS=true; shift ;;
        --batch)    BATCH=$2; shift 2 ;;
        --all)      BATCH="all"; shift ;;
        --smoke)    BATCH="smoke"; shift ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--list] [--batch NAME] [--all] [--smoke]"
            echo ""
            echo "Batches:"
            echo "  smoke  — COCO + ScanNet++ (smoke test)"
            echo "  depth  — All depth/normal datasets"
            echo "  seg    — All segmentation datasets"
            echo "  recon  — Multi-view reconstruction datasets"
            echo "  all    — depth + seg + recon combined"
            echo ""
            echo "Options:"
            echo "  --list      List directories in tarball (first volume)"
            echo "  --dry-run   Show what would be extracted without extracting"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $LIST_DIRS; then
    list_tarball_dirs
    exit 0
fi

if [[ -z "$BATCH" ]]; then
    echo "No batch specified. Use --batch smoke|depth|seg|recon|all or --list"
    exit 1
fi

if $DRY_RUN; then
    log "[DRY RUN] Would extract to: $EXTRACT_ROOT"
fi

case $BATCH in
    smoke)
        extract_batch "smoke" "coco2017" "scannetpp"
        ;;
    depth)
        extract_batch "depth" "${DEPTH_NORMAL_DIRS[@]}"
        ;;
    seg)
        extract_batch "seg" "${SEG_DIRS[@]}"
        ;;
    recon)
        extract_batch "recon" "${RECON_DIRS[@]}"
        ;;
    all)
        # Combine all needed dirs (deduplicating scannetpp)
        ALL_DIRS=()
        declare -A seen
        for d in "${DEPTH_NORMAL_DIRS[@]}" "${SEG_DIRS[@]}" "${RECON_DIRS[@]}"; do
            if [[ -z "${seen[$d]:-}" ]]; then
                ALL_DIRS+=("$d")
                seen[$d]=1
            fi
        done
        extract_batch "all" "${ALL_DIRS[@]}"
        ;;
    *)
        echo "Unknown batch: $BATCH"
        exit 1
        ;;
esac

log "Done. Extracted data at: $EXTRACT_ROOT"
log "Next step: python scripts/build_snvc50m_manifest.py --jsonl-root $DATASET_ROOT"
