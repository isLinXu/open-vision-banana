#!/usr/bin/env bash
# One-command dataset fetcher driven by configs/datasets/sources.tsv.
#
#   bash scripts/download_datasets.sh --list
#   bash scripts/download_datasets.sh --group train                 # everything needed to start training
#   bash scripts/download_datasets.sh --only hypersim,coco2017-images,coco2017-instances,lvis-v1,refcoco,omniedit
#   bash scripts/download_datasets.sh --task depth --dry-run
#   DATA_ROOT=/data/ovb HYPERSIM_SCENES="ai_001_001 ai_001_002" bash scripts/download_datasets.sh --only hypersim
#
# Adding a dataset = adding one line to configs/datasets/sources.tsv (see its header).
#
# Environment:
#   DATA_ROOT         where everything lands (default: data/raw)
#   HF_ENDPOINT       Hugging Face mirror (default: https://hf-mirror.com)
#   MODELSCOPE_TOKEN  for gated ModelScope datasets (OpenDataLab/* need a logged-in account)
#   HF_TOKEN          for gated HF datasets
#   HYPERSIM_SCENES   space-separated scene ids (default: 8 scenes, ~25 GB)
#   TARTANAIR_CHUNKS  number of chunk tarballs (default: 2)
#   OMNIEDIT_SHARDS   number of parquet shards (default: 4, ~20 GB)
#   SKIP_EXTRACT=1    download archives only
set -euo pipefail
# Works with macOS bash 3.2 (no mapfile / associative arrays) and GNU bash.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${REGISTRY:-$ROOT/configs/datasets/sources.tsv}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/raw}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HYPERSIM_SCENES="${HYPERSIM_SCENES:-ai_001_001 ai_001_002 ai_001_003 ai_001_004 ai_002_001 ai_002_002 ai_003_001 ai_004_001}"
TARTANAIR_CHUNKS="${TARTANAIR_CHUNKS:-2}"
OMNIEDIT_SHARDS="${OMNIEDIT_SHARDS:-4}"
PYTHON="${PYTHON:-python3}"

DRY_RUN=0; LIST=0; ONLY=""; GROUP=""; TASK=""
usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --only) ONLY="$2"; shift ;;
    --group) GROUP="$2"; shift ;;
    --task) TASK="$2"; shift ;;
    --root) DATA_ROOT="$2"; shift ;;
    --registry) REGISTRY="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
  shift
done

log() { printf '\033[1;34m[datasets]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[datasets]\033[0m %s\n' "$*" >&2; }
run() { if [[ $DRY_RUN -eq 1 ]]; then printf '  $ %s\n' "$*"; else "$@"; fi; }

# ---------------------------------------------------------------- registry
# Fields: name group tasks source locator include dest post size license note
read_registry() {  # drop comments, blank lines and an optional literal header row
  grep -v '^[[:space:]]*#' "$REGISTRY" | grep -v '^[[:space:]]*$' | grep -v $'^name\tgroup\t'
}

selected() {  # name group tasks -> 0 if this row is selected by the CLI filters
  local name="$1" group="$2" tasks="$3"
  if [[ -n $ONLY ]]; then
    [[ ",$ONLY," == *",$name,"* ]] || return 1
  fi
  if [[ -n $GROUP ]]; then
    [[ "$group" == "$GROUP" ]] || return 1
  fi
  if [[ -n $TASK ]]; then
    [[ ",$tasks," == *",$TASK,"* ]] || return 1
  fi
  return 0
}

expand_include() {  # macro-aware expansion; prints one pattern per line
  local spec="$1" item
  IFS=';' read -r -a items <<< "$spec"
  for item in "${items[@]}"; do
    case "$item" in
      -) ;;
      @hypersim_scenes) for s in $HYPERSIM_SCENES; do printf 'raw/%s.zip\n' "$s"; done ;;
      @tartanair_chunks) for ((i = 0; i < TARTANAIR_CHUNKS; i++)); do printf 'chunk__part%06d.tar.gz\n' "$i"; done ;;
      @omniedit_shards) for ((i = 0; i < OMNIEDIT_SHARDS; i++)); do printf 'data/train-%05d-of-00571.parquet\n' "$i"; done ;;
      *) printf '%s\n' "$item" ;;
    esac
  done
}

# ---------------------------------------------------------------- fetchers
have() { command -v "$1" >/dev/null 2>&1; }

fetch_ms() {  # repo dest patterns...
  local repo="$1" dest="$2"; shift 2
  local -a patterns=("$@")
  if have modelscope; then
    local -a args=(download --dataset "$repo" --local_dir "$dest")
    [[ ${#patterns[@]} -gt 0 ]] && args+=(--include "${patterns[@]}")
    run modelscope "${args[@]}"
  else
    local py_patterns="None"
    if [[ ${#patterns[@]} -gt 0 ]]; then
      py_patterns="[$(printf "'%s'," "${patterns[@]}")]"
    fi
    run "$PYTHON" - "$repo" "$dest" "$py_patterns" <<'PY'
import sys
try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    sys.exit("modelscope is not installed: pip install modelscope  (or: uv pip install modelscope)")
repo, dest, patterns = sys.argv[1], sys.argv[2], eval(sys.argv[3])
snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=patterns)
PY
  fi
}

fetch_hf() {  # repo dest patterns...
  local repo="$1" dest="$2"; shift 2
  local -a patterns=("$@")
  local cli
  if have hf; then cli=hf; elif have huggingface-cli; then cli=huggingface-cli; else
    echo "huggingface CLI not found: pip install -U huggingface_hub" >&2; return 1; fi
  local -a args=(download "$repo" --repo-type dataset --local-dir "$dest")
  for p in "${patterns[@]}"; do args+=(--include "$p"); done
  run "$cli" "${args[@]}"
}

fetch_url() {  # url dest
  local url="$1" dest="$2" file
  # basename of the URL path; Wayback Machine URLs embed the original URL, whose basename is kept.
  file="$dest/$(basename "${url%%\?*}")"
  if [[ -s $file && $DRY_RUN -eq 0 ]]; then log "  exists: $file"; return 0; fi
  # --retry-all-errors covers "connection reset"; --speed-* aborts stalls (<1 KB/s for 60 s) so a dead mirror fails fast
  run curl -L --fail --retry 5 --retry-delay 5 --retry-all-errors --retry-connrefused -C - \
    --speed-time 60 --speed-limit 1000 -o "$file" "$url"
}

coco2014_links() {  # dest: expose 2017 images under the 2014 names the RefCOCO refer files use
  local dest="$1"
  run "$PYTHON" - "$dest" <<'PY'
import os, sys
from pathlib import Path
root = Path(sys.argv[1])
made = 0
for split2017, split2014 in (("train2017", "train2014"), ("val2017", "train2014")):
    src = root / split2017
    if not src.is_dir():
        continue
    dst = root / split2014
    dst.mkdir(exist_ok=True)
    for img in src.iterdir():
        if img.suffix.lower() != ".jpg":
            continue
        link = dst / f"COCO_train2014_{int(img.stem):012d}.jpg"
        if not link.exists():
            # Hard link: resolves inside train2014/, which build_res_manifest's
            # traversal guard requires (a symlink to ../train2017 would be rejected).
            try:
                os.link(img, link)
            except OSError:  # cross-device: fall back to a relative symlink
                os.symlink(os.path.relpath(img, dst), link)
            made += 1
print(f"coco2014_links: {made} new links under {root / 'train2014'}")
PY
}

post_process() {  # post dest   (post may chain steps with '+', e.g. unzip_each+coco2014_links)
  local spec="$1" dest="$2" post
  [[ ${SKIP_EXTRACT:-0} -eq 1 || $spec == none ]] && return 0
  IFS='+' read -r -a steps <<< "$spec"
  for post in "${steps[@]}"; do post_step "$post" "$dest" || return 1; done
}

refer_convert() {  # dest split_by : <root>/refer_parquet/<name> -> <root>/refer/<name>
  local dest="$1" split_by="$2" name out
  name="$(basename "$dest")"
  out="$(dirname "$(dirname "$dest")")/refer/$name"
  run "$PYTHON" "$ROOT/scripts/convert_refer_parquet.py" --input "$dest" --output "$out" --split-by "$split_by" --name "$name"
}

post_step() {
  local post="$1" dest="$2"
  case "$post" in
    coco2014_links) coco2014_links "$dest" ;;
    refer_umd) refer_convert "$dest" umd ;;
    refer_unc) refer_convert "$dest" unc ;;
    nyu_mat)
      local splits=() ; [[ -f "$dest/splits.mat" ]] && splits=(--splits "$dest/splits.mat")
      [[ ${#splits[@]} -eq 0 ]] && warn "   splits.mat not found in $dest; frames go to all/ (fetch nyu-depth-v2-splits first)"
      run "$PYTHON" "$ROOT/scripts/convert_eval_sets.py" nyu-mat --mat "$dest/nyu_depth_v2_labeled.mat" "${splits[@]}" --out "$dest" ;;
    genai_prompts) run "$PYTHON" "$ROOT/scripts/convert_eval_sets.py" genai-bench --parquet "$dest" --out "$dest" ;;
    unzip|unzip_each)
      for z in "$dest"/*.zip "$dest"/raw/*.zip; do
        [[ -e $z ]] || continue
        local marker="$z.extracted"
        [[ -e $marker && $DRY_RUN -eq 0 ]] && continue
        run unzip -q -o "$z" -d "$dest" && run touch "$marker"
      done ;;
    un7z|un7z_each)
      have 7z || { warn "   7z not found (brew install p7zip / apt install p7zip-full); archives kept unextracted"; return 0; }
      for a in "$dest"/*.7z; do
        [[ -e $a ]] || continue
        local marker="$a.extracted"
        [[ -e $marker && $DRY_RUN -eq 0 ]] && continue
        run 7z x -y -o"$dest" "$a" >/dev/null && run touch "$marker"
      done ;;
    untar|untar_each)
      for t in "$dest"/*.tar "$dest"/*.tar.gz "$dest"/*.tgz; do
        [[ -e $t ]] || continue
        local marker="$t.extracted"
        [[ -e $marker && $DRY_RUN -eq 0 ]] && continue
        run tar -xf "$t" -C "$dest" && run touch "$marker"
      done ;;
    *) warn "unknown post step '$post' (skipped)" ;;
  esac
}

# ---------------------------------------------------------------- main
if [[ ! -f $REGISTRY ]]; then echo "registry not found: $REGISTRY" >&2; exit 1; fi

if [[ $LIST -eq 1 ]]; then
  printf '%-22s %-7s %-34s %-7s %-9s %s\n' NAME GROUP TASKS SOURCE SIZE_GB LOCATOR
  while IFS=$'\t' read -r name group tasks source locator include dest post size license note; do
    selected "$name" "$group" "$tasks" || continue
    printf '%-22s %-7s %-34s %-7s %-9s %s\n' "$name" "$group" "$tasks" "$source" "$size" "$locator"
  done < <(read_registry)
  exit 0
fi

mkdir -p "$DATA_ROOT"
n_done=0; n_manual=0; n_fail=0
while IFS=$'\t' read -r name group tasks source locator include dest post size license note; do
  selected "$name" "$group" "$tasks" || continue
  target="$DATA_ROOT/$dest"
  log "== $name  [$group | $tasks | $source | ~${size} GB | $license]"
  [[ -n $note ]] && log "   $note"
  if [[ $source == manual ]]; then
    warn "   requires a login; download from $locator into $target then re-run with --only $name --dry-run to see post steps"
    n_manual=$((n_manual + 1)); continue
  fi
  run mkdir -p "$target"
  patterns=()
  while IFS= read -r line; do [[ -n $line ]] && patterns+=("$line"); done < <(expand_include "$include")
  if { case "$source" in
         # ${arr[@]+"${arr[@]}"}: bash 3.2 + set -u treats an empty array as unbound
         ms)  fetch_ms "$locator" "$target" ${patterns[@]+"${patterns[@]}"} ;;
         hf)  fetch_hf "$locator" "$target" ${patterns[@]+"${patterns[@]}"} ;;
         url) fetch_url "$locator" "$target" ;;
         *)   echo "unknown source '$source'" >&2; false ;;
       esac; } && post_process "$post" "$target"; then
    n_done=$((n_done + 1))
  else
    warn "   FAILED: $name"; n_fail=$((n_fail + 1))
  fi
done < <(read_registry)

log "done: $n_done fetched, $n_manual need manual download, $n_fail failed  (root: $DATA_ROOT)"
[[ $n_fail -eq 0 ]]
