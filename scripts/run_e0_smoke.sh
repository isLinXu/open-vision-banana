#!/usr/bin/env bash
# E0 — pipeline smoke on real data, end to end (exp_setting.md §E0).
#
#   download -> prepare (6 sources) -> merge -> strict audit -> data gates
#   -> 500-step train -> run gates -> inference -> paper-benchmark eval -> eval gates
#
# Every stage is idempotent and gated; a failing gate stops the script (exit 1).
# Nothing here changes experiment settings: all values come from exp_setting.md.
#
#   bash scripts/run_e0_smoke.sh                 # full E0 (needs 1 GPU, ~30 GB disk, ~1 h)
#   bash scripts/run_e0_smoke.sh --dry-run       # print every command, run nothing
#   bash scripts/run_e0_smoke.sh --from train    # resume from a stage: download|prepare|merge|train|infer|eval
#   DATA_ROOT=/data/ovb RUN_DIR=output/e0_custom bash scripts/run_e0_smoke.sh
#
# Env knobs (defaults = E0 in exp_setting.md):
#   DATA_ROOT      raw downloads (default data/raw)        PREP_ROOT  prepared manifests (default data/prep)
#   HYPERSIM_SCENES "ai_001_001 ai_001_002"                TARTANAIR_CHUNKS 1    OMNIEDIT_SHARDS 1
#   MODEL_NAME     qwen-image-2.1                          CONFIG     configs/experiments/qwen_image_21_paper_aligned.yaml
#   STEPS 500      SAVE_EVERY 250  EVAL_EVERY 250           PYTHON     python (absolute path recommended for nohup)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-data/raw}"
PREP_ROOT="${PREP_ROOT:-data/prep}"
MERGED="${MERGED:-data/merged/pairs.jsonl}"
FINAL="${FINAL:-data/qwen_baseline/pairs.jsonl}"
RUN_DIR="${RUN_DIR:-output/e0_qwen21_six_smoke_$(date +%Y%m%d_%H%M)}"
MODEL_NAME="${MODEL_NAME:-qwen-image-2.1}"
CONFIG="${CONFIG:-configs/experiments/qwen_image_21_paper_aligned.yaml}"
STEPS="${STEPS:-500}"; SAVE_EVERY="${SAVE_EVERY:-250}"; EVAL_EVERY="${EVAL_EVERY:-250}"
PYTHON="${PYTHON:-python}"
export HYPERSIM_SCENES="${HYPERSIM_SCENES:-ai_001_001 ai_001_002}"
export TARTANAIR_CHUNKS="${TARTANAIR_CHUNKS:-1}"
export OMNIEDIT_SHARDS="${OMNIEDIT_SHARDS:-1}"
TASKS="depth,normal,semantic,instance,referring,generation"
REFS_UMD="${DATA_ROOT}/refer/refcocog/refs(umd).p"

DRY_RUN=0; FROM="download"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --from) FROM="$2"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

STAGES=(download prepare merge train infer eval)
stage_index() { local i; for i in "${!STAGES[@]}"; do [[ "${STAGES[$i]}" == "$1" ]] && { echo "$i"; return; }; done; echo "unknown stage: $1" >&2; exit 2; }
START=$(stage_index "$FROM")
active() { [[ $(stage_index "$1") -ge $START ]]; }

log()  { printf '\033[1;34m[e0]\033[0m %s\n' "$*"; }
run()  { printf '  $ %s\n' "$*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }
gate() { log "gate: $*"; run "$PYTHON" scripts/check_experiment_gates.py "$@"; }

# ---------------------------------------------------------------- 1. download
if active download; then
  log "E0.1 download (HYPERSIM_SCENES='${HYPERSIM_SCENES}', TARTANAIR_CHUNKS=${TARTANAIR_CHUNKS}, OMNIEDIT_SHARDS=${OMNIEDIT_SHARDS})"
  run env DATA_ROOT="$DATA_ROOT" bash scripts/download_datasets.sh --group train
  run env DATA_ROOT="$DATA_ROOT" bash scripts/download_datasets.sh --only refcocog,reasonseg,nyu-depth-v2-splits,nyu-depth-v2-labeled
fi

# ---------------------------------------------------------------- 2. prepare
if active prepare; then
  log "E0.1 prepare (one script per source; manifest-relative pairs.jsonl each)"
  run "$PYTHON" scripts/prepare_hypersim.py --source "${DATA_ROOT}/hypersim" --output "${PREP_ROOT}/hypersim" \
      --tasks depth,normal,semantic,instance
  run "$PYTHON" scripts/prepare_tartanair.py --source "${DATA_ROOT}/tartanair" --output "${PREP_ROOT}/tartanair"
  run "$PYTHON" scripts/build_openvocab_semantic_manifest.py \
      --images "${DATA_ROOT}/ade20k/ADEChallengeData2016/images/training" \
      --labels "${DATA_ROOT}/ade20k/ADEChallengeData2016/annotations/training" \
      --classes configs/ade20k_150_classes.json --source-name ade20k --label-offset 1 --ignore-index -1 \
      --out-dir "${PREP_ROOT}/ade20k" --max-images 2000
  run "$PYTHON" scripts/build_instance_manifest.py \
      --annotations "${DATA_ROOT}/coco2017/annotations/instances_train2017.json" \
      --images-dir "${DATA_ROOT}/coco2017/train2017" --targets-dir "${PREP_ROOT}/inst/targets" \
      --output "${PREP_ROOT}/inst/instance.jsonl" --data-root "${PREP_ROOT}/inst" \
      --exclude-image-ids "$REFS_UMD" --max-rows 5000
  run "$PYTHON" scripts/build_res_manifest.py --annotations-root "${DATA_ROOT}/refer" \
      --image-root "${DATA_ROOT}/coco2017/train2014" --output-dir "${PREP_ROOT}/res"
  run "$PYTHON" scripts/build_generation_replay_manifest.py --kind edit --preset omniedit \
      --records "${DATA_ROOT}/omniedit/data" --images-dir "${PREP_ROOT}/replay/images" \
      --data-root "${PREP_ROOT}/replay" --output "${PREP_ROOT}/replay/omniedit.jsonl" --max-rows 5000
fi

# ---------------------------------------------------------------- 3. merge + strict audit + data gates
if active merge; then
  log "E0.1 merge -> strict audit"
  run "$PYTHON" scripts/merge_multitask_manifest.py \
      --inputs "${PREP_ROOT}/hypersim/pairs.jsonl" "${PREP_ROOT}/tartanair/pairs.jsonl" "${PREP_ROOT}/ade20k/pairs.jsonl" \
               "${PREP_ROOT}/inst/instance.jsonl" "${PREP_ROOT}/res/pairs.jsonl" "${PREP_ROOT}/replay/omniedit.jsonl" \
      --out "$MERGED" --val-percent 2
  run "$PYTHON" scripts/build_qwen_baseline_manifest.py --base-manifest "$MERGED" --output "$FINAL" \
      --data-root "$(dirname "$MERGED")" --protocol strict --exclude-image-ids "$REFS_UMD"
  gate data --merged-audit "${MERGED%.jsonl}.audit.json" --final-manifest "$FINAL" \
       --hypersim-manifest "${PREP_ROOT}/hypersim/manifest.json" --tasks "$TASKS" --model-name "$MODEL_NAME" \
       --json-out "$(dirname "$FINAL")/gates_data.json"
fi

# ---------------------------------------------------------------- 4. train 500 steps + run gates
if active train; then
  log "E0.2 train ${STEPS} steps -> ${RUN_DIR}"
  run mkdir -p "$RUN_DIR"
  run "$PYTHON" scripts/train.py --config "$CONFIG" \
      --pair-manifest "$FINAL" --data-dir "$(dirname "$FINAL")" \
      --steps "$STEPS" --warmup-steps 50 --save-every "$SAVE_EVERY" --eval-every "$EVAL_EVERY" \
      --output-dir "$RUN_DIR" --print-effective-config
  run "$PYTHON" scripts/train.py --config "$CONFIG" \
      --pair-manifest "$FINAL" --data-dir "$(dirname "$FINAL")" \
      --steps "$STEPS" --warmup-steps 50 --save-every "$SAVE_EVERY" --eval-every "$EVAL_EVERY" \
      --output-dir "$RUN_DIR"
  gate run --run-dir "$RUN_DIR" --tasks "$TASKS" --vision-ratio 0.5 --json-out "${RUN_DIR}/gates_run.json"
fi

CKPT="${RUN_DIR}/checkpoint-${STEPS}_final"

# ---------------------------------------------------------------- 5. inference spot checks
if active infer; then
  log "E0.3 inference spot checks from ${CKPT}"
  run "$PYTHON" scripts/inference.py --model-name "$MODEL_NAME" --lora-path "$CKPT" \
      --image-dir "${PREP_ROOT}/hypersim/images" --task depth --output-dir "${RUN_DIR}/infer_depth"
  run "$PYTHON" scripts/inference.py --model-name "$MODEL_NAME" --lora-path "$CKPT" \
      --image-dir "${PREP_ROOT}/hypersim/images" --task normal --output-dir "${RUN_DIR}/infer_normal"
fi

# ---------------------------------------------------------------- 6. paper-benchmark eval (50 samples) + eval gates
if active eval; then
  log "E0.3 paper-benchmark evaluation (max 50 samples per set)"
  run "$PYTHON" scripts/evaluate_paper_benchmarks.py --model-name "$MODEL_NAME" --lora-path "$CKPT" \
      --nyu-root "${DATA_ROOT}/nyu_depth_v2" --refcocog-root "${DATA_ROOT}/refer/refcocog" \
      --reasonseg-root "${DATA_ROOT}/reasonseg" --max-samples 50 \
      --output-json "${RUN_DIR}/eval_smoke.json" --output-csv "${RUN_DIR}/eval_smoke.csv"
  gate eval --eval-json "${RUN_DIR}/eval_smoke.json" --require AbsRel,cIoU --json-out "${RUN_DIR}/gates_eval.json"
fi

log "E0 complete. Artefacts: ${RUN_DIR} (effective_config.json, gates_*.json, eval_smoke.json). Next: exp_setting.md §E1."
