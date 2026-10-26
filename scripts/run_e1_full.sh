#!/usr/bin/env bash
# E1 — paper-aligned six-task run, end to end (exp_setting.md §E1), gated and resumable.
#
#   download -> prepare (6 sources) -> merge -> strict audit -> data gates
#   -> 30k-step train (auto-resume) -> run gates -> eval 5k/15k/30k + no-CFG + seed-43 -> eval gates
#
# Designed so a GPU node is never left idle:
#   * data stages are CPU/network bound: run them on a login/CPU node (or before the GPU allocation)
#     with `--to merge`, then start the GPU job with `--from train`;
#   * the run directory is created *before* any redirect (the exp_setting snippet redirected into a
#     directory that did not exist yet -> training never started);
#   * training auto-resumes from the newest checkpoint in RUN_DIR if the job is restarted;
#   * `--from eval` / `--watch` evaluates checkpoints as they appear (on EVAL_GPU if set), so
#     evaluation overlaps training instead of waiting for it;
#   * W&B is only enabled when WANDB_API_KEY is set (otherwise `wandb login` would block the job).
#
#   bash scripts/run_e1_full.sh --dry-run                 # print every command
#   bash scripts/run_e1_full.sh --to merge                # CPU node: data only, ends with the data gate
#   bash scripts/run_e1_full.sh --from train              # GPU node: train + eval (+ gates)
#   bash scripts/run_e1_full.sh --from eval --watch       # second GPU: evaluate checkpoints as they land
#   bash scripts/run_e1_full.sh --from train --to train   # train only
#
# Env knobs (defaults = exp_setting.md E1):
#   DATA_ROOT data/raw   PREP_ROOT data/prep   MERGED data/merged/pairs.jsonl   FINAL data/qwen_baseline/pairs.jsonl
#   RUN_DIR output/e1_qwen21_paper_aligned_<ts>   (pass the same RUN_DIR to resume / watch)
#   MODEL_NAME qwen-image-2.1   CONFIG configs/experiments/qwen_image_21_paper_aligned.yaml
#   STEPS 30000  SAVE_EVERY 2500  EVAL_EVERY 2500   CAP 40000  VAL_PERCENT 2
#   HYPERSIM_SCENES (60 scenes)  TARTANAIR_ENVS (6 envs)  OMNIEDIT_SHARDS 8
#   INFER_STEPS 50  GUIDANCE 4.0  EVAL_SEED 42   EVAL_CKPTS "5000 15000 30000"   EVAL_GPU (CUDA index for eval)
#   MAX_EVAL_SAMPLES (unset = full benchmarks)   GEN_PROXY_SAMPLES 200
#   EVAL_DTYPE bfloat16   GEN_REF_DIR output/reference/genai_base_<model>_s<seed>_n<steps>
#   PYTHON python (absolute path recommended)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-data/raw}"
PREP_ROOT="${PREP_ROOT:-data/prep}"
MERGED="${MERGED:-data/merged/pairs.jsonl}"
FINAL="${FINAL:-data/qwen_baseline/pairs.jsonl}"
RUN_DIR="${RUN_DIR:-output/e1_qwen21_paper_aligned_$(date +%Y%m%d_%H%M)}"
MODEL_NAME="${MODEL_NAME:-qwen-image-2.1}"
CONFIG="${CONFIG:-configs/experiments/qwen_image_21_paper_aligned.yaml}"
STEPS="${STEPS:-30000}"; SAVE_EVERY="${SAVE_EVERY:-2500}"; EVAL_EVERY="${EVAL_EVERY:-2500}"
CAP="${CAP:-40000}"; VAL_PERCENT="${VAL_PERCENT:-2}"
PYTHON="${PYTHON:-python}"
INFER_STEPS="${INFER_STEPS:-50}"; GUIDANCE="${GUIDANCE:-4.0}"; EVAL_SEED="${EVAL_SEED:-42}"
EVAL_CKPTS="${EVAL_CKPTS:-5000 15000 30000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
GEN_PROXY_SAMPLES="${GEN_PROXY_SAMPLES:-200}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
GEN_REF_DIR="${GEN_REF_DIR:-output/reference/genai_base_${MODEL_NAME}_s${EVAL_SEED}_n${INFER_STEPS}}"
TASKS="depth,normal,semantic,instance,referring,generation"
REFS_UMD="${DATA_ROOT}/refer/refcocog/refs(umd).p"
export HYPERSIM_SCENES="${HYPERSIM_SCENES:-$(for a in 1 2 3 4 5 6 7 8 9 10; do for b in 1 2 3 4 5 6; do printf 'ai_%03d_%03d ' "$a" "$b"; done; done)}"
export TARTANAIR_ENVS="${TARTANAIR_ENVS:-japanesealley carwelding westerndesert office2 seasidetown office}"
export OMNIEDIT_SHARDS="${OMNIEDIT_SHARDS:-8}"

EVAL_FLAGS=(--num-inference-steps "$INFER_STEPS" --guidance-scale "$GUIDANCE" --seed "$EVAL_SEED" --resolution 512 --native-resolution --dtype "$EVAL_DTYPE")
[[ -n $MAX_EVAL_SAMPLES ]] && EVAL_FLAGS+=(--max-samples "$MAX_EVAL_SAMPLES")
EVAL_ROOTS=(--nyu-root "${DATA_ROOT}/nyu_depth_v2" --diode-root "${DATA_ROOT}/diode" --eth3d-root "${DATA_ROOT}/eth3d"
            --cityscapes-root "${DATA_ROOT}/cityscapes" --saco-root "${DATA_ROOT}/saco_gold"
            --refcocog-root "${DATA_ROOT}/refer/refcocog" --reasonseg-root "${DATA_ROOT}/reasonseg")

DRY_RUN=0; FROM="download"; TO="eval"; WATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --from) FROM="$2"; shift ;;
    --to) TO="$2"; shift ;;
    --watch) WATCH=1 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

STAGES=(download prepare merge train eval)
stage_index() { local i; for i in "${!STAGES[@]}"; do [[ "${STAGES[$i]}" == "$1" ]] && { echo "$i"; return; }; done; echo "unknown stage: $1" >&2; exit 2; }
START=$(stage_index "$FROM"); END=$(stage_index "$TO")
active() { local i; i=$(stage_index "$1"); [[ $i -ge $START && $i -le $END ]]; }

log()  { printf '\033[1;35m[e1]\033[0m %s\n' "$*"; }
run()  { printf '  $ %s\n' "$*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }
gate() { log "gate: $*"; run "$PYTHON" scripts/check_experiment_gates.py "$@"; }

# ---------------------------------------------------------------- 1. download (CPU / network)
if active download; then
  log "E1.1 download: $(echo "$HYPERSIM_SCENES" | wc -w | tr -d ' ') Hypersim scenes, TartanAir '${TARTANAIR_ENVS}', OMNIEDIT_SHARDS=${OMNIEDIT_SHARDS} (~150 GB raw; run on a CPU node)"
  run env DATA_ROOT="$DATA_ROOT" bash scripts/download_datasets.sh --group train
  run env DATA_ROOT="$DATA_ROOT" bash scripts/download_datasets.sh --only omniedit
  run env DATA_ROOT="$DATA_ROOT" bash scripts/download_datasets.sh --group eval
fi

# ---------------------------------------------------------------- 2. prepare (CPU)
if active prepare; then
  log "E1.1 prepare (full tier; semantic variants 2)"
  run "$PYTHON" scripts/prepare_hypersim.py --source "${DATA_ROOT}/hypersim" --output "${PREP_ROOT}/hypersim" \
      --tasks depth,normal,semantic,instance --semantic-variants 2
  run "$PYTHON" scripts/prepare_tartanair.py --source "${DATA_ROOT}/tartanair" --output "${PREP_ROOT}/tartanair"
  run "$PYTHON" scripts/build_openvocab_semantic_manifest.py \
      --images "${DATA_ROOT}/ade20k/ADEChallengeData2016/images/training" \
      --labels "${DATA_ROOT}/ade20k/ADEChallengeData2016/annotations/training" \
      --classes configs/ade20k_150_classes.json --source-name ade20k --label-offset 1 --ignore-index -1 \
      --out-dir "${PREP_ROOT}/ade20k" --variants 2
  if [[ -d "${DATA_ROOT}/coco_stuff/train2017" ]]; then
    run "$PYTHON" scripts/build_openvocab_semantic_manifest.py \
        --images "${DATA_ROOT}/coco2017/train2017" --labels "${DATA_ROOT}/coco_stuff/train2017" \
        --classes configs/coco_stuff_182_labels.json --source-name coco_stuff --label-offset 0 --ignore-index 255 \
        --out-dir "${PREP_ROOT}/cocostuff" --variants 2 --max-images 10000
  fi
  run "$PYTHON" scripts/build_instance_manifest.py \
      --annotations "${DATA_ROOT}/coco2017/annotations/instances_train2017.json" \
      --images-dir "${DATA_ROOT}/coco2017/train2017" --targets-dir "${PREP_ROOT}/inst/targets" \
      --output "${PREP_ROOT}/inst/instance.jsonl" --data-root "${PREP_ROOT}/inst" \
      --exclude-image-ids "$REFS_UMD" --min-instances 1
  if [[ -f "${DATA_ROOT}/lvis/lvis_v1_train.json" ]]; then
    run "$PYTHON" scripts/build_instance_manifest.py --format lvis \
        --annotations "${DATA_ROOT}/lvis/lvis_v1_train.json" --images-dir "${DATA_ROOT}/coco2017" \
        --targets-dir "${PREP_ROOT}/inst_lvis/targets" --output "${PREP_ROOT}/inst_lvis/instance.jsonl" \
        --data-root "${PREP_ROOT}/inst_lvis" --exclude-image-ids "$REFS_UMD" --min-instances 1
  fi
  run "$PYTHON" scripts/build_res_manifest.py --annotations-root "${DATA_ROOT}/refer" \
      --image-root "${DATA_ROOT}/coco2017/train2014" --output-dir "${PREP_ROOT}/res"
  run "$PYTHON" scripts/build_generation_replay_manifest.py --kind edit --preset omniedit \
      --records "${DATA_ROOT}/omniedit/data" --images-dir "${PREP_ROOT}/replay/images" \
      --data-root "${PREP_ROOT}/replay" --output "${PREP_ROOT}/replay/omniedit.jsonl"
fi

# ---------------------------------------------------------------- 3. merge + strict audit + data gates
if active merge; then
  log "E1.1 merge (--cap ${CAP} --val-percent ${VAL_PERCENT}) -> strict audit -> data gates"
  INPUTS=("${PREP_ROOT}/hypersim/pairs.jsonl" "${PREP_ROOT}/tartanair/pairs.jsonl" "${PREP_ROOT}/ade20k/pairs.jsonl"
          "${PREP_ROOT}/inst/instance.jsonl" "${PREP_ROOT}/res/pairs.jsonl" "${PREP_ROOT}/replay/omniedit.jsonl")
  [[ -f "${PREP_ROOT}/cocostuff/pairs.jsonl" || $DRY_RUN -eq 1 ]] && INPUTS+=("${PREP_ROOT}/cocostuff/pairs.jsonl")
  [[ -f "${PREP_ROOT}/inst_lvis/instance.jsonl" || $DRY_RUN -eq 1 ]] && INPUTS+=("${PREP_ROOT}/inst_lvis/instance.jsonl")
  run "$PYTHON" scripts/merge_multitask_manifest.py --inputs "${INPUTS[@]}" --out "$MERGED" --cap "$CAP" --val-percent "$VAL_PERCENT"
  run "$PYTHON" scripts/build_qwen_baseline_manifest.py --base-manifest "$MERGED" --output "$FINAL" \
      --data-root "$(dirname "$MERGED")" --protocol strict --exclude-image-ids "$REFS_UMD"
  gate data --merged-audit "${MERGED%.jsonl}.audit.json" --final-manifest "$FINAL" \
       --hypersim-manifest "${PREP_ROOT}/hypersim/manifest.json" --tasks "$TASKS" --model-name "$MODEL_NAME" \
       --json-out "$(dirname "$FINAL")/gates_data.json"
  log "data stages done. Start the GPU job with: RUN_DIR=<dir> bash $0 --from train"
fi

# ---------------------------------------------------------------- 4. train (GPU), auto-resume
if active train; then
  run mkdir -p "$RUN_DIR"          # BEFORE any redirect into it
  log "E1.2 train ${STEPS} steps -> ${RUN_DIR}"
  WANDB_FLAG=()
  if [[ -n ${WANDB_API_KEY:-} ]]; then WANDB_FLAG=(--wandb); else log "WANDB_API_KEY unset -> W&B off (an interactive wandb login would block the node)"; fi
  RESUME_FLAG=()
  latest=$(ls -d "${RUN_DIR}"/checkpoint-[0-9]* 2>/dev/null | grep -vE '_best$|_final$' | sed -E 's/.*checkpoint-//' | sort -n | tail -1 || true)
  if [[ -n $latest ]]; then
    RESUME_FLAG=(--resume "${RUN_DIR}/checkpoint-${latest}")
    log "found checkpoint-${latest}; resuming full state"
  else
    run "$PYTHON" scripts/train.py --config "$CONFIG" --pair-manifest "$FINAL" --data-dir "$(dirname "$FINAL")" \
        --steps "$STEPS" --save-every "$SAVE_EVERY" --eval-every "$EVAL_EVERY" --output-dir "$RUN_DIR" --print-effective-config
    [[ $DRY_RUN -eq 1 ]] || cp "$(dirname "$FINAL")/pairs.audit.json" "${MERGED%.jsonl}.audit.json" "${PREP_ROOT}/hypersim/manifest.json" "$RUN_DIR"/ 2>/dev/null || true
  fi
  # foreground here; launch THIS SCRIPT under nohup/tmux. train.log lives in the run dir.
  TRAIN_CMD=("$PYTHON" scripts/train.py --config "$CONFIG" --pair-manifest "$FINAL" --data-dir "$(dirname "$FINAL")"
             --steps "$STEPS" --save-every "$SAVE_EVERY" --eval-every "$EVAL_EVERY" --output-dir "$RUN_DIR"
             ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} ${WANDB_FLAG[@]+"${WANDB_FLAG[@]}"})
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  $ %s 2>&1 | tee -a %s\n' "${TRAIN_CMD[*]}" "${RUN_DIR}/train.log"
  else
    "${TRAIN_CMD[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
  fi
  gate run --run-dir "$RUN_DIR" --tasks "$TASKS" --vision-ratio 0.5 --json-out "${RUN_DIR}/gates_run.json"
fi

# ---------------------------------------------------------------- 5. eval (GPU; may run alongside training with --watch)
eval_ckpt() {  # <step> [extra flags...] ; writes eval_<step>[_tag].json
  local step="$1"; shift
  local tag="${TAG:-}"
  local expected_guidance="$GUIDANCE" expected_seed="$EVAL_SEED"
  [[ $tag == "_nocfg" ]] && expected_guidance=0.0
  [[ $tag == "_seed43" ]] && expected_seed=43
  local ckpt="${RUN_DIR}/checkpoint-${step}"
  local out="${RUN_DIR}/eval_${step}${tag}.json"
  run "$PYTHON" scripts/check_eval_checkpoint.py --checkpoint "$ckpt"
  if [[ -f $out && $DRY_RUN -eq 0 ]]; then
    run "$PYTHON" scripts/check_eval_checkpoint.py --checkpoint "$ckpt" --eval-json "$out" \
        --expected-steps "$INFER_STEPS" --expected-guidance "$expected_guidance" \
        --expected-seed "$expected_seed" --expected-dtype "$EVAL_DTYPE"
    log "skip ${out} (exists)"
  else
  run env ${EVAL_GPU:+CUDA_VISIBLE_DEVICES="$EVAL_GPU"} "$PYTHON" scripts/evaluate_paper_benchmarks.py \
      --model-name "$MODEL_NAME" --lora-path "$ckpt" "${EVAL_FLAGS[@]}" "$@" "${EVAL_ROOTS[@]}" \
      --output-json "$out" --output-csv "${out%.json}.csv"
    run "$PYTHON" scripts/check_eval_checkpoint.py --checkpoint "$ckpt" --eval-json "$out" \
        --expected-steps "$INFER_STEPS" --expected-guidance "$expected_guidance" \
        --expected-seed "$expected_seed" --expected-dtype "$EVAL_DTYPE"
  fi
  gate eval --eval-json "$out" --profile e1 --json-out "${RUN_DIR}/gates_eval_${step}${tag}.json"
}

if active eval; then
  if [[ $WATCH -eq 1 ]]; then
    log "E1.3 watch mode: evaluating checkpoints ${EVAL_CKPTS} as they appear (EVAL_GPU=${EVAL_GPU:-same})"
    for step in $EVAL_CKPTS; do
      while [[ $DRY_RUN -eq 0 && ! -f "${RUN_DIR}/checkpoint-${step}/metadata.json" ]]; do sleep 120; done
      eval_ckpt "$step"
    done
  else
    log "E1.3 evaluate ${EVAL_CKPTS} (pinned protocol: ${INFER_STEPS} steps, CFG ${GUIDANCE}, seed ${EVAL_SEED}, native)"
    for step in $EVAL_CKPTS; do eval_ckpt "$step"; done
  fi
  last=$(echo "$EVAL_CKPTS" | awk '{print $NF}')
  log "E1.3 extra readings on ${last}: no-CFG and seed 43 (noise floor)"
  TAG="_nocfg" eval_ckpt "$last" --guidance-scale 0.0
  TAG="_seed43" eval_ckpt "$last" --seed 43
  log "E1.3 generation proxy: fixed base reference, then checkpoint-${last} (online)"
  run env ${EVAL_GPU:+CUDA_VISIBLE_DEVICES="$EVAL_GPU"} "$PYTHON" scripts/build_generation_reference.py \
      --model-name "$MODEL_NAME" --prompts-root "${DATA_ROOT}/genai_bench" --output-dir "$GEN_REF_DIR" \
      --num-inference-steps "$INFER_STEPS" --guidance-scale "$GUIDANCE" --seed "$EVAL_SEED" \
      --dtype "$EVAL_DTYPE" --max-samples "$GEN_PROXY_SAMPLES"
  run env ${EVAL_GPU:+CUDA_VISIBLE_DEVICES="$EVAL_GPU"} "$PYTHON" scripts/evaluate_generation_preservation.py --model-name "$MODEL_NAME" \
      --lora-path "${RUN_DIR}/checkpoint-${last}" --dataset-root "$GEN_REF_DIR" --dtype "$EVAL_DTYPE" \
      --num-inference-steps "$INFER_STEPS" --guidance-scale "$GUIDANCE" --seed "$EVAL_SEED" \
      --max-samples "$GEN_PROXY_SAMPLES" \
      --output-json "${RUN_DIR}/gen_${last}.json"
  run "$PYTHON" scripts/check_eval_checkpoint.py --checkpoint "${RUN_DIR}/checkpoint-${last}" --eval-json "${RUN_DIR}/gen_${last}.json" \
      --expected-steps "$INFER_STEPS" --expected-guidance "$GUIDANCE" --expected-seed "$EVAL_SEED" --expected-dtype "$EVAL_DTYPE"
  gate generation --eval-json "${RUN_DIR}/gen_${last}.json" --reference-dir "$GEN_REF_DIR" \
      --json-out "${RUN_DIR}/gates_generation_${last}.json"
  log "E1 complete. Artefacts in ${RUN_DIR}: effective_config.json, gates_*.json, eval_*.json, gen_*.json, train.log. Next: exp_setting.md §E2."
fi
