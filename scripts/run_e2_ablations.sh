#!/usr/bin/env bash
# E2 + E2b — ablations on top of E1 (exp_setting.md §E2, §E2b), chained so the GPU is never idle.
#
#   E2  : vision_ratio in {0.2, 0.8} x training seed in {42, 43}  -> 4 runs x 10k steps  (E1 @10k is the 0.5 point)
#   E2b : quarter vision data (merge --cap 10000), ratio 0.5, seed 42 -> 1 run x 10k steps
#   each run: train (auto-resume) -> run gate -> eval @10k with the pinned protocol -> eval gate -> generation proxy
#
#   bash scripts/run_e2_ablations.sh --dry-run
#   bash scripts/run_e2_ablations.sh                      # all 5 runs, sequentially, in one GPU job
#   RUNS="vr0.2_s42 vr0.8_s42" bash scripts/run_e2_ablations.sh     # subset (names below)
#   bash scripts/run_e2_ablations.sh --only e2b           # just the data-scale run (does its own merge + audit)
#
# Env knobs: DATA_ROOT PREP_ROOT FINAL (E1 manifest) E1_RUN_DIR (for the 0.5 @10k point) OUT_ROOT (output/e2)
#            STEPS 10000  CAP_E2B 10000  INFER_STEPS 50  GUIDANCE 4.0  EVAL_SEED 42  MAX_EVAL_SAMPLES  PYTHON
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-data/raw}"; PREP_ROOT="${PREP_ROOT:-data/prep}"
FINAL="${FINAL:-data/qwen_baseline/pairs.jsonl}"
E1_RUN_DIR="${E1_RUN_DIR:-}"
OUT_ROOT="${OUT_ROOT:-output/e2}"
CONFIG="${CONFIG:-configs/experiments/qwen_image_21_paper_aligned.yaml}"
MODEL_NAME="${MODEL_NAME:-qwen-image-2.1}"
STEPS="${STEPS:-10000}"; CAP_E2B="${CAP_E2B:-10000}"
PYTHON="${PYTHON:-python}"
INFER_STEPS="${INFER_STEPS:-50}"; GUIDANCE="${GUIDANCE:-4.0}"; EVAL_SEED="${EVAL_SEED:-42}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
TASKS="depth,normal,semantic,instance,referring,generation"
REFS_UMD="${DATA_ROOT}/refer/refcocog/refs(umd).p"
RUNS="${RUNS:-vr0.2_s42 vr0.2_s43 vr0.8_s42 vr0.8_s43 e2b}"

EVAL_FLAGS=(--num-inference-steps "$INFER_STEPS" --guidance-scale "$GUIDANCE" --seed "$EVAL_SEED" --resolution 512 --native-resolution)
[[ -n $MAX_EVAL_SAMPLES ]] && EVAL_FLAGS+=(--max-samples "$MAX_EVAL_SAMPLES")
EVAL_ROOTS=(--nyu-root "${DATA_ROOT}/nyu_depth_v2" --diode-root "${DATA_ROOT}/diode" --refcocog-root "${DATA_ROOT}/refer/refcocog" --reasonseg-root "${DATA_ROOT}/reasonseg")
[[ -d "${DATA_ROOT}/cityscapes" ]] && EVAL_ROOTS+=(--cityscapes-root "${DATA_ROOT}/cityscapes")

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --only) RUNS="$2"; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac; shift
done
log()  { printf '\033[1;36m[e2]\033[0m %s\n' "$*"; }
run()  { printf '  $ %s\n' "$*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }
gate() { log "gate: $*"; run "$PYTHON" scripts/check_experiment_gates.py "$@"; }
[[ -n ${WANDB_API_KEY:-} ]] && WANDB_FLAG=(--wandb) || WANDB_FLAG=()

train_eval() {  # name manifest vision_ratio seed
  local name="$1" manifest="$2" ratio="$3" seed="$4"
  local run_dir="${OUT_ROOT}/${name}"
  run mkdir -p "$run_dir"
  local resume=()
  local latest; latest=$(ls -d "${run_dir}"/checkpoint-[0-9]* 2>/dev/null | grep -vE '_best$|_final$' | sed -E 's/.*checkpoint-//' | sort -n | tail -1 || true)
  [[ -n $latest ]] && resume=(--resume "${run_dir}/checkpoint-${latest}") && log "${name}: resuming from checkpoint-${latest}"
  log "${name}: train ${STEPS} steps (vision_ratio=${ratio}, seed=${seed}, manifest=${manifest})"
  local cmd=("$PYTHON" scripts/train.py --config "$CONFIG" --pair-manifest "$manifest" --data-dir "$(dirname "$manifest")"
             --vision-ratio "$ratio" --seed "$seed" --steps "$STEPS" --save-every 2500 --eval-every 2500 --output-dir "$run_dir"
             ${resume[@]+"${resume[@]}"} ${WANDB_FLAG[@]+"${WANDB_FLAG[@]}"})
  if [[ $DRY_RUN -eq 1 ]]; then printf '  $ %s 2>&1 | tee -a %s\n' "${cmd[*]}" "${run_dir}/train.log"; else "${cmd[@]}" 2>&1 | tee -a "${run_dir}/train.log"; fi
  gate run --run-dir "$run_dir" --tasks "$TASKS" --vision-ratio "$ratio" --json-out "${run_dir}/gates_run.json"
  local ckpt="${run_dir}/checkpoint-${STEPS}_best"; [[ -d $ckpt || $DRY_RUN -eq 1 ]] || ckpt="${run_dir}/checkpoint-${STEPS}_final"
  if [[ ! -f "${run_dir}/eval_${STEPS}.json" || $DRY_RUN -eq 1 ]]; then
    run "$PYTHON" scripts/evaluate_paper_benchmarks.py --model-name "$MODEL_NAME" --lora-path "$ckpt" "${EVAL_FLAGS[@]}" "${EVAL_ROOTS[@]}" \
        --output-json "${run_dir}/eval_${STEPS}.json" --output-csv "${run_dir}/eval_${STEPS}.csv"
  fi
  gate eval --eval-json "${run_dir}/eval_${STEPS}.json" --require AbsRel,cIoU --json-out "${run_dir}/gates_eval.json"
  # this ablation is *about* generation preservation: proxy for every run (base reference is shared with E1)
  [[ -f "output/reference/genai_base_${MODEL_NAME}.json" && $DRY_RUN -eq 0 ]] || \
    run "$PYTHON" scripts/evaluate_generation_preservation.py --model-name "$MODEL_NAME" --dataset-root "${DATA_ROOT}/genai_bench" \
        --num-inference-steps "$INFER_STEPS" --seed "$EVAL_SEED" --output-json "output/reference/genai_base_${MODEL_NAME}.json"
  run "$PYTHON" scripts/evaluate_generation_preservation.py --model-name "$MODEL_NAME" --lora-path "$ckpt" --dataset-root "${DATA_ROOT}/genai_bench" \
      --num-inference-steps "$INFER_STEPS" --seed "$EVAL_SEED" --output-json "${run_dir}/gen_${STEPS}.json"
}

for name in $RUNS; do
  case "$name" in
    vr*_s*)
      ratio="${name#vr}"; ratio="${ratio%%_s*}"; seed="${name##*_s}"
      train_eval "$name" "$FINAL" "$ratio" "$seed" ;;
    e2b)
      log "E2b: quarter vision data -> merge --cap ${CAP_E2B} -> strict audit -> data gate"
      MERGED_Q="data/merged_e2b/pairs.jsonl"; FINAL_Q="data/qwen_baseline_e2b/pairs.jsonl"
      INPUTS=("${PREP_ROOT}/hypersim/pairs.jsonl" "${PREP_ROOT}/tartanair/pairs.jsonl" "${PREP_ROOT}/ade20k/pairs.jsonl"
              "${PREP_ROOT}/inst/instance.jsonl" "${PREP_ROOT}/res/pairs.jsonl" "${PREP_ROOT}/replay/omniedit.jsonl")
      [[ -f "${PREP_ROOT}/cocostuff/pairs.jsonl" || $DRY_RUN -eq 1 ]] && INPUTS+=("${PREP_ROOT}/cocostuff/pairs.jsonl")
      [[ -f "${PREP_ROOT}/inst_lvis/instance.jsonl" || $DRY_RUN -eq 1 ]] && INPUTS+=("${PREP_ROOT}/inst_lvis/instance.jsonl")
      run "$PYTHON" scripts/merge_multitask_manifest.py --inputs "${INPUTS[@]}" --out "$MERGED_Q" --cap "$CAP_E2B" --val-percent 2
      run "$PYTHON" scripts/build_qwen_baseline_manifest.py --base-manifest "$MERGED_Q" --output "$FINAL_Q" \
          --data-root "$(dirname "$MERGED_Q")" --protocol strict --exclude-image-ids "$REFS_UMD"
      gate data --merged-audit "${MERGED_Q%.jsonl}.audit.json" --final-manifest "$FINAL_Q" \
           --hypersim-manifest "${PREP_ROOT}/hypersim/manifest.json" --tasks "$TASKS" --model-name "$MODEL_NAME" \
           --json-out "$(dirname "$FINAL_Q")/gates_data.json"
      train_eval "e2b_cap${CAP_E2B}_s42" "$FINAL_Q" 0.5 42 ;;
    *) echo "unknown run name: $name (use vr<ratio>_s<seed> or e2b)" >&2; exit 2 ;;
  esac
done

if [[ -n $E1_RUN_DIR ]]; then
  log "0.5 point: E1 @${STEPS} from ${E1_RUN_DIR} (evaluated by run_e1_full.sh --from eval with EVAL_CKPTS=${STEPS} if missing)"
  [[ -f "${E1_RUN_DIR}/eval_${STEPS}.json" ]] || log "  missing ${E1_RUN_DIR}/eval_${STEPS}.json -> run: RUN_DIR=${E1_RUN_DIR} EVAL_CKPTS=${STEPS} bash scripts/run_e1_full.sh --from eval --to eval"
fi
log "E2/E2b complete. Decision rule: a ratio wins only if >=2 of {Cityscapes mIoU, NYU delta1, DIODE normal mean} beat E1 by more than the E1 noise floor with no regression; otherwise 'no significant difference'."
