#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Open-Vision-Banana 分步分批提交脚本
# ═══════════════════════════════════════════════════════════════════
#
# 将 v0.3.1 → v0.3.4 的代码变更按主题分批提交至 GitHub。
# 每批一个独立 commit，保持历史清晰可回溯。
#
# 用法:
#   bash scripts/submit_batches.sh          # 执行全部批次
#   bash scripts/submit_batches.sh --dry-run # 仅打印将要执行的命令
#   bash scripts/submit_batches.sh --batch 3 # 仅执行第3批
#
# 安全机制:
#   - 每批提交前打印确认提示
#   - 支持 --dry-run 预览
#   - 出错时立即停止 (set -e)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN=false
BATCH_ONLY=0

# ── Parse args ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --batch) BATCH_ONLY="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

_run() {
    if $DRY_RUN; then
        echo "  [DRY-RUN] $*"
    else
        echo "  → $*"
        "$@"
    fi
}

_confirm() {
    if $DRY_RUN; then
        return 0
    fi
    read -r -p "  Proceed? [Y/n] " response
    if [[ ! "$response" =~ ^[Yy]$ ]] && [[ -n "$response" ]]; then
        echo "  Skipped."
        return 1
    fi
    return 0
}

# ═══════════════════════════════════════════════════════════════════
# 批次定义
# ═══════════════════════════════════════════════════════════════════

batch_1() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Batch 1/6: 清理 .gitignore + 更新本地忽略规则"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  文件: .gitignore"
    echo "  说明: 添加 crablet.db, wandb/, .thumbs/, .benchmarks/"
    _confirm || return

    _run git add .gitignore
    _run git commit -m "chore(gitignore): add runtime artifacts (crablet, wandb, thumbs)"
}

batch_2() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Batch 2/6: 核心代码修复与改进（v0.3.1 遗留）"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  文件: 29 个修改文件，涵盖 core, data, training, eval, inference, export, scripts"
    echo "  说明: 深度映射幂变换、分割解码、训练器、pipeline、评估等核心修复"
    _confirm || return

    # 按模块分步 add 以展示变更范围
    _run git add src/open_vision_banana/core/
    _run git add src/open_vision_banana/data/
    _run git add src/open_vision_banana/training/
    _run git add src/open_vision_banana/evaluation/
    _run git add src/open_vision_banana/inference/
    _run git add src/open_vision_banana/export/
    _run git add src/open_vision_banana/config.py
    _run git add scripts/train.py scripts/evaluate.py scripts/export.py scripts/inference.py scripts/train_smoke.py
    _run git add configs/training_config.yaml
    _run git add pyproject.toml
    _run git add .github/workflows/ci.yml
    _run git add tests/test_data_config.py
    _run git add MODEL_CARD.md

    _run git commit -m "fix+feat(core,training,inference): v0.3.1 core improvements

- depth_mapper: power transform + spectral colormap roundtrip fix
- segmentation: Union-Find vectorization, adaptive color tolerance
- training: TrainerInterface, forward probing, VAE encode fix
- inference: pipeline eval mode, KV-cache, generator signature fix
- evaluation: vectorized IoU, evaluator fixes
- scripts: train.py resume logic, evaluate/export robustness
- CI: extended test matrix, export-check, smoke-train jobs"
}

batch_3() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Batch 3/6: v0.3.2 — 训练基础设施 + Docker + 数据验证"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  文件: Docker生态, FSDP, 推理预设, 数据验证, 实验"
    _confirm || return

    _run git add docker/
    _run git add scripts/train_fsdp.sh
    _run git add scripts/validate_dataset.py
    _run git add scripts/validate_data.py
    _run git add scripts/prepare_hypersim.py
    _run git add scripts/download_checkpoint.py
    _run git add scripts/build_generation_dataset.py
    _run git add configs/optimization_presets.yaml
    _run git add src/open_vision_banana/training/fsdp_utils.py
    _run git add src/open_vision_banana/inference/optimization.py
    _run git add experiments/segmentation_ablation_experiment.py
    _run git add experiments/segmentation_ablation_report.md
    _run git add experiments/segmentation_ablation_results.json
    _run git add experiments/ablation_metrics.png
    _run git add experiments/adaptive_vs_fixed.png
    _run git add experiments/tau_sensitivity.png
    _run git add EXPERIMENTAL_VALIDATION_REPORT.md

    _run git commit -m "feat(v0.3.2): training infrastructure, Docker, data validation

- FSDP: multi-node launch script + checkpoint save/load with CPU offload
- Docker: GPU Dockerfile, docker-compose (dev/training/inference/notebook)
- Inference: optimization presets (balanced/speed/memory/quality/cpu)
- Data validation: validate_dataset.py integrity checks + JSON reports
- Segmentation ablation experiment: 5-stage analysis + adaptive tau"
}

batch_4() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Batch 4/6: v0.3.3 — 推理优化 + 注意力可视化 + Colab Demo"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  文件: optimize_inference, visualize_attention, benchmark_pipeline, notebook"
    _confirm || return

    _run git add scripts/optimize_inference.py
    _run git add scripts/visualize_attention.py
    _run git add scripts/benchmark_pipeline.py
    _run git add notebooks/colab_demo.ipynb

    _run git commit -m "feat(v0.3.3): inference optimization, attention viz, benchmark

- optimize_inference.py: full preset CLI with --list-presets, benchmark, --run
- visualize_attention.py: hook-based FLUX attention extraction + grid overlay
- benchmark_pipeline.py: end-to-end latency/throughput/VRAM benchmark, JSON export
- colab_demo.ipynb: 12-cell interactive demo (depth, normal, segmentation, presets)"
}

batch_5() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Batch 5/6: v0.3.4 — 数据自动化 + 社区生态（HF Hub）"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  文件: download_data, publish_to_hub, model card template, registry"
    _confirm || return

    _run git add scripts/download_data.py
    _run git add scripts/publish_to_hub.py
    _run git add templates/
    _run git add configs/checkpoint_registry.yaml

    _run git commit -m "feat(v0.3.4): data automation + HuggingFace ecosystem

- download_data.py: unified dataset downloader (Hypersim, ScanNet, Cityscapes, SA-1B, RefCOCOg)
- publish_to_hub.py: one-click HF Hub publish with auto-generated model card
- templates/MODEL_CARD_TEMPLATE.md: structured HF model card template
- checkpoint_registry.yaml: expanded to all 5 tasks (depth, normal, semantic, instance, referring)"
}

batch_6() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Batch 6/6: 测试覆盖 + 文档与报告"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  文件: 新增测试文件 + CHANGELOG + README + 成熟度报告"
    _confirm || return

    _run git add tests/test_adaptive_depth.py
    _run git add tests/test_baselines.py
    _run git add tests/test_checkpoint.py
    _run git add tests/test_checkpoint_manager.py
    _run git add tests/test_datasets.py
    _run git add tests/test_deepspeed.py
    _run git add tests/test_evaluator.py
    _run git add tests/test_fsdp.py
    _run git add tests/test_generation_builder.py
    _run git add tests/test_onnx_export.py
    _run git add tests/test_optimization.py
    _run git add tests/test_optimizations_v021.py
    _run git add tests/test_prepare_hypersim.py
    _run git add tests/test_training_features.py
    _run git add tests/test_training_logging.py
    _run git add tests/test_v032_features.py
    _run git add tests/test_v03_coverage.py
    _run git add tests/test_v03_features.py
    _run git add tests/test_validation.py
    _run git add tests/conftest.py

    _run git add CHANGELOG.md
    _run git add README.md
    _run git add MATURITY_IMPROVEMENT_REPORT_v032.md
    _run git add MATURITY_IMPROVEMENT_REPORT_v033.md
    _run git add MATURITY_IMPROVEMENT_REPORT_v034.md

    _run git commit -m "test+docs: 19 new test suites + CHANGELOG + README + maturity reports

- Tests: adaptive_depth, checkpoint_manager, datasets, evaluator, fsdp,
  generation_builder, onnx_export, optimization, prepare_hypersim,
  training_features, training_logging, v032_features, validation
- CHANGELOG: v0.3.2, v0.3.3, v0.3.4 entries with full feature descriptions
- README: inference presets, attention viz, benchmark, data prep, HF publishing
- Maturity reports: v032, v033, v034 with detailed scoring analysis"
}

push_to_github() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Push to GitHub"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  将 6 个 commit 推送至 origin main"
    _confirm || return

    _run git push origin main
    echo ""
    echo "✅ 全部推送完成！"
}

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if $DRY_RUN; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  DRY RUN MODE — 仅预览，不执行任何 git 命令                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
fi

if [[ $BATCH_ONLY -gt 0 ]]; then
    echo ""
    echo "Running batch $BATCH_ONLY only..."
    case $BATCH_ONLY in
        1) batch_1 ;;
        2) batch_2 ;;
        3) batch_3 ;;
        4) batch_4 ;;
        5) batch_5 ;;
        6) batch_6 ;;
        *) echo "Invalid batch number: $BATCH_ONLY"; exit 1 ;;
    esac
else
    batch_1
    batch_2
    batch_3
    batch_4
    batch_5
    batch_6
    push_to_github
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if $DRY_RUN; then
    echo "  DRY RUN 完成。去掉 --dry-run 以实际执行。"
else
    echo "  所有批次已提交！"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
