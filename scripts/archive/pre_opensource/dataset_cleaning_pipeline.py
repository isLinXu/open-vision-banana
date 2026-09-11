#!/usr/bin/env python3
"""
数据集清洗与优化流水线

功能：
1. 验证并清洗 manifest
2. 重新生成高质量 Normal 伪标签
3. 改进 Segmentation 伪标签
4. 生成最终干净 manifest

用法：
    python scripts/dataset_cleaning_pipeline.py \
        --manifest output/snvc_v5full_pairs/pairs_5task_v3.jsonl \
        --output-dir output/snvc_v5full_pairs/cleaned/ \
        --normal-dir output/normal_pseudo_v2/ \
        --seg-dir output/seg_pseudo_v2/
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_clean_manifest(manifest: Path, output: Path) -> dict:
    """运行 manifest 清洗"""
    logger.info("=" * 60)
    logger.info("步骤 1: 清洗 manifest")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "clean_dataset.py"),
        "--manifest", str(manifest),
        "--output", str(output),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"清洗失败: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    # 解析输出
    stats = {}
    for line in result.stdout.split("\n"):
        if "保留:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                stats["kept"] = parts[1].strip()
        if "跳过缺失:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                stats["skipped_missing"] = parts[1].strip()
    
    logger.info(f"清洗完成: {output}")
    return {"success": True, "stats": stats}


def run_regen_normals(manifest: Path, output_dir: Path) -> dict:
    """重新生成 Normal 伪标签"""
    logger.info("=" * 60)
    logger.info("步骤 2: 重新生成 Normal 伪标签")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "regen_normal_pseudo.py"),
        "--manifest", str(manifest),
        "--output-dir", str(output_dir),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"生成失败: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    logger.info(f"Normal 伪标签生成完成: {output_dir}")
    return {"success": True}


def run_improve_seg(manifest: Path, output_dir: Path) -> dict:
    """改进 Segmentation 伪标签"""
    logger.info("=" * 60)
    logger.info("步骤 3: 改进 Segmentation 伪标签")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "improve_seg_pseudo.py"),
        "--manifest", str(manifest),
        "--output-dir", str(output_dir),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"生成失败: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    logger.info(f"Seg 伪标签生成完成: {output_dir}")
    return {"success": True}


def generate_final_manifest(
    clean_manifest: Path,
    normal_dir: Path,
    seg_dir: Path,
    output: Path,
) -> dict:
    """生成最终 manifest"""
    logger.info("=" * 60)
    logger.info("步骤 4: 生成最终 manifest")
    logger.info("=" * 60)
    
    # 加载清洗后的 manifest
    rows = []
    with open(clean_manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    logger.info(f"基础样本数: {len(rows)}")
    
    # 添加 Normal 样本
    normal_added = 0
    depth_rows = [r for r in rows if r.get("task") == "depth"]
    
    for row in depth_rows:
        depth_target = row.get("target", "")
        if not depth_target:
            continue
        
        depth_name = Path(depth_target).stem
        normal_path = normal_dir / f"{depth_name}_normal.png"
        
        if normal_path.exists():
            normal_row = row.copy()
            normal_row["task"] = "normal"
            normal_row["target"] = str(normal_path)
            rows.append(normal_row)
            normal_added += 1
    
    logger.info(f"添加 Normal 样本: {normal_added}")
    
    # 添加 Seg 样本
    seg_added = 0
    input_rows = [r for r in rows if r.get("task") == "depth"]  # 使用 depth 的 input
    
    for row in input_rows:
        input_path = row.get("input", "")
        if not input_path:
            continue
        
        input_name = Path(input_path).stem
        seg_path = seg_dir / f"{input_name}_seg.png"
        
        if seg_path.exists():
            seg_row = row.copy()
            seg_row["task"] = "seg"
            seg_row["target"] = str(seg_path)
            rows.append(seg_row)
            seg_added += 1
    
    logger.info(f"添加 Seg 样本: {seg_added}")
    
    # 写入最终 manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # 统计
    task_counts = {}
    for row in rows:
        task = row.get("task", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    
    logger.info(f"最终 manifest: {output}")
    logger.info(f"  总样本: {len(rows)}")
    logger.info(f"  任务分布: {task_counts}")
    
    return {
        "total": len(rows),
        "by_task": task_counts,
        "normal_added": normal_added,
        "seg_added": seg_added,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="原始 manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--normal-dir", type=Path, required=True, help="Normal 输出目录")
    parser.add_argument("--seg-dir", type=Path, required=True, help="Seg 输出目录")
    parser.add_argument("--skip-clean", action="store_true", help="跳过清洗步骤")
    parser.add_argument("--skip-normal", action="store_true", help="跳过 Normal 生成")
    parser.add_argument("--skip-seg", action="store_true", help="跳过 Seg 生成")
    
    args = parser.parse_args()
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.normal_dir.mkdir(parents=True, exist_ok=True)
    args.seg_dir.mkdir(parents=True, exist_ok=True)
    
    # 步骤 1: 清洗 manifest
    clean_manifest_path = args.output_dir / "pairs_clean.jsonl"
    if not args.skip_clean:
        run_clean_manifest(args.manifest, clean_manifest_path)
    else:
        logger.info("跳过清洗步骤")
        clean_manifest_path = args.manifest
    
    # 步骤 2: 重新生成 Normal
    if not args.skip_normal:
        run_regen_normals(clean_manifest_path, args.normal_dir)
    else:
        logger.info("跳过 Normal 生成")
    
    # 步骤 3: 改进 Seg
    if not args.skip_seg:
        run_improve_seg(clean_manifest_path, args.seg_dir)
    else:
        logger.info("跳过 Seg 生成")
    
    # 步骤 4: 生成最终 manifest
    final_manifest = args.output_dir / "pairs_final.jsonl"
    generate_final_manifest(
        clean_manifest_path,
        args.normal_dir,
        args.seg_dir,
        final_manifest,
    )
    
    logger.info("=" * 60)
    logger.info("流水线完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
