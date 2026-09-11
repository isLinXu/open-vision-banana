#!/usr/bin/env python3
"""
Open-Vision-Banana 数据集质量清洗工具

功能：
1. 验证 manifest 中所有文件路径是否存在
2. 检查图像质量（尺寸、格式、有效性）
3. 检测异常目标图像（全黑、全白、低对比度）
4. 生成清洗后的干净 manifest

用法：
    python scripts/clean_dataset.py --manifest output/snvc_v5full_pairs/pairs_5task_v3.jsonl --output output/snvc_v5full_pairs/pairs_5task_v3_clean.jsonl
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_file_exists(path: str) -> bool:
    """检查文件是否存在"""
    return Path(path).exists()


def check_image_valid(path: str, min_size: int = 64) -> tuple[bool, str]:
    """
    检查图像是否有效
    
    Returns:
        (is_valid, error_message)
    """
    try:
        img = Image.open(path)
        img.verify()
        
        # 重新打开以获取尺寸（verify 后需要重新打开）
        img = Image.open(path)
        w, h = img.size
        
        if w < min_size or h < min_size:
            return False, f"尺寸过小: {w}x{h}"
        
        return True, ""
    except Exception as e:
        return False, str(e)


def check_image_quality(path: str) -> dict:
    """
    检查图像质量指标
    
    Returns:
        dict with quality metrics
    """
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        
        # 基本统计
        mean = arr.mean()
        std = arr.std()
        
        # 检测全黑/全白
        is_all_black = mean < 5
        is_all_white = mean > 250
        
        # 检测低对比度
        low_contrast = std < 10
        
        # 检测单通道（灰度）
        channel_std = arr.std(axis=(0, 1))
        is_grayscale = channel_std.max() < 5
        
        return {
            "mean": float(mean),
            "std": float(std),
            "is_all_black": bool(is_all_black),
            "is_all_white": bool(is_all_white),
            "low_contrast": bool(low_contrast),
            "is_grayscale": bool(is_grayscale),
            "size": list(arr.shape[:2]),
        }
    except Exception as e:
        return {"error": str(e)}


def validate_manifest(manifest_path: Path, sample_rate: float = 1.0) -> dict:
    """
    验证 manifest 文件
    
    Args:
        manifest_path: manifest 文件路径
        sample_rate: 采样率（1.0 = 全部检查）
    
    Returns:
        验证结果统计
    """
    logger.info(f"加载 manifest: {manifest_path}")
    
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    logger.info(f"总样本数: {len(rows)}")
    
    # 按任务分组
    task_counts = {}
    for row in rows:
        task = row.get("task", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    
    logger.info(f"任务分布: {task_counts}")
    
    # 验证结果
    results = {
        "total": len(rows),
        "valid": 0,
        "missing_input": 0,
        "missing_target": 0,
        "invalid_image": 0,
        "quality_issues": 0,
        "errors": [],
    }
    
    # 采样检查
    if sample_rate < 1.0:
        import random
        random.seed(42)
        rows = random.sample(rows, int(len(rows) * sample_rate))
        logger.info(f"采样检查: {len(rows)} 样本 ({sample_rate*100:.0f}%)")
    
    for i, row in enumerate(rows):
        input_path = row.get("input", "")
        target_path = row.get("target", "")
        task = row.get("task", "unknown")
        
        # 检查文件存在性
        input_exists = check_file_exists(input_path)
        target_exists = check_file_exists(target_path)
        
        if not input_exists:
            results["missing_input"] += 1
            if len(results["errors"]) < 10:
                results["errors"].append(f"[{i}] 输入缺失: {input_path}")
            continue
        
        if not target_exists:
            results["missing_target"] += 1
            if len(results["errors"]) < 10:
                results["errors"].append(f"[{i}] 目标缺失: {target_path}")
            continue
        
        # 检查图像有效性
        input_valid, input_err = check_image_valid(input_path)
        target_valid, target_err = check_image_valid(target_path)
        
        if not input_valid or not target_valid:
            results["invalid_image"] += 1
            if len(results["errors"]) < 10:
                results["errors"].append(f"[{i}] 图像无效: input={input_err}, target={target_err}")
            continue
        
        # 检查目标图像质量
        quality = check_image_quality(target_path)
        
        has_quality_issue = False
        if quality.get("is_all_black"):
            has_quality_issue = True
        if quality.get("is_all_white"):
            has_quality_issue = True
        if quality.get("low_contrast") and task != "depth":
            # depth 任务允许低对比度
            has_quality_issue = True
        
        if has_quality_issue:
            results["quality_issues"] += 1
            if len(results["errors"]) < 10:
                results["errors"].append(f"[{i}] 质量问题: {quality}")
        
        results["valid"] += 1
        
        if (i + 1) % 1000 == 0:
            logger.info(f"  进度: {i+1}/{len(rows)}")
    
    return results


def clean_manifest(
    manifest_path: Path,
    output_path: Path,
    skip_missing: bool = True,
    skip_invalid: bool = True,
    skip_quality_issues: bool = True,
) -> dict:
    """
    清洗 manifest，生成干净版本
    
    Args:
        manifest_path: 原始 manifest
        output_path: 输出干净 manifest
        skip_missing: 跳过缺失文件
        skip_invalid: 跳过无效图像
        skip_quality_issues: 跳过质量问题
    
    Returns:
        清洗统计
    """
    logger.info(f"清洗 manifest: {manifest_path}")
    
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    logger.info(f"原始样本数: {len(rows)}")
    
    clean_rows = []
    stats = {
        "total": len(rows),
        "kept": 0,
        "skipped_missing": 0,
        "skipped_invalid": 0,
        "skipped_quality": 0,
        "by_task": {},
    }
    
    for i, row in enumerate(rows):
        input_path = row.get("input", "")
        target_path = row.get("target", "")
        task = row.get("task", "unknown")
        
        # 检查文件存在性
        if skip_missing:
            if not check_file_exists(input_path) or not check_file_exists(target_path):
                stats["skipped_missing"] += 1
                continue
        
        # 检查图像有效性
        if skip_invalid:
            input_valid, _ = check_image_valid(input_path)
            target_valid, _ = check_image_valid(target_path)
            if not input_valid or not target_valid:
                stats["skipped_invalid"] += 1
                continue
        
        # 检查目标图像质量
        if skip_quality_issues:
            quality = check_image_quality(target_path)
            
            has_issue = False
            if quality.get("is_all_black"):
                has_issue = True
            if quality.get("is_all_white"):
                has_issue = True
            if quality.get("low_contrast") and task != "depth":
                has_issue = True
            
            if has_issue:
                stats["skipped_quality"] += 1
                continue
        
        clean_rows.append(row)
        stats["kept"] += 1
        stats["by_task"][task] = stats["by_task"].get(task, 0) + 1
        
        if (i + 1) % 1000 == 0:
            logger.info(f"  进度: {i+1}/{len(rows)}, 保留: {stats['kept']}")
    
    # 写入干净 manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in clean_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    logger.info(f"清洗完成: {output_path}")
    logger.info(f"  保留: {stats['kept']}/{stats['total']} ({stats['kept']/stats['total']*100:.1f}%)")
    logger.info(f"  跳过缺失: {stats['skipped_missing']}")
    logger.info(f"  跳过无效: {stats['skipped_invalid']}")
    logger.info(f"  跳过质量: {stats['skipped_quality']}")
    logger.info(f"  任务分布: {stats['by_task']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="原始 manifest 文件")
    parser.add_argument("--output", type=Path, required=True, help="输出干净 manifest")
    parser.add_argument("--validate-only", action="store_true", help="仅验证，不生成清洗后文件")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="采样率 (0.0-1.0)")
    parser.add_argument("--keep-missing", action="store_true", help="保留缺失文件的样本")
    parser.add_argument("--keep-invalid", action="store_true", help="保留无效图像")
    parser.add_argument("--keep-quality-issues", action="store_true", help="保留质量问题的样本")
    
    args = parser.parse_args()
    
    if args.validate_only:
        results = validate_manifest(args.manifest, args.sample_rate)
        logger.info(f"\n验证结果:")
        logger.info(f"  总样本: {results['total']}")
        logger.info(f"  有效: {results['valid']}")
        logger.info(f"  缺失输入: {results['missing_input']}")
        logger.info(f"  缺失目标: {results['missing_target']}")
        logger.info(f"  无效图像: {results['invalid_image']}")
        logger.info(f"  质量问题: {results['quality_issues']}")
        if results["errors"]:
            logger.info(f"\n错误示例 (前 10 个):")
            for err in results["errors"]:
                logger.info(f"  {err}")
    else:
        stats = clean_manifest(
            args.manifest,
            args.output,
            skip_missing=not args.keep_missing,
            skip_invalid=not args.keep_invalid,
            skip_quality_issues=not args.keep_quality_issues,
        )


if __name__ == "__main__":
    main()
