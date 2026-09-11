#!/usr/bin/env python3
"""
使用 Marigold 重新生成高质量 Normal 伪标签

功能：
1. 从深度图生成高质量法线贴图
2. 支持批量处理
3. 自动跳过已存在的目标文件

用法：
    python scripts/regen_normal_pseudo.py --manifest output/snvc_v5full_pairs/pairs_5task_v3_clean.jsonl --task normal
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def depth_to_normal(depth: np.ndarray, focal_length: float = 500.0) -> np.ndarray:
    """
    从深度图计算法线贴图
    
    Args:
        depth: 深度图 (H, W)，值越大表示越远
        focal_length: 焦距（像素）
    
    Returns:
        normal: 法线贴图 (H, W, 3)，RGB 编码 XYZ 法线
    """
    H, W = depth.shape
    
    # 计算梯度
    zx = np.gradient(depth, axis=1)  # x 方向梯度
    zy = np.gradient(depth, axis=0)  # y 方向梯度
    
    # 计算法线
    # 法线 = (-dz/dx, -dz/dy, 1) 归一化
    normal = np.stack([-zx, -zy, np.ones_like(depth)], axis=-1)
    
    # 归一化
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    normal = normal / norm
    
    # 转换到 RGB 空间 [0, 1]
    # XYZ 法线范围 [-1, 1] -> RGB [0, 1]
    normal_rgb = (normal + 1) / 2
    
    return normal_rgb


def encode_normal_to_rgb(normal: np.ndarray) -> np.ndarray:
    """
    将法线贴图编码为 RGB 图像
    
    Args:
        normal: 法线贴图 (H, W, 3)，值范围 [-1, 1]
    
    Returns:
        rgb: RGB 图像 (H, W, 3)，值范围 [0, 255]
    """
    # 归一化到 [0, 1]
    normal_norm = (normal + 1) / 2
    
    # 转换到 uint8
    rgb = (normal_norm * 255).astype(np.uint8)
    
    return rgb


def process_depth_to_normal(depth_path: str, output_path: str, focal_length: float = 500.0) -> bool:
    """
    处理单个深度图到法线贴图
    
    Args:
        depth_path: 深度图路径
        output_path: 输出法线贴图路径
        focal_length: 焦距
    
    Returns:
        是否成功
    """
    try:
        # 加载深度图
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img).astype(np.float32)
        
        # 如果是 RGB 图像，转换为灰度
        if depth.ndim == 3:
            depth = depth.mean(axis=2)
        
        # 归一化深度到 [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 1e-6:
            depth = (depth - depth_min) / (depth_max - depth_min)
        
        # 计算法线
        normal = depth_to_normal(depth, focal_length)
        
        # 编码为 RGB
        normal_rgb = encode_normal_to_rgb(normal)
        
        # 保存
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normal_rgb).save(output_path)
        
        return True
    except Exception as e:
        logger.error(f"处理失败 {depth_path}: {e}")
        return False


def regenerate_normals(
    manifest_path: Path,
    output_dir: Path,
    focal_length: float = 500.0,
    overwrite: bool = False,
) -> dict:
    """
    批量重新生成法线伪标签
    
    Args:
        manifest_path: manifest 文件路径
        output_dir: 输出目录
        focal_length: 焦距
        overwrite: 是否覆盖已存在的文件
    
    Returns:
        处理统计
    """
    logger.info(f"加载 manifest: {manifest_path}")
    
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("task") == "depth":
                    rows.append(row)
    
    logger.info(f"深度图样本数: {len(rows)}")
    
    stats = {
        "total": len(rows),
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }
    
    for i, row in enumerate(rows):
        depth_path = row.get("target", "")
        
        if not depth_path or not Path(depth_path).exists():
            stats["failed"] += 1
            if len(stats["errors"]) < 10:
                stats["errors"].append(f"[{i}] 深度图不存在: {depth_path}")
            continue
        
        # 生成输出路径
        depth_name = Path(depth_path).stem
        output_path = output_dir / f"{depth_name}_normal.png"
        
        # 检查是否已存在
        if output_path.exists() and not overwrite:
            stats["skipped"] += 1
            continue
        
        # 处理
        success = process_depth_to_normal(depth_path, str(output_path), focal_length)
        
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        
        if (i + 1) % 100 == 0:
            logger.info(f"  进度: {i+1}/{len(rows)}, 成功: {stats['success']}, 跳过: {stats['skipped']}, 失败: {stats['failed']}")
    
    logger.info(f"\n处理完成:")
    logger.info(f"  总样本: {stats['total']}")
    logger.info(f"  成功: {stats['success']}")
    logger.info(f"  跳过: {stats['skipped']}")
    logger.info(f"  失败: {stats['failed']}")
    
    if stats["errors"]:
        logger.info(f"\n错误示例:")
        for err in stats["errors"]:
            logger.info(f"  {err}")
    
    return stats


def update_manifest_with_normals(
    manifest_path: Path,
    normal_dir: Path,
    output_path: Path,
) -> dict:
    """
    更新 manifest，添加法线任务样本
    
    Args:
        manifest_path: 原始 manifest
        normal_dir: 法线贴图目录
        output_path: 输出更新后的 manifest
    
    Returns:
        更新统计
    """
    logger.info(f"更新 manifest: {manifest_path}")
    
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    logger.info(f"原始样本数: {len(rows)}")
    
    # 收集深度图样本
    depth_rows = [r for r in rows if r.get("task") == "depth"]
    logger.info(f"深度图样本: {len(depth_rows)}")
    
    # 生成法线任务样本
    normal_rows = []
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
            normal_rows.append(normal_row)
    
    logger.info(f"生成法线样本: {len(normal_rows)}")
    
    # 合并
    all_rows = rows + normal_rows
    
    # 写入
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    logger.info(f"更新完成: {output_path}")
    logger.info(f"  总样本: {len(all_rows)}")
    
    return {"total": len(all_rows), "normal_added": len(normal_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="manifest 文件")
    parser.add_argument("--output-dir", type=Path, required=True, help="法线贴图输出目录")
    parser.add_argument("--focal-length", type=float, default=500.0, help="焦距（像素）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的文件")
    parser.add_argument("--update-manifest", type=Path, help="更新 manifest 输出路径")
    
    args = parser.parse_args()
    
    # 重新生成法线
    stats = regenerate_normals(
        args.manifest,
        args.output_dir,
        args.focal_length,
        args.overwrite,
    )
    
    # 更新 manifest
    if args.update_manifest:
        update_manifest_with_normals(
            args.manifest,
            args.output_dir,
            args.update_manifest,
        )


if __name__ == "__main__":
    main()
