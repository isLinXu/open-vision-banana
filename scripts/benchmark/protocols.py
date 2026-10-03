"""Explicit depth masks; main's legacy NYU crop must not be called standard.

NYU Eigen indices: official Marigold src/dataset/nyu_dataset.py, 2026-10-03.
KITTI Garg ratios: official ZoeDepth zoedepth/utils/misc.py, 2026-10-03.
This adapter deliberately leaves the existing project's evaluator untouched.
"""
import torch


def depth_protocol(protocol, pred, gt, valid_mask=None):
    valid = torch.isfinite(gt) & (gt > 1e-3)
    if valid_mask is not None:
        valid &= valid_mask.bool()
    if protocol == "full":
        return pred, gt, valid, {"name": "full", "crop": None, "cap": None}
    if protocol == "nyu":
        if tuple(gt.shape[-2:]) != (480, 640):
            raise ValueError("NYU Eigen requires original 480x640 GT; scaled/cropped GT needs a separate protocol")
        cap = 10.0
        crop = [45, 471, 41, 601]
        name = "nyu_eigen_480x640"
    elif protocol == "kitti":
        height, width = gt.shape[-2:]
        cap = 80.0
        crop = [int(0.40810811 * height), int(0.99189189 * height),
                int(0.03594771 * width), int(0.96405229 * width)]
        name = "kitti_garg"
    elif protocol == "eth3d":
        # Preserve the project's cap-only option without asserting official parity.
        cap, crop, name = 80.0, None, "project_eth3d_cap_only_unverified"
    else:
        raise ValueError("Unknown depth protocol")
    valid &= gt < cap
    if crop is not None:
        crop_mask = torch.zeros_like(valid)
        y0, y1, x0, x1 = crop
        crop_mask[..., y0:y1, x0:x1] = True
        valid &= crop_mask
    return pred.clamp(1e-3, cap), gt, valid, {"name": name, "crop": crop,
                                                         "cap": cap, "min_gt_exclusive": 1e-3,
                                                         "max_gt_exclusive": cap}
