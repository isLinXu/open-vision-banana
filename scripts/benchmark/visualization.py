"""PNG artifacts with Pillow only; plotting libraries are optional."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def tensor_rgb_image(tensor):
    pixels = tensor.detach().float().cpu()[0].permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(np.clip(pixels, 0, 1) * 255).astype(np.uint8), "RGB")


def save_preview(original, decoded, directory, title):
    directory.mkdir(parents=True, exist_ok=True)
    rgb = tensor_rgb_image(decoded.rgb)
    rgb.save(str(directory / "prediction-rgb.png"))
    colored = rgb.resize(original.size, resample=Image.NEAREST)
    overlay = Image.blend(original, colored, 0.45)
    overlay.save(str(directory / "overlay.png"))
    # Compact original-grid visualization; inference dimensions remain in JSON.
    tiles = [original.copy(), colored, overlay]
    thumb_w = min(original.width, 320)
    thumb_h = max(1, round(original.height * thumb_w / original.width))
    canvas = Image.new("RGB", (thumb_w * 3, thumb_h + 42), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 3), title, fill="black")
    for index, (tile, label) in enumerate(zip(tiles, ["Input", "Prediction", "Overlay"])):
        canvas.paste(tile.resize((thumb_w, thumb_h)), (index * thumb_w, 42))
        draw.text((index * thumb_w + 5, 23), label, fill="black")
    canvas.save(str(directory / "preview.png"))
    return str(directory / "preview.png")


def save_raw(decoded, directory, task):
    directory.mkdir(parents=True, exist_ok=True)
    array = decoded.values.detach().cpu().numpy()
    np.save(str(directory / "prediction.npy"), array, allow_pickle=False)
    if task == "semantic":
        labels = array[0, 0]
        if labels.max() <= 255:
            Image.fromarray(labels.astype(np.uint8), "L").save(str(directory / "labels.png"))
