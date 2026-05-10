from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


WINDOW_MIN = -250.0
WINDOW_MAX = 500.0
WIDE_MIN = -1000.0
WIDE_MAX = 1500.0
BONE_WINDOW_MIN = 150.0
BONE_WINDOW_MAX = 1200.0
TARGET_HEIGHT = 384
TARGET_WIDTH = 256
DEFAULT_VERTEBRAE = ["vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]
DEFAULT_COLORS = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899"]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.head(x)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, list[str], dict]:
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
    vertebrae = list(checkpoint.get("vertebrae", DEFAULT_VERTEBRAE))
    in_channels = int(checkpoint.get("in_channels", 3))
    base_channels = int(checkpoint.get("base_channels", 32))
    model = SmallUNet(in_channels, len(vertebrae), base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vertebrae, checkpoint


def normalize_to_nifti(input_path: Path, work_dir: Path) -> tuple[Path, str]:
    lower_name = input_path.name.lower()
    if lower_name.endswith(".nii") or lower_name.endswith(".nii.gz"):
        target = work_dir / ("input.nii.gz" if lower_name.endswith(".nii.gz") else "input.nii")
        shutil.copy2(input_path, target)
        return target, "nii.gz" if lower_name.endswith(".nii.gz") else "nii"

    image = sitk.ReadImage(str(input_path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D CT image, got dimension={image.GetDimension()}: {input_path}")
    target = work_dir / "input.nii.gz"
    sitk.WriteImage(image, str(target))
    return target, input_path.suffix.lstrip(".").lower() or "unknown"


def resize_array(array: np.ndarray, size: tuple[int, int], source_spacing_hw: tuple[float, float]) -> np.ndarray:
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing((float(source_spacing_hw[1]), float(source_spacing_hw[0])))
    source_h, source_w = array.shape
    target_h, target_w = size
    target_spacing = (
        max(float(source_spacing_hw[1]) * float(source_w) / max(target_w, 1), 1e-6),
        max(float(source_spacing_hw[0]) * float(source_h) / max(target_h, 1), 1e-6),
    )
    resampled = sitk.Resample(
        image,
        [target_w, target_h],
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        target_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampled)


def normalize_window(image_2d: np.ndarray, low: float, high: float) -> np.ndarray:
    clipped = np.clip(image_2d.astype(np.float32), low, high)
    return ((clipped - low) / max(high - low, 1e-6)).clip(0.0, 1.0)


def normalize_bone_image(image_2d: np.ndarray) -> np.ndarray:
    scaled = normalize_window(image_2d, BONE_WINDOW_MIN, BONE_WINDOW_MAX)
    scaled[image_2d < BONE_WINDOW_MIN] = 0.0
    return scaled


def estimate_spine_center_x(volume_xyz: np.ndarray) -> int:
    body = volume_xyz > -500
    coords = np.argwhere(body)
    if coords.size == 0:
        return volume_xyz.shape[0] // 2
    return int(round(float(coords[:, 0].mean())))


def sagittal_from_yz(yz: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(yz.T)).astype(np.float32)


def full_info_slab_projection(volume_xyz: np.ndarray, center_x: int, slab_half_width: int) -> tuple[np.ndarray, dict]:
    x0 = max(0, int(center_x) - int(slab_half_width))
    x1 = min(volume_xyz.shape[0], int(center_x) + int(slab_half_width) + 1)
    slab = volume_xyz[x0:x1, :, :]
    if slab.size == 0:
        safe_x = max(0, min(int(center_x), volume_xyz.shape[0] - 1))
        slab = volume_xyz[safe_x : safe_x + 1, :, :]

    soft_mean = sagittal_from_yz(slab.mean(axis=0))
    wide_mip = sagittal_from_yz(slab.max(axis=0))
    bone_source = slab.copy()
    bone_source[bone_source < BONE_WINDOW_MIN] = -1024.0
    bone_mip = sagittal_from_yz(bone_source.max(axis=0))
    meta = {"slab_range_x": [int(x0), int(x1 - 1)], "source_shape_hw": list(map(int, soft_mean.shape))}
    return np.stack([soft_mean, wide_mip, bone_mip], axis=0), meta


def preprocess_volume(input_nifti: Path, slab_half_width: int = 12) -> tuple[np.ndarray, dict]:
    image = nib.as_closest_canonical(nib.load(str(input_nifti)))
    volume_xyz = image.get_fdata().astype(np.float32)
    spacing_xyz = tuple(float(item) for item in image.header.get_zooms()[:3])
    center_x = estimate_spine_center_x(volume_xyz)
    raw_channels, projection_meta = full_info_slab_projection(volume_xyz, center_x, slab_half_width)
    sagittal_spacing_hw = (spacing_xyz[2], spacing_xyz[1])

    normalized = [
        normalize_window(raw_channels[0], WINDOW_MIN, WINDOW_MAX),
        normalize_window(raw_channels[1], WIDE_MIN, WIDE_MAX),
        normalize_bone_image(raw_channels[2]),
    ]
    resized = [
        resize_array(channel, (TARGET_HEIGHT, TARGET_WIDTH), source_spacing_hw=sagittal_spacing_hw).astype(np.float32)
        for channel in normalized
    ]
    meta = {
        "sagittal_center_x": int(center_x),
        "slab_half_width": int(slab_half_width),
        "canonical_shape_xyz": list(map(int, volume_xyz.shape)),
        "canonical_spacing_xyz": [round(value, 4) for value in spacing_xyz],
        "source_spacing_hw": [round(value, 4) for value in sagittal_spacing_hw],
        "target_shape_hw": [TARGET_HEIGHT, TARGET_WIDTH],
        "channel_names": ["soft_mean", "wide_mip", "bone_mip"],
        **projection_meta,
    }
    return np.stack(resized, axis=0).astype(np.float32), meta


def predict_heatmaps(model: nn.Module, image_chw: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(image_chw[None, :, :, :]).to(device)
    with torch.no_grad():
        heatmaps = torch.sigmoid(model(tensor)).cpu().numpy()[0]
    return heatmaps.astype(np.float32)


def decode_heatmaps(heatmaps: np.ndarray, vertebrae: list[str]) -> list[dict]:
    channels, _height, width = heatmaps.shape
    points = []
    for idx in range(channels):
        flat_idx = int(heatmaps[idx].argmax())
        y = flat_idx // width
        x = flat_idx % width
        points.append({"vertebra": vertebrae[idx], "center_yx": [int(y), int(x)], "confidence": round(float(heatmaps[idx].max()), 6)})
    return points


def map_prediction_to_volume(meta: dict, point: dict) -> dict:
    target_h, target_w = meta["target_shape_hw"]
    source_h, source_w = meta["source_shape_hw"]
    pred_y, pred_x = point["center_yx"]
    source_row = float(pred_y) * float(source_h) / max(float(target_h), 1.0)
    source_col = float(pred_x) * float(source_w) / max(float(target_w), 1.0)
    z_index = int(round((source_h - 1) - source_row))
    y_index = int(round((source_w - 1) - source_col))
    x_index = int(meta["sagittal_center_x"])
    return {
        "vertebra": point["vertebra"],
        "confidence": point["confidence"],
        "pred_center_resized_yx": [int(pred_y), int(pred_x)],
        "source_sagittal_row_col": [round(source_row, 3), round(source_col, 3)],
        "canonical_index_xyz": [int(x_index), int(y_index), int(z_index)],
        "axial_slice_index_z": int(z_index),
    }


def draw_points(image_chw: np.ndarray, points: list[dict], title: str) -> Image.Image:
    base = Image.fromarray((image_chw[1] * 255.0).clip(0, 255).astype(np.uint8)).convert("RGB")
    canvas = Image.new("RGB", (base.width, base.height + 28), color=(18, 18, 18))
    canvas.paste(base, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(255, 255, 255))
    for idx, item in enumerate(points):
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        y, x = item["center_yx"]
        draw.ellipse((x - 4, y + 28 - 4, x + 4, y + 28 + 4), outline=color, width=2)
        draw.text((x + 6, y + 22), item["vertebra"].replace("vertebrae_", ""), fill=color)
    return canvas


def save_l3_slice(input_nifti: Path, l3_result: dict, output_dir: Path, output_format: str) -> Path:
    image = sitk.ReadImage(str(input_nifti))
    array_zyx = sitk.GetArrayFromImage(image)
    slice_index = int(max(0, min(l3_result["axial_slice_index_z"], array_zyx.shape[0] - 1)))
    slice_image = sitk.GetImageFromArray(array_zyx[slice_index : slice_index + 1])
    slice_image.SetSpacing(image.GetSpacing())
    slice_image.SetDirection(image.GetDirection())
    slice_image.SetOrigin(image.TransformIndexToPhysicalPoint((0, 0, slice_index)))
    extension = ".nrrd" if output_format.lower() == "nrrd" else ".nii.gz"
    slice_path = output_dir / f"l3_single_slice{extension}"
    sitk.WriteImage(slice_image, str(slice_path))
    return slice_path


def write_csv(results: list[dict], csv_path: Path) -> None:
    fieldnames = ["vertebra", "confidence", "pred_y", "pred_x", "canonical_x", "canonical_y", "canonical_z", "axial_slice_index_z"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "vertebra": item["vertebra"],
                    "confidence": item["confidence"],
                    "pred_y": item["pred_center_resized_yx"][0],
                    "pred_x": item["pred_center_resized_yx"][1],
                    "canonical_x": item["canonical_index_xyz"][0],
                    "canonical_y": item["canonical_index_xyz"][1],
                    "canonical_z": item["canonical_index_xyz"][2],
                    "axial_slice_index_z": item["axial_slice_index_z"],
                }
            )


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with tempfile.TemporaryDirectory(prefix="vertebra_locator_") as tmp:
        input_nifti, source_format = normalize_to_nifti(args.input.resolve(), Path(tmp))
        model, vertebrae, checkpoint = load_checkpoint(args.model.resolve(), device)
        image_chw, meta = preprocess_volume(input_nifti, slab_half_width=args.slab_half_width)
        heatmaps = predict_heatmaps(model, image_chw, device)
        points = decode_heatmaps(heatmaps, vertebrae)
        locations = [map_prediction_to_volume(meta, point) for point in points]
        l3 = next((item for item in locations if item["vertebra"] == "vertebrae_L3"), None)
        if l3 is None:
            raise RuntimeError("The model did not return vertebrae_L3.")

        preview_path = output_dir / "vertebra_locator_preview.png"
        draw_points(image_chw, points, title=f"Vertebra locator: {args.input.name}").save(preview_path)
        l3_slice_path = save_l3_slice(input_nifti, l3, output_dir, args.slice_format)
        csv_path = output_dir / "vertebra_locations.csv"
        json_path = output_dir / "vertebra_locations.json"
        write_csv(locations, csv_path)
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(args.input.resolve()),
            "source_format": source_format,
            "model_path": str(args.model.resolve()),
            "device": str(device),
            "checkpoint": {
                "vertebrae": vertebrae,
                "best_val_loss": checkpoint.get("best_val_loss"),
                "epochs": checkpoint.get("epochs"),
                "in_channels": checkpoint.get("in_channels"),
                "base_channels": checkpoint.get("base_channels"),
            },
            "preprocess": meta,
            "locations": locations,
            "selected_l3": l3,
            "outputs": {
                "preview_png": str(preview_path),
                "l3_single_slice": str(l3_slice_path),
                "locations_csv": str(csv_path),
                "locations_json": str(json_path),
            },
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight full-info 2D sagittal T12-L5 vertebra locator.")
    parser.add_argument("--input", type=Path, required=True, help="Input 3D CT volume: .nii.gz, .nii, .nrrd, .mha, etc.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "model" / "best_model.pt")
    parser.add_argument("--device", default="")
    parser.add_argument("--slab-half-width", type=int, default=12)
    parser.add_argument("--slice-format", choices=["nii.gz", "nrrd"], default="nii.gz")
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
